#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSL Classifier for Motor Imagery

This script streams data from an Emotiv headset via LSL,
processes it in real-time, and classifies motor imagery tasks.
"""

import time
import numpy as np
import mne
from mne import create_info
from mne.io import RawArray
from pylsl import StreamInlet, resolve_stream
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import joblib
import os
import matplotlib.pyplot as plt
from collections import deque
import threading

# Constants
SRATE = 128  # Sample rate in Hz
BUFFER_DURATION = 4.0  # Buffer duration in seconds
BUFFER_SIZE = int(SRATE * BUFFER_DURATION)  # Buffer size in samples
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
N_CHANNELS = len(CHANNELS)
CLASS_NAMES = ['left_fist', 'both_feet']

# Feature extraction classes
class EnsureFloat64(BaseEstimator, TransformerMixin):
    """Ensure data is float64 for compatibility with other transformers."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.astype(np.float64)

class TangentSpaceFeatures(BaseEstimator, TransformerMixin):
    """Extract tangent space features from EEG epochs."""
    def __init__(self, n_channels=14):
        self.n_channels = n_channels
        self.cov = Covariances(estimator='oas')
        self.ts = TangentSpace(metric='riemann')
        
    def fit(self, X, y=None):
        # X shape: (n_epochs, n_channels, n_times)
        covs = self.cov.fit_transform(X)
        self.ts.fit(covs)
        return self
    
    def transform(self, X):
        covs = self.cov.transform(X)
        ts_features = self.ts.transform(covs)
        return ts_features

class SpectralFeatures(BaseEstimator, TransformerMixin):
    """Extract spectral features (PSD) from EEG epochs."""
    def __init__(self, sfreq=128, bands=[(8, 12), (12, 30)]):
        self.sfreq = sfreq
        self.bands = bands  # mu and beta bands
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X shape: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, _ = X.shape
        n_bands = len(self.bands)
        
        features = np.zeros((n_epochs, n_channels * n_bands))
        
        for i in range(n_epochs):
            for j in range(n_channels):
                signal = X[i, j, :]
                freqs, psd = welch(signal, fs=self.sfreq, nperseg=self.sfreq)
                
                for k, (fmin, fmax) in enumerate(self.bands):
                    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                    band_power = np.mean(psd[idx_band])
                    features[i, j * n_bands + k] = band_power
        
        return features

class FeatureCombiner(BaseEstimator, TransformerMixin):
    """Combine multiple feature sets."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.concatenate(X, axis=1) if isinstance(X, list) else X

def create_model_pipeline():
    """Create a pipeline for processing and classifying EEG data."""
    # Feature extraction pipeline
    tangent_space = Pipeline([
        ('ensure_float64', EnsureFloat64()),
        ('tangent_space', TangentSpaceFeatures(n_channels=N_CHANNELS))
    ])
    
    spectral = Pipeline([
        ('ensure_float64', EnsureFloat64()),
        ('spectral', SpectralFeatures(sfreq=SRATE))
    ])
    
    # Feature union pipeline
    feature_extraction = Pipeline([
        ('features', FeatureCombiner())
    ])
    
    # Classification pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def preprocess_buffer(buffer):
    """Preprocess the EEG buffer."""
    # Convert to MNE Raw object
    data = np.array(buffer).T  # Shape: (n_channels, n_samples)
    info = create_info(ch_names=CHANNELS, sfreq=SRATE, ch_types=['eeg'] * N_CHANNELS)
    raw = RawArray(data, info)
    
    # Apply bandpass filter
    raw.filter(4, 40, method='fir', phase='zero-double', verbose=False)
    
    # Extract epoch (the entire buffer is treated as one epoch)
    data = raw.get_data()
    # Reshape to match training data format: (n_epochs, n_channels, n_samples)
    epoch = data.reshape(1, N_CHANNELS, -1)
    
    return epoch

def extract_features(epoch):
    """
    Extract features from the preprocessed epoch.
    
    This function should extract features in the same way as in datamotor14.py
    to ensure consistency between training and inference.
    """
    # Initialize feature extractors
    ts_extractor = TangentSpaceFeatures()
    spec_extractor = SpectralFeatures()
    
    # Extract features
    ts_features = ts_extractor.fit_transform(epoch)
    spec_features = spec_extractor.fit_transform(epoch)
    
    # Combine features (ensure this matches the feature combination in datamotor14.py)
    features = np.concatenate([ts_features, spec_features], axis=1)
    
    return features

def load_model(model_path='motor_imagery_model.joblib'):
    """
    Load a pretrained model.
    Returns the trained pipeline.
    """
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        pipeline = joblib.load(model_path)
        return pipeline
    
    # If no pretrained model, alert the user
    print(f"Error: No pretrained model found at {model_path}")
    print("Please run train_model.py first to train and save the model.")
    print("Exiting...")
    exit(1)

def update_plot(probabilities, ax, line_left, line_feet, past_probs):
    """Update the real-time plot with new probabilities."""
    past_probs.append(probabilities)
    if len(past_probs) > 50:  # Keep only the last 50 predictions
        past_probs.popleft()
    
    data = np.array(past_probs)
    x = np.arange(len(data))
    
    line_left.set_data(x, data[:, 0])
    line_feet.set_data(x, data[:, 1])
    
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

def main():
    # Set up visualization
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1)
    ax.set_title('Real-time Motor Imagery Classification')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    line_left, = ax.plot([], [], 'b-', label='Left Fist')
    line_feet, = ax.plot([], [], 'r-', label='Both Feet')
    ax.legend()
    plt.tight_layout()
    
    past_probs = deque(maxlen=50)  # Store past probabilities for plotting
    
    # Load the pretrained model from datamotor14.py
    model_path = 'motor_imagery_model.joblib'  # Path to the saved model
    pipeline = load_model(model_path)
    
    # Look for an EEG stream on the network
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    
    # Create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    
    # Buffer to store EEG data
    buffer = []
    
    try:
        while True:
            # Get a new sample
            sample, timestamp = inlet.pull_sample()
            sample = [el / 1000000 for el in sample]  # Convert to microvolts
            
            # Add to buffer
            buffer.append(sample)
            
            # Once buffer is full, process and classify
            if len(buffer) >= BUFFER_SIZE:
                # Keep only the most recent data (sliding window)
                buffer = buffer[-BUFFER_SIZE:]
                
                # Preprocess the buffer
                epoch = preprocess_buffer(buffer)
                
                # Extract features
                features = extract_features(epoch)
                
                # Classify
                probabilities = pipeline.predict_proba(features)[0]
                prediction = pipeline.predict(features)[0]
                
                # Print results
                print(f"Prediction: {CLASS_NAMES[prediction]}")
                print(f"Probabilities: Left Fist: {probabilities[0]:.2f}, Both Feet: {probabilities[1]:.2f}")
                
                # Update visualization
                update_plot(probabilities, ax, line_left, line_feet, past_probs)
                
                # Short delay to avoid overwhelming the CPU
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStream ended by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close()

if __name__ == '__main__':
    main()
