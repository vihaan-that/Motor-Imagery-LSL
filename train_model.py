#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Model for Motor Imagery Classification

This script extracts the training components from datamotor14.py
and saves a trained model to be used with lsl_classifier.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# Import functions from datamotor14.py
# You may need to adjust these imports based on your local file structure
from datamotor14 import (
    load_features, 
    create_training_pipeline, 
    train_and_evaluate
)

def main():
    """Train and save the model for real-time classification."""
    # Step 1: Load preprocessed features
    print("Loading features from preprocessed data...")
    feature_file = 'path/to/your/features.pkl'  # Update with your actual file path
    
    try:
        X, y, feature_names = load_features(feature_file)
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
    except FileNotFoundError:
        print(f"Error: Feature file '{feature_file}' not found.")
        print("Please run datamotor14.py first to extract features, or provide the correct path.")
        return
    
    # Step 2: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 3: Create and train the pipeline
    print("Training model...")
    pipeline = create_training_pipeline()
    
    # Step 4: Train and evaluate the model
    train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
    
    # Step 5: Retrain on all data for final model
    print("Training final model on all data...")
    pipeline.fit(X, y)
    
    # Step 6: Save the trained model
    model_path = 'motor_imagery_model.joblib'
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Model saved successfully!")
    
    print("\nYou can now use this model with lsl_classifier.py for real-time classification.")

if __name__ == '__main__':
    main()
