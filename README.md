# BCI-MOTORIMAGERY: Motor Imagery Classification with LSL & Emotiv

## Overview

This project provides a framework for classifying motor imagery tasks (left fist vs. both feet) in real-time using EEG data streamed from an Emotiv headset via Lab Streaming Layer (LSL). 

The system builds on the `emotiv-lsl` streaming server and adds a classification pipeline for real-time motor imagery detection.

## System Architecture

The system follows this workflow:

1. **Data Collection** - EEG data is collected using an Emotiv headset
2. **Feature Extraction** - Both tangent space (Riemannian geometry) and spectral features are extracted
3. **Model Training** - A classifier (Random Forest) is trained on the extracted features
4. **Real-time Classification** - Streaming data is processed and classified in real-time

## Files

### train_model.py

This script extracts and processes features from previously collected EEG data, then trains a classification model.

**Key Functions:**
- `main()`: Orchestrates the feature loading, model training, and model saving process

**Usage:**
1. Update the feature file path in the script
2. Run: `python train_model.py`

### lsl_classifier.py

This script performs real-time classification of EEG data streamed from an Emotiv headset via LSL.

**Key Components:**
- Feature extraction classes (TangentSpaceFeatures, SpectralFeatures)
- Data preprocessing pipeline
- Real-time visualization
- LSL data streaming interface

**Key Functions:**
- `preprocess_buffer()`: Converts raw EEG buffer to preprocessed epoch
- `extract_features()`: Extracts features from preprocessed epoch
- `load_model()`: Loads the trained classifier
- `update_plot()`: Updates the real-time visualization
- `main()`: Orchestrates the real-time classification process

**Usage:**
1. Train a model using train_model.py
2. Run: `python lsl_classifier.py`

## Requirements for Classification System

- Python 3.7+
- MNE-Python
- PyLSL
- scikit-learn
- pyRiemann
- matplotlib
- numpy
- scipy
- joblib

## Installation of Classification Components

```bash
# After setting up the emotiv-lsl base, install additional dependencies
pip install mne scikit-learn pyriemann matplotlib numpy scipy joblib
```

## Classification System Workflow

### 1. Data Collection and Preprocessing

If you don't have preprocessed data yet:
1. Start the emotiv-lsl server as described above
2. Collect EEG data using the Emotiv headset while performing motor imagery tasks
3. Preprocess and extract features using the functions in datamotor14.py

### 2. Train the Model

1. Update the feature file path in train_model.py
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. The trained model will be saved as "motor_imagery_model.joblib"

### 3. Real-time Classification

1. Start the emotiv-lsl server in one terminal:
   ```bash
   python -m pipenv run python main.py
   ```
2. In another terminal, run the real-time classifier:
   ```bash
   python lsl_classifier.py
   ```
3. View the real-time classification results and probability plot

## Implementation Details

### Feature Extraction

Two types of features are extracted from the EEG data:

1. **Tangent Space Features**:
   - Uses Riemannian geometry to capture spatial relationships between EEG channels
   - Computes covariance matrices and projects to tangent space

2. **Spectral Features**:
   - Extracts power spectral density in relevant frequency bands (mu: 8-12 Hz, beta: 12-30 Hz)
   - Important for motor imagery classification as these bands show characteristic changes during motor imagery

### Classification

The system uses a **Random Forest Classifier** due to its:
- Robustness to overfitting
- Ability to handle high-dimensional feature spaces
- Good performance on non-linear classification problems

### Real-time Processing

- Uses a sliding window approach to continuously process incoming EEG data
- Window size: 4 seconds (512 samples at 128 Hz)
- Preprocessing includes bandpass filtering (4-40 Hz)
- Real-time visualization shows classification probabilities over time

## Troubleshooting

### Common Issues:

1. **No LSL stream found**:
   - Ensure the Emotiv headset is connected and streaming data
   - Check if the LSL outlet is properly configured

2. **Model not found**:
   - Make sure to run train_model.py before lsl_classifier.py
   - Verify that "motor_imagery_model.joblib" exists in the project directory

3. **Feature mismatch**:
   - Ensure feature extraction is consistent between training and inference
   - Check that the same preprocessing steps are applied in both stages

## Extending the System

### Adding New Features:

1. Create a new feature extractor class that implements fit/transform methods
2. Add the feature extractor to the feature extraction pipeline
3. Update both training and real-time classification code

### Supporting Additional Motor Imagery Tasks:

1. Collect and preprocess data for the new tasks
2. Update CLASS_NAMES variable in lsl_classifier.py
3. Retrain the model with the expanded dataset

## References

- Lab Streaming Layer (LSL): https://github.com/sccn/labstreaminglayer
- MNE-Python: https://mne.tools/
- pyRiemann: https://pyriemann.readthedocs.io/
- Emotiv Documentation: https://emotiv.gitbook.io/
## Project Components

The system consists of two main components:

1. **emotiv-lsl** - Base LSL server for Emotiv EPOC X headset
2. **Classification Pipeline** - Processing and machine learning components for motor imagery detection

## Setting Up emotiv-lsl Base Server

The base component is an LSL server for Emotiv EPOC X (original code derived from [CyKit](https://github.com/CymatiCorp/CyKit)).

### Dependencies for emotiv-lsl

```bash
pip install pipenv
python -m pipenv install
```

### Starting the LSL Stream

1. Disable the motion data in Emotiv app settings  
2. Connect dongle, turn on the headset, wait for the light from two indicators
3. Start the LSL server:
   ```bash
   # In first terminal
   python -m pipenv run python main.py
   ```

### Testing the LSL Stream

You can verify the stream is working with:

1. **bskl viewer** ([bskl](https://github.com/bsl-tools/bsl)):
   ```bash
   bsl_stream_viewer
   ```

   ![BSL Stream Viewer](images/bsl_stream_viewer.png)

2. **Read raw data** with the included example:
   ```bash
   # In second terminal
   python -m pipenv run python examples/read_data.py
   ```

### Config

Change device sampling rate in config.py and emotiv app

### Examples

Get raw data:

```
python main.py & # start lsl server
python examples/read_data.py # get raw data
[4179.35888671875, 4320.5126953125, 4263.84619140625, 4311.53857421875, 4393.58984375, 4347.56396484375, 4371.41015625, 4549.4873046875, 4511.9228515625, 4434.1025390625, 4378.46142578125, 5053.33349609375, 4283.33349609375, 4228.46142578125] 104573.455064594
[4163.33349609375, 4318.0771484375, 4258.7177734375, 4310.384765625, 4396.41015625, 4350.384765625, 4374.615234375, 4550.76904296875, 4508.0771484375, 4429.4873046875, 4374.4873046875, 5058.205078125, 4274.4873046875, 4222.94873046875] 104573.457074205
[4164.615234375, 4316.02587890625, 4255.384765625, 4312.3076171875, 4398.0771484375, 4350.12841796875, 4376.15380859375, 4552.94873046875, 4513.97412109375, 4430.8974609375, 4375.384765625, 5063.7177734375, 4275.384765625, 4225.0] 104573.464060118
[4177.94873046875, 4321.66650390625, 4261.794921875, 4313.46142578125, 4397.1796875, 4347.05126953125, 4373.58984375, 4551.66650390625, 4521.2822265625, 4436.794921875, 4378.7177734375, 5060.76904296875, 4283.7177734375, 4232.8203125] 104573.472085895
```

Write raw data via mne to .fif:

```
python main.py & # start lsl server
python examples/read_and_export_mne.py # write raw data to fif file
Ready.
Writing emotiv-lsl/data_2023-09-20 18:36:10.775860_raw.fif
Closing emotiv-lsl/data_2023-09-20 18:36:10.775860_raw.fif
```
