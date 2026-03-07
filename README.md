# Human Activity Recognition Using Hidden Markov Models (HMM)

## Overview
This project implements a complete pipeline for recognizing human activities (Standing, Walking, Jumping, Still) from smartphone sensor data using Hidden Markov Models.

## Project Structure
```
HMM/
├── Data/                              # Sensor recordings organized by activity
│   ├── Standing/                      # 8 standing recordings
│   ├── Walking/                       # 10 walking recordings
│   ├── Jumping/                       # 10 jumping recordings
│   └── Still/                         # 10 still recordings
├── hmm_activity_recognition.ipynb     # Main Jupyter notebook
├── hmm_activity_recognition.py        # Python script version
├── report.md                          # Project report (4-5 pages)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Setup & Installation
```bash
pip install -r requirements.txt
```

## Data Collection
- **Device**: iPhone 13 (iOS 18.6.2)
- **App**: Sensor Logger v1.54
- **Sampling Rate**: 100 Hz (10ms interval)
- **Sensors**: Accelerometer (x, y, z), Gyroscope (x, y, z)
- **Total Recordings**: 38 samples across 4 activities

## Methodology
1. **Feature Extraction**: 108 features (72 time-domain + 36 frequency-domain) per recording
2. **Model**: Gaussian HMM (one per activity class) trained via Baum-Welch (EM) algorithm
3. **Decoding**: Manual Viterbi algorithm implementation for sequence decoding
4. **Evaluation**: Stratified 75/25 train/test split with sensitivity, specificity, and accuracy metrics

## Results
- **Training Accuracy**: 100%
- **Test Accuracy**: 100%
- **Viterbi Sequence Decoding**: 6/6 correct (100%)

## Running the Notebook
```bash
jupyter notebook hmm_activity_recognition.ipynb
```

## Libraries Used
- `hmmlearn` - Gaussian HMM training (Baum-Welch)
- `numpy`, `scipy` - Feature extraction and manual Viterbi
- `scikit-learn` - Preprocessing and evaluation metrics
- `matplotlib`, `seaborn` - Visualization
