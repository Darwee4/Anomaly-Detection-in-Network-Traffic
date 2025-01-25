# Network Anomaly Detection using Autoencoders

This project implements an anomaly detection system for network traffic using autoencoders. The system is designed to detect unusual patterns in network traffic that could indicate security threats or system malfunctions.

## Features
- Data preprocessing pipeline for network traffic data
- Autoencoder-based anomaly detection model
- Real-time anomaly detection capability
- Support for the UNSW-NB15 dataset

## Requirements
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the anomaly detection system:
```python
python anomaly_detection.py
```

## Implementation Details
The system uses an autoencoder neural network to learn normal network traffic patterns. Key components include:
- Data preprocessing and feature selection
- Autoencoder model architecture with encoder/decoder components
- Reconstruction error-based anomaly detection
- Real-time detection capability

## Dataset
The system is designed to work with the UNSW-NB15 dataset, which contains various types of modern network attacks and normal traffic patterns.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
