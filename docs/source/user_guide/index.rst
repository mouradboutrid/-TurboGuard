User Guide
==========
Welcome to the User Guide for TurboGuard, your toolkit for time series anomaly detection and forecasting on turbofan engine data.
This guide walks you through the essential components to help you get started, understand the core functionality, and customize the toolkit to your needs.
.. toctree::
   :maxdepth: 2
   :caption: User Guide Contents
   data_preprocessing
   model_training
   anomaly_detection
   visualization
Overview
--------
TurboGuard is designed to provide:
- **Robust data preprocessing** for CMAPSS and similar datasets.
- **Modular model training workflows**, including forecasting and autoencoder-based anomaly detection models.
- **Advanced anomaly detection algorithms** tuned for turbofan engine health monitoring.
- **Rich visualization tools** to analyze results and monitor engine health intuitively.
Getting Started
---------------
Before diving into the individual components, ensure you have installed TurboGuard and its dependencies. Refer to the installation section in the Tutorials if you haven't yet.
This guide assumes you have a working Python environment (3.9+) with necessary packages installed as specified in `requirements.txt`.
Core Components
---------------
- **Data Preprocessing**  
  Explains how to load, clean, and transform raw CMAPSS datasets into sequences suitable for model input. Covers normalization, feature selection, and sequence windowing.
- **Model Training**  
  Details the process to configure, train, and validate different neural network architectures (LSTM, CNN-LSTM, Attention models) on prepared data.
- **Anomaly Detection**  
  Describes the techniques implemented for detecting anomalies using forecasting residuals and autoencoder reconstruction errors, including thresholding and scoring strategies.
- **Visualization**  
  Provides instructions to generate diagnostic plots, time series anomaly heatmaps, and dashboard views to interpret model outputs and detected anomalies.
Support
-------
For troubleshooting, bug reports, or feature requests, please check the Development section or open an issue on the GitHub repository.
---
