TurboGuard Documentation
========================

.. image:: https://img.shields.io/badge/Python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/TensorFlow-2.x-orange.svg
   :target: https://tensorflow.org/
   :alt: TensorFlow

.. image:: https://img.shields.io/badge/Streamlit-1.x-red.svg
   :target: https://streamlit.io/
   :alt: Streamlit

**TurboGuard** is a deep learning framework for predictive maintenance and anomaly detection in turbofan engines, built on dual LSTM architectures using CMAPSS dataset.

Overview
--------

TurboGuard implements a comprehensive system for turbofan engine health monitoring through two synergistic LSTM-based methods:

- **LSTM AutoEncoder**: Learns to reconstruct input sequences and flags deviations as anomalies.
- **Forecasting LSTM**: Predicts future values to detect abnormal trends.

The framework enables proactive maintenance, minimizes downtime, and optimizes operational efficiency.

Key Features
------------

✨ **Dual Model Architecture**  
Combines reconstruction-based and forecasting-based methods for more reliable anomaly detection.

🎯 **Interactive Dashboard**  
Real-time visualization and health analytics using Streamlit.

📊 **Multivariate Sensor Analysis**  
Processes all 21 sensor channels with full temporal and contextual awareness.

🔧 **Modular and Scalable**  
Designed for both research and production environments with pluggable components.

⚡ **Advanced Preprocessing**  
Supports robust normalization, dynamic sequence generation, and feature selection.

🚨 **Multiple Detection Strategies**  
Uses LSTM reconstruction errors, forecasting deviations, and statistical also dynamical thresholds.

📈 **Detailed Evaluation Metrics**  
Includes MSE, MAE, RMSE, F1-score, precision-recall, and anomaly lead-time.

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/mouradboutrid/TurboGuard.git
   cd TurboGuard

   # Install dependencies
   pip install -r requirements.txt

   # Launch the dashboard
   streamlit run app/app.py

.. code-block:: python

   import numpy as np
   from data_loader import DataLoader
   from data_preprocessor import DataPreprocessor
   from lstm_autoencoder import LSTMAutoencoder
   from anomaly_detector import AnomalyDetector

   # Load dataset (returns a dict with keys 'train', 'test', 'rul')
   loader = DataLoader(data_dir='/content/drive/MyDrive/CMAPSSData')
   dataset = loader.load_dataset('FD001')

   train_raw = dataset['train']  # pandas DataFrame
   test_raw = dataset['test']    # pandas DataFrame
   rul_raw = dataset['rul']      # pandas DataFrame

   # Preprocess the train and test data
   preprocessor = DataPreprocessor()
   train_processed = preprocessor.preprocess_data(train_raw, calculate_rul=True, normalize=True)
   test_processed = preprocessor.preprocess_data(test_raw, calculate_rul=False, normalize=True)

   # Create sequences from preprocessed data
   X_train, y_train = preprocessor.create_sequences(train_processed, sequence_length=50, target_col='RUL')
   X_test = preprocessor.create_sequences(test_processed, sequence_length=50)

   print("X_train shape:", X_train.shape)
   print("X_test shape:", X_test.shape)

   # Build and train the LSTM Autoencoder
   autoencoder = LSTMAutoencoder()
   autoencoder.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
   autoencoder.train(X_train, epochs=50, batch_size=32)

   # Detect anomalies on test set
   detector = AnomalyDetector()
   anomaly_scores, anomaly_flags, threshold = detector.detect_lstm_anomalies(X_test, autoencoder)

   print(f"Anomaly threshold: {threshold:.4f}")
   print(f"Detected {np.sum(anomaly_flags)} anomalies out of {len(anomaly_flags)} test samples")
   print(f"Anomaly rate: {np.sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%")

Getting Started Tutorials
-------------------------

Follow our comprehensive tutorial series to master TurboGuard:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   tutorials/index
   tutorials/installation
   tutorials/quickstart
   tutorials/first_model

📚 **Tutorial Overview:**

**Installation Tutorial** - Complete setup guide with system requirements, dependency installation, GPU configuration, and troubleshooting for common installation issues.

**Quick Start Tutorial** - Get TurboGuard running in 3 steps! Launch the interactive dashboard, explore the CMAPSS dataset, and run your first anomaly detection in minutes.

**First Model Tutorial** - Build your complete first model from scratch:

- 🔧 **Data Preparation**: Load and preprocess CMAPSS FD001 dataset
- 🤖 **LSTM AutoEncoder**: Build 64-dimensional encoder-decoder architecture  
- 📊 **Training Pipeline**: Train with 50 epochs, monitor validation metrics
- 🚨 **Anomaly Detection**: Implement threshold-based anomaly detection
- 📈 **Forecasting LSTM**: Build multi-step prediction model for RUL estimation
- 💾 **Model Management**: Save, load, and version your trained models
- 🎯 **Production Pipeline**: Create complete prediction function with preprocessing
- 📊 **Visualization Dashboard**: Generate comprehensive 6-panel performance dashboard

**What You'll Achieve:**
- ✅ Functional LSTM AutoEncoder with <0.15 MSE reconstruction error
- ✅ Forecasting model with <15 cycles RUL prediction RMSE  
- ✅ Complete anomaly detection pipeline with >50% F1-score
- ✅ Production-ready model saving and loading system
- ✅ Interactive visualization dashboard for real-time monitoring

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/data_preprocessing
   user_guide/model_training
   user_guide/anomaly_detection
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/basic_usage
   examples/advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/autoencoder
   api/forecasting

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index
   development/contributing
   development/architecture
   development/testing

.. toctree::
   :maxdepth: 1
   :caption: About

   about/changelog
   about/license

Performance Metrics
-------------------

AutoEncoder Model
~~~~~~~~~~~~~~~~~

- **Reconstruction Error**: MSE < 0.15 on validation data
- **Detection F1-Score**: > 0.52
- **False Positives**: < 20% in nominal operating ranges

Forecasting Model
~~~~~~~~~~~~~~~~~

- **RUL Prediction Accuracy**: RMSE < 15 cycles
- **Early Warning**: > 60% anomalies flagged at least 20 cycles pre-failure
- **Long-Horizon Forecasting**: Maintains performance for up to 50 steps

Dataset Summary
---------------

**NASA CMAPSS Dataset** (Commercial Modular Aero-Propulsion System Simulation)

.. list-table::
   :header-rows: 1

   * - Subset
     - Fault Modes
     - Operating Conditions
     - Training Units
     - Test Units
   * - FD001
     - 1
     - 1
     - 100
     - 100
   * - FD002
     - 1
     - 6
     - 260
     - 259
   * - FD003
     - 2
     - 1
     - 100
     - 100
   * - FD004
     - 2
     - 6
     - 248
     - 249

**Sensors**: 21 channels including fan speed, core speed, various temperatures and pressures, fuel flow, and vibration.

Authors
-------

**Boutrid Mourad**  
*AI Engineering Student*  
📧 muurad.boutrid@gmail.com  
🔗 LinkedIn <https://www.linkedin.com/in/mourad-boutrid-981659336>_

**Kassimi Achraf**  
*AI Engineering Student*  
📧 ac.kassimi@edu.umi.ac.ma  
🔗 LinkedIn <https://www.linkedin.com/in/achraf-kassimi-605418285>_
