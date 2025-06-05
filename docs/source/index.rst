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

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

A state-of-the-art deep learning framework for predictive maintenance and anomaly detection in turbofan engines using dual LSTM architectures and the CMAPSS dataset.

.. mermaid::

   graph TD
       A[📊 DataLoader] --> B[🔄 DataPreprocessor]
       B --> C[🧠 LSTMAutoEncoder]
       C --> D[🚨 AnomalyDetector]
       D --> E[📈 Visualizer]
       E --> F[🔍 CMAPSSAnomalyAnalyzer]
       F --> G[💾 ModelManager]

       style A fill:#e1f5fe
       style C fill:#f3e5f5
       style D fill:#fff3e0
       style F fill:#e8f5e8

Overview
--------

TurboGuard implements a comprehensive anomaly detection and forecasting system for turbofan engines by leveraging two complementary deep learning approaches:

- **LSTM AutoEncoder**: Reconstruction-based anomaly detection through sequence-to-sequence learning
- **Forecasting LSTM**: Next-step prediction for early fault detection and remaining useful life estimation

The system provides robust, interpretable insights into engine health, enabling proactive maintenance strategies and reducing operational costs.

Key Features
------------

✨ **Dual Model Architecture**: Combines reconstruction and forecasting approaches for comprehensive anomaly detection

🎯 **Interactive Dashboard**: Real-time Streamlit applications for engine health monitoring and visualization

📊 **Multivariate Analysis**: Processes 21 sensor channels with temporal dependencies and inter-correlations

🔧 **Modular Design**: Scalable pipeline supporting both research and production deployment

⚡ **Advanced Preprocessing**: Robust data normalization, sequence generation, and feature selection

🚨 **Multiple Detection Methods**: LSTM-based reconstruction error and statistical anomaly detection

📈 **Comprehensive Evaluation**: Performance metrics including MSE, MAE, precision-recall, and accuracy

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/mouradboutrid/TurboGuard.git
   cd TurboGuard
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Launch interactive dashboard
   streamlit run app/app.py

.. code-block:: python

   # Basic usage example
   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
   
   # Load data
   loader = CMAPSSDataLoader()
   data = loader.load_dataset('FD001')
   
   # Create and train model
   model = LSTMAutoEncoder()
   model.build_model(input_shape=(50, 21))
   model.train(data)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   tutorials/index
   tutorials/installation
   tutorials/quickstart
   tutorials/first_model

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user_guide/index
   user_guide/data_preprocessing
   user_guide/model_training
   user_guide/anomaly_detection
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/index
   examples/basic_usage
   examples/advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index
   api/autoencoder
   api/forecasting

.. toctree::
   :maxdepth: 2
   :caption: Development:

   development/index
   development/contributing
   development/architecture
   development/testing

.. toctree::
   :maxdepth: 1
   :caption: About:

   about/changelog
   about/license

Performance Metrics
------------------

AutoEncoder Model Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Reconstruction Accuracy**: MSE < 0.15 on validation set
- **Anomaly Detection**: F1-Score > 0.52
- **False Positive Rate**: < 20% on normal operations

Forecasting Model Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Prediction Accuracy**: RMSE < 15 cycles for RUL estimation
- **Early Detection**: 60%+ anomalies detected 20+ cycles before failure
- **Multi-step Forecasting**: Maintains accuracy up to 50-step horizon

Dataset Information
------------------

**CMAPSS Dataset** (Commercial Modular Aero-Propulsion System Simulation by NASA)

.. list-table::
   :header-rows: 1

   * - Dataset
     - Fault Modes
     - Operating Conditions
     - Training Engines
     - Test Engines
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

**Sensor Measurements**: 21 channels including fan speed, core speed, turbine temperatures, pressures, fuel flow, and vibration data.

Authors
-------

**Boutrid Mourad** - AI Engineering Student

- 📧 Email: muurad.boutrid@gmail.com
- 🔗 LinkedIn: `Mourad Boutrid <https://www.linkedin.com/in/mourad-boutrid-981659336>`_

**Kassimi Achraf** - AI Engineering Student

- 📧 Email: ac.kassimi@edu.umi.ac.ma
- 🔗 LinkedIn: `Achraf Kassimi <https://www.linkedin.com/in/achraf-kassimi-605418285>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
