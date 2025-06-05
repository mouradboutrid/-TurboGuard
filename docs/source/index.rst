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

**TurboGuard** is a state-of-the-art deep learning framework for predictive maintenance and anomaly detection in turbofan engines using dual LSTM architectures and the CMAPSS dataset.

.. note::
   This documentation covers TurboGuard v2.0.0. For older versions, please check the :doc:`about/changelog`.

Key Features
------------

✨ **Dual Model Architecture**: Combines reconstruction and forecasting approaches for comprehensive anomaly detection

🖥️ **Interactive Dashboard**: Real-time Streamlit applications for engine health monitoring and visualization

📊 **Multivariate Analysis**: Processes 21 sensor channels with temporal dependencies and inter-correlations

🔧 **Modular Design**: Scalable pipeline supporting both research and production deployment

🔄 **Advanced Preprocessing**: Robust data normalization, sequence generation, and feature selection

🚨 **Multiple Detection Methods**: LSTM-based reconstruction error and statistical anomaly detection

📈 **Comprehensive Evaluation**: Performance metrics including MSE, MAE, precision-recall, and accuracy

System Architecture
-------------------

TurboGuard implements two complementary deep learning pipelines:

LSTM AutoEncoder Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

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

The AutoEncoder pipeline focuses on **reconstruction-based anomaly detection** through sequence-to-sequence learning.

Forecasting LSTM Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TD
       A[📊 DataProcessor] --> B[🎯 PrognosticFeatureSelector]
       B --> C[🔮 PrognosticLSTMModel]
       C --> D[📊 PrognosticVisualizationSuite]
       D --> E[🏥 CMAPSSPrognosticHealthMonitor]
       E --> F[⚡ AnomalyDetectionEngine]
       
       style A fill:#e1f5fe
       style B fill:#f1f8e9
       style C fill:#f3e5f5
       style E fill:#fff8e1
       style F fill:#ffebee

The Forecasting pipeline enables **next-step prediction** for early fault detection and remaining useful life estimation.

Quick Start
-----------

Get started with TurboGuard in just a few minutes:

.. code-block:: bash

   git clone https://github.com/mouradboutrid/TurboGuard.git
   cd TurboGuard
   pip install -r requirements.txt
   streamlit run app/app.py

.. code-block:: python

   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
   
   # Load and analyze turbofan engine data
   loader = CMAPSSDataLoader()
   data = loader.load_dataset('FD001')
   
   # Detect anomalies using LSTM AutoEncoder
   autoencoder = LSTMAutoEncoder()
   autoencoder.build_model(input_shape=(50, 21))
   anomalies = autoencoder.detect_anomalies(data)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   tutorials/index
   tutorials/installation
   tutorials/quickstart
   tutorials/first_model

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/index
   user_guide/data_preprocessing
   user_guide/model_training
   user_guide/anomaly_detection
   user_guide/visualization
   user_guide/dashboard

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index
   examples/basic_usage
   examples/advanced_usage
   examples/custom_datasets
   examples/production_deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/autoencoder
   api/forecasting
   api/utilities

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/index
   development/contributing
   development/architecture
   development/testing
   development/performance

.. toctree::
   :maxdepth: 1
   :caption: About
   :hidden:

   about/changelog
   about/license
   about/authors

Dataset Information
-------------------

TurboGuard works with the **CMAPSS Dataset** (Commercial Modular Aero-Propulsion System Simulation) provided by NASA:

.. list-table:: CMAPSS Dataset Overview
   :widths: 15 15 20 20 20
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

The dataset includes **21 sensor measurements** covering fan speed, core speed, turbine temperatures, pressures, fuel flow, and vibration data.

Performance Highlights
----------------------

.. container:: performance-grid

   .. container:: performance-item

      **AutoEncoder Model**
      
      * Reconstruction Accuracy: MSE < 0.15
      * Anomaly Detection F1-Score: > 0.52
      * False Positive Rate: < 20%

   .. container:: performance-item

      **Forecasting Model**
      
      * RUL Prediction RMSE: < 15 cycles
      * Early Detection: 60%+ anomalies detected 20+ cycles before failure
      * Multi-step Forecasting: Up to 50-step horizon

Need Help?
----------

.. container:: help-grid

   .. container:: help-item

      📚 **New to TurboGuard?**
      
      Start with our :doc:`tutorials/quickstart` guide and follow the step-by-step :doc:`tutorials/first_model` tutorial.

   .. container:: help-item

      🔧 **Ready to Build?**
      
      Check out the :doc:`user_guide/index` for comprehensive documentation on all features.

   .. container:: help-item

      💻 **Need Code Examples?**
      
      Browse our :doc:`examples/index` section for real-world use cases and implementation patterns.

   .. container:: help-item

      🚀 **Contributing?**
      
      Read our :doc:`development/contributing` guide to get started with development.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
