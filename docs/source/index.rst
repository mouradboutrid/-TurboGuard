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

**TurboGuard** is a state-of-the-art deep learning framework for predictive maintenance and anomaly detection in turbofan engines, built on dual LSTM architectures and powered by the CMAPSS dataset.




Overview
--------

TurboGuard implements a comprehensive system for turbofan engine health monitoring through two synergistic LSTM-based methods:

- **LSTM AutoEncoder**: Learns to reconstruct input sequences and flags deviations as anomalies.
- **Forecasting LSTM**: Predicts future values to detect abnormal trends and estimate Remaining Useful Life (RUL).

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
Uses LSTM reconstruction errors, forecasting deviations, and statistical thresholds.

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

   # Minimal example
   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder

   loader = CMAPSSDataLoader()
   data = loader.load_dataset('FD001')

   model = LSTMAutoEncoder()
   model.build_model(input_shape=(50, 21))
   model.train(data)

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   tutorials/index
   tutorials/installation
   tutorials/quickstart
   tutorials/first_model

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
🔗 `LinkedIn <https://www.linkedin.com/in/mourad-boutrid-981659336>`_

**Kassimi Achraf**  
*AI Engineering Student*  
📧 ac.kassimi@edu.umi.ac.ma  
🔗 `LinkedIn <https://www.linkedin.com/in/achraf-kassimi-605418285>`_

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
