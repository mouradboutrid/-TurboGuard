User Guide
==========

Welcome to the User Guide for TurboGuard, your comprehensive toolkit for time series anomaly detection and forecasting on turbofan engine data. This guide walks you through the essential components to help you get started, understand the core functionality, and customize the toolkit to your specific needs.

.. toctree::
   :maxdepth: 2
   :caption: User Guide Contents

   installation
   data_preprocessing
   feature_engineering
   model_training
   anomaly_detection
   forecasting
   visualization
   configuration
   api_reference
   troubleshooting

Overview
--------

TurboGuard is designed to provide a complete solution for turbofan engine health monitoring and predictive maintenance:

- **Comprehensive data preprocessing** for CMAPSS and custom turbofan datasets
- **Advanced feature engineering** with automated feature selection and domain-specific transformations
- **Multiple model architectures** including LSTM, CNN-LSTM, Transformer, and Autoencoder networks
- **Dual-mode anomaly detection** using both forecasting residuals and reconstruction errors
- **Real-time forecasting capabilities** for remaining useful life (RUL) prediction
- **Interactive visualization suite** with dashboards, heatmaps, and diagnostic plots
- **Flexible configuration system** supporting multiple deployment scenarios
- **Production-ready API** for integration with existing monitoring systems

Getting Started
---------------

Before diving into the individual components, ensure you have installed TurboGuard and its dependencies. This guide assumes you have:

- Python 3.9+ environment
- Required packages installed via ``pip install -r requirements.txt``
- Access to CMAPSS dataset or similar turbofan engine data
- Basic understanding of time series analysis and machine learning concepts

Core Components
---------------

Installation & Setup
~~~~~~~~~~~~~~~~~~~~~

Complete installation guide including environment setup, dependency management, and initial configuration for different deployment scenarios.

Data Preprocessing
~~~~~~~~~~~~~~~~~~

Comprehensive data handling workflows covering:

- CMAPSS dataset loading and validation
- Custom data format adapters
- Data quality assessment and cleaning
- Missing value imputation strategies
- Outlier detection and treatment

Feature Engineering
~~~~~~~~~~~~~~~~~~~

Advanced feature extraction and selection:

- Sensor data transformations
- Statistical feature derivation
- Domain-specific turbofan features
- Automated feature selection algorithms
- Feature importance analysis

Model Training
~~~~~~~~~~~~~~

Detailed training workflows for multiple architectures:

- **Forecasting Models**: LSTM, CNN-LSTM, Transformer networks for RUL prediction
- **Autoencoder Models**: Various architectures for reconstruction-based anomaly detection
- **Hybrid Models**: Combined forecasting and reconstruction approaches
- Training optimization, hyperparameter tuning, and validation strategies

Anomaly Detection
~~~~~~~~~~~~~~~~~

Multi-layered anomaly detection system:

- **Threshold-based detection** using statistical methods
- **Forecasting residual analysis** for prediction-based anomalies
- **Reconstruction error analysis** for pattern-based anomalies
- **Ensemble methods** combining multiple detection approaches
- **Adaptive thresholding** for evolving operational conditions

Forecasting
~~~~~~~~~~~

Real-time prediction capabilities:

- Remaining Useful Life (RUL) forecasting
- Multi-horizon predictions
- Uncertainty quantification
- Model confidence intervals
- Prediction performance monitoring

Visualization
~~~~~~~~~~~~~

Rich visualization ecosystem:

- **Interactive dashboards** for real-time monitoring
- **Anomaly heatmaps** for temporal pattern analysis
- **Model performance plots** including training curves and validation metrics
- **Engine health trajectories** with confidence bands
- **Feature importance visualizations**
- **Comparative analysis tools** for multiple engines or time periods

Configuration
~~~~~~~~~~~~~

Flexible configuration management:

- **YAML-based configuration** for all components
- **Environment-specific settings** (development, staging, production)
- **Model configuration templates** for common use cases
- **Logging and monitoring configuration**
- **API deployment settings**

API Reference
~~~~~~~~~~~~~

Complete documentation for programmatic access:

- **Core API classes** and methods
- **Data pipeline interfaces**
- **Model training and inference APIs**
- **Visualization function reference**
- **Configuration utilities**
- **Example code snippets** and integration patterns

Troubleshooting
~~~~~~~~~~~~~~~

Common issues and solutions:

- **Installation problems** and dependency conflicts
- **Data loading errors** and format issues
- **Model training failures** and performance optimization
- **Memory and computational considerations**
- **Deployment and scaling guidance**

Quick Start Example
-------------------

.. code-block:: python

   from turboguard import TurboGuard, Config

   # Initialize with configuration
   config = Config.from_file('config/default.yaml')
   tg = TurboGuard(config)

   # Load and preprocess data
   data = tg.load_data('path/to/cmapss/data')
   processed_data = tg.preprocess(data)

   # Train models
   forecasting_model = tg.train_forecasting_model(processed_data)
   anomaly_model = tg.train_anomaly_model(processed_data)

   # Detect anomalies and forecast
   anomalies = tg.detect_anomalies(test_data)
   rul_predictions = tg.forecast_rul(test_data)

   # Visualize results
   tg.create_dashboard(anomalies, rul_predictions)

Advanced Features
-----------------

- **Multi-engine monitoring** with fleet-wide anomaly detection
- **Transfer learning** for adapting models to new engine types
- **Online learning** for continuous model updates
- **A/B testing framework** for model comparison
- **Integration hooks** for external monitoring systems

Performance Considerations
--------------------------

TurboGuard is optimized for both research and production use:

- **Scalable preprocessing** for large datasets
- **GPU acceleration** support for model training
- **Memory-efficient** streaming data processing
- **Parallel processing** capabilities
- **Production deployment** patterns and best practices

Support & Community
-------------------

- **Documentation**: Comprehensive guides and API documentation
- **Examples**: Jupyter notebooks with real-world use cases  
- **GitHub Issues**: Bug reports and feature requests
- **Community Forum**: Discussion and user support
- **Professional Support**: Enterprise consulting and custom development

.. note::
   For detailed implementation examples and tutorials, see the accompanying Jupyter notebooks in the ``examples/`` directory.
