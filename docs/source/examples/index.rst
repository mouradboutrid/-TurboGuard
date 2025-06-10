Examples
========

This section provides practical examples demonstrating how to use TurboGuard for common tasks such as data loading, training models, performing anomaly detection, and visualizing results. The examples are designed to help you quickly get hands-on experience and understand the workflow of the toolkit.

.. toctree::
   :maxdepth: 2
   :caption: Example Tutorials

   basic_usage
   advanced_usage

Overview
--------

The examples range from simple scripts showing how to run the core functions with default settings to more advanced use cases where you can customize parameters, extend models, or integrate with other systems.

**Basic Usage - LSTM Autoencoder Analysis**

    Demonstrates the core LSTM Autoencoder functionality for anomaly detection on CMAPSS datasets. Shows how to train models, save/load them, and perform basic anomaly analysis with minimal configuration.

    **What you'll learn:**

    - Training LSTM Autoencoder models on individual datasets (e.g., FD004)
    - Loading and reusing pre-trained models for new predictions
    - Comparing performance across all CMAPSS datasets
    - Basic model configuration and hyperparameter settings
    - Understanding model summaries and performance metrics

    **Key functions demonstrated:**

    - ``analyze_dataset()`` - Train and analyze single dataset
    - ``load_saved_model()`` - Load pre-trained models
    - ``compare_all_datasets()`` - Comprehensive dataset comparison
    - ``predict_anomalies()`` - Make predictions on new data


**Advanced Usage - Forecasting-based Anomaly Detection**

    Covers advanced anomaly detection using forecasting models with comprehensive visualization and analysis capabilities. Includes detailed sensor-level analysis, timeline visualization, and multi-method ensemble detection.

    **What you'll learn:**

    - Loading and configuring pre-trained forecasting models
    - Advanced data preprocessing with operational mode clustering
    - Multi-method anomaly detection (MSE, MAE, Max Error, Ensemble)
    - Comprehensive visualization techniques:
        - Error distribution analysis
        - Unit-level anomaly rate analysis  
        - Sensor-specific anomaly patterns
        - Timeline-based anomaly progression
        - Heatmap visualizations
    - Statistical analysis and anomaly summarization

    **Key features demonstrated:**

    - ``AnomalyPredictorTest`` class for comprehensive analysis
    - Multiple error calculation methods and thresholding
    - Advanced plotting functions for sensor data analysis
    - Unit-level anomaly tracking and progression analysis
    - Ensemble-based detection for improved accuracy


Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Before running the examples, ensure you have:

- TurboGuard and all dependencies installed
- Python 3.7 or higher  
- Required libraries: ``tensorflow``, ``sklearn``, ``matplotlib``, ``pandas``, ``numpy``
- CMAPSS dataset files in the appropriate directory
- A virtual environment (recommended)

Installation Check
~~~~~~~~~~~~~~~~~~

Verify your installation by running:

.. code-block:: bash

    python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

Example Files Structure
~~~~~~~~~~~~~~~~~~~~~~~

The example scripts are organized as follows:

.. code-block:: text

    src/
    ├── LSTM_Autoencoder/
    ├── Forecasting_LSTM/         
    │   autoencoder_anomaly_predictor_test    # LSTM Autoencoder analysis
    ├── forecaster_anomaly_predictor_test.py  # Advanced forecasting analysis
    └── data/
        ├── train_FD001.txt               # Training data samples
        ├── test_FD001.txt                # Test data samples
        └── ...                           # Other CMAPSS datasets

Quick Start - Basic Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

To run the basic LSTM Autoencoder examples:

.. code-block:: bash

    cd src/LSTM_Autoencoder
    python lstm_autoencoder_demo.py

**Available demo functions:**

- ``demo_single_dataset_analysis()`` - Analyze FD004 dataset with default parameters
- ``demo_load_and_predict()`` - Load saved model and make predictions  
- ``demo_full_comparison()`` - Compare all CMAPSS datasets (longer runtime)

**Example output:**  
The script will display model training progress, performance metrics, and save trained models to the ``saved_models/`` directory.

Quick Start - Advanced Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the advanced forecasting-based anomaly detection:

.. code-block:: bash

    cd src
    python forecaster_anomaly_predictor_test.py

**Prerequisites for advanced usage:**

- Pre-trained forecasting model (``lstm_model_*.h5``)
- Configuration file (``analysis_config_*.json``) 
- Test dataset file (``test_FD004.txt``)

**Key parameters to customize:**

.. code-block:: python

    # In main() function
    model_path = 'path/to/your/saved_model.h5'
    config_path = 'path/to/your/config.json'  
    test_data_path = 'path/to/test_data.txt'
    threshold_percentile = 95  # Anomaly detection threshold

**Expected outputs:**

- Comprehensive anomaly detection summary
- Multiple visualization plots:
    - Anomaly detection method comparison
    - Error distribution histograms
    - Top anomalous units analysis
    - Sensor-level anomaly patterns
    - Timeline progression plots

Understanding the Results
-------------------------

**Basic Usage Results:**

- Model performance metrics (MSE, MAE)
- Dataset comparison statistics
- Saved model files for future use
- Performance summaries across different datasets

**Advanced Usage Results:**

- Detailed anomaly statistics and rates
- Unit-level anomaly progression analysis
- Sensor-specific anomaly patterns  
- Visual insights into failure progression
- Ensemble detection results for improved accuracy

Customization Options
---------------------

**Basic Usage Customization:**

.. code-block:: python

    # Customize training parameters
    results = analyzer.analyze_dataset(
        dataset_id='FD001',           # Choose dataset
        sequence_length=50,           # Adjust sequence length
        sensors_to_drop=[1, 5, 10],  # Remove specific sensors
        epochs=50,                   # Training epochs
        save_model=True              # Save trained model
    )

**Advanced Usage Customization:**

.. code-block:: python

    # Customize anomaly detection
    results = predictor.predict_and_analyze(
        test_filepath='your_test_data.txt',
        threshold_percentile=90        # Adjust sensitivity
    )
    
    # Customize visualizations
    predictor.plot_sensor_anomalies(
        test_df, anomaly_results, unit_ids, 
        sequence_indices, top_n_units=3   # Show fewer units
    )

Troubleshooting
---------------

**Common Issues:**

1. **Model Loading Errors:** The advanced example includes multiple fallback methods for loading TensorFlow models with different configurations.

2. **Data Path Issues:** Ensure all file paths in the scripts match your actual data locations.

3. **Memory Issues:** For large datasets, consider reducing batch sizes or sequence lengths.

4. **Visualization Issues:** If plots don't display, ensure you have matplotlib with appropriate backend configured.

**Performance Tips:**

- Use GPU acceleration for faster training (basic usage)
- Adjust ``threshold_percentile`` to control anomaly sensitivity
- Use smaller ``top_n_units`` values for faster visualization generation
- Consider data sampling for very large datasets

Need Help?
----------

- Check the individual README files in each example directory for detailed instructions
- Refer to the documentation for function details
- Visit our FAQ section for common questions
- Report issues on our GitHub repository: https://github.com/mouradboutrid/turboguard

---
