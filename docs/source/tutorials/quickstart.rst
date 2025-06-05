Quick Start Guide
=================

This guide will get you up and running with TurboGuard in minutes.

Installation
------------

.. tabs::

   .. tab:: PyPI (Recommended)

      .. code-block:: bash

         pip install turboguard

   .. tab:: From Source

      .. code-block:: bash

         git clone https://github.com/mouradboutrid/TurboGuard.git
         cd TurboGuard
         pip install -e .

   .. tab:: Development Setup

      .. code-block:: bash

         git clone https://github.com/mouradboutrid/TurboGuard.git
         cd TurboGuard
         python -m venv turbo_env
         source turbo_env/bin/activate  # On Windows: turbo_env\Scripts\activate
         pip install -r requirements.txt

Verify Installation
-------------------

Test your installation by running:

.. code-block:: python

   import sys
   sys.path.append('src')
   
   from LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   print("TurboGuard installed successfully!")

Basic Usage
-----------

Here's a simple example to get you started:

.. code-block:: python

   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   from src.LSTM_AutoEncoder.data_preprocessor import DataPreprocessor
   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
   from src.LSTM_AutoEncoder.anomaly_detector import AnomalyDetector
   
   # Step 1: Load data
   loader = CMAPSSDataLoader()
   train_data, test_data = loader.load_dataset('FD001')
   
   # Step 2: Preprocess data
   preprocessor = DataPreprocessor()
   X_train, y_train = preprocessor.create_sequences(train_data)
   X_test, y_test = preprocessor.create_sequences(test_data)
   
   # Step 3: Create and train model
   autoencoder = LSTMAutoEncoder(
       sequence_length=50,
       n_features=21,
       latent_dim=64
   )
   autoencoder.build_model(input_shape=(50, 21))
   autoencoder.train(X_train, epochs=50)
   
   # Step 4: Detect anomalies
   detector = AnomalyDetector(autoencoder)
   anomalies = detector.detect_anomalies(X_test)
   
   print(f"Detected {len(anomalies)} anomalies")

Interactive Dashboard
--------------------

Launch the interactive Streamlit dashboard for a visual experience:

.. code-block:: bash

   streamlit run app/app.py

This will open a web interface where you can:

- 📊 **Upload datasets**: Load your own CMAPSS data or use built-in datasets
- ⚙️ **Configure models**: Adjust hyperparameters through an intuitive interface
- 🎯 **Train models**: Monitor training progress in real-time
- 📈 **Visualize results**: Interactive plots and anomaly detection results
- 💾 **Export models**: Save trained models for later use

Dashboard Features
~~~~~~~~~~~~~~~~~~

The dashboard includes several specialized apps:

1. **Data Loader App** (``app/loader_app.py``):
   - Load and explore CMAPSS datasets
   - Visualize sensor data distributions
   - Check data quality and completeness

2. **Preprocessor App** (``app/preprocessor_app.py``):
   - Configure preprocessing parameters
   - Preview processed sequences
   - Analyze feature correlations

3. **AutoEncoder App** (``app/autoencoder_anomaly_detector_app.py``):
   - Train LSTM AutoEncoder models
   - Visualize reconstruction errors
   - Detect anomalies with interactive thresholds

4. **Forecaster App** (``app/forecaster_anomaly_predictor_app.py``):
   - Train prognostic LSTM models
   - Predict remaining useful life (RUL)
   - Early anomaly detection

Example Workflow
---------------

Here's a complete workflow example:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from src.LSTM_AutoEncoder import *
   
   # Configuration
   DATASET = 'FD001'
   SEQUENCE_LENGTH = 50
   LATENT_DIM = 64
   EPOCHS = 100
   
   # 1. Data Loading and Exploration
   loader = CMAPSSDataLoader()
   train_df, test_df, rul_df = loader.load_all_data(DATASET)
   
   print(f"Training data shape: {train_df.shape}")
   print(f"Test data shape: {test_df.shape}")
   print(f"RUL data shape: {rul_df.shape}")
   
   # 2. Data Preprocessing
   preprocessor = DataPreprocessor(
       sequence_length=SEQUENCE_LENGTH,
       normalize=True,
       feature_columns=loader.get_sensor_columns()
   )
   
   X_train, y_train = preprocessor.fit_transform(train_df)
   X_test, y_test = preprocessor.transform(test_df)
   
   print(f"Training sequences shape: {X_train.shape}")
   print(f"Test sequences shape: {X_test.shape}")
   
   # 3. Model Building and Training
   autoencoder = LSTMAutoEncoder(
       sequence_length=SEQUENCE_LENGTH,
       n_features=X_train.shape[2],
       latent_dim=LATENT_DIM,
       learning_rate=0.001
   )
   
   autoencoder.build_model(input_shape=X_train.shape[1:])
   
   # Train with validation split
   history = autoencoder.train(
       X_train, 
       epochs=EPOCHS,
       validation_split=0.2,
       batch_size=32,
       verbose=1
   )
   
   # 4. Anomaly Detection
   detector = AnomalyDetector(autoencoder)
   
   # Calculate reconstruction errors
   train_errors = detector.calculate_reconstruction_error(X_train)
   test_errors = detector.calculate_reconstruction_error(X_test)
   
   # Set threshold based on training data
   threshold = detector.set_threshold(train_errors, method='percentile', percentile=95)
   
   # Detect anomalies
   anomalies = detector.detect_anomalies(X_test, threshold=threshold)
   
   print(f"Threshold: {threshold:.4f}")
   print(f"Anomalies detected: {np.sum(anomalies)}/{len(anomalies)}")
   
   # 5. Visualization
   visualizer = Visualizer()
   
   # Plot training history
   visualizer.plot_training_history(history)
   
   # Plot reconstruction errors
   visualizer.plot_reconstruction_errors(train_errors, test_errors, threshold)
   
   # Plot anomalies
   visualizer.plot_anomalies(test_errors, anomalies, threshold)
   
   plt.show()

Next Steps
----------

Now that you have TurboGuard running, explore these advanced features:

- :doc:`../user_guide/data_preprocessing` - Learn about advanced data preparation techniques
- :doc:`../user_guide/model_training` - Understand model architecture and hyperparameter tuning
- :doc:`../user_guide/anomaly_detection` - Explore different anomaly detection methods
- :doc:`../examples/advanced_usage` - See comprehensive examples and use cases
- :doc:`../api/index` - Dive deep into the API documentation

Common Issues
-------------

.. note::
   **Memory Issues**: If you encounter memory errors during training, try:
   
   - Reducing batch size: ``batch_size=16`` or ``batch_size=8``
   - Reducing sequence length: ``sequence_length=30``
   - Using a smaller model: ``latent_dim=32``

.. warning::
   **CUDA/GPU Issues**: If you have GPU-related errors:
   
   - Ensure TensorFlow-GPU is properly installed
   - Set ``os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`` to force CPU usage
   - Check GPU memory availability

.. tip::
   **Performance Optimization**: For better performance:
   
   - Use GPU acceleration when available
   - Implement early stopping during training
   - Use model checkpointing to save progress
   - Consider data augmentation for small datasets

Need Help?
----------

If you encounter any issues:

1. Check the :doc:`../development/troubleshooting` section
2. Review our `GitHub Issues <https://github.com/mouradboutrid/TurboGuard/issues>`_
3. Join our community discussions
4. Contact the authors directly