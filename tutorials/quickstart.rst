Quick Start Guide
=================

Welcome to TurboGuard! This guide will get you up and running with turbofan engine anomaly detection in just a few minutes.

.. contents::
   :local:
   :depth: 2

What You'll Learn
-----------------

By the end of this quickstart, you'll be able to:

* Install TurboGuard and its dependencies
* Load and preprocess CMAPSS turbofan engine data
* Train your first anomaly detection model
* Visualize anomalies in engine sensor data
* Launch the interactive dashboard

Prerequisites
-------------

Before we begin, make sure you have:

.. tabs::

   .. tab:: System Requirements

      * Python 3.8 or higher
      * 6GB+ RAM (recommended for model training)
      * 2GB free disk space

   .. tab:: Python Knowledge

      * Basic Python programming
      * Familiarity with NumPy/Pandas (helpful but not required)
      * Basic understanding of time series data

Installation
------------

Choose your preferred installation method:

.. tabs::

   .. tab:: From Source (Recommended)

      .. code-block:: bash

         # Clone the repository
         git clone https://github.com/mouradboutrid/TurboGuard.git
         cd TurboGuard

         # Create virtual environment
         python -m venv turbo_env
         source turbo_env/bin/activate  # On Windows: turbo_env\Scripts\activate

         # Install dependencies
         pip install -r requirements.txt

   .. tab:: Direct Download

      .. code-block:: bash

         # Download as ZIP and extract
         wget https://github.com/mouradboutrid/TurboGuard/archive/main.zip
         unzip main.zip
         cd TurboGuard-main

         # Install dependencies
         pip install -r requirements.txt

Verify Installation
-------------------

Let's make sure everything is working correctly:

.. code-block:: python

   # Test basic imports
   import tensorflow as tf
   import numpy as np
   import pandas as pd
   import streamlit as st
   
   print("✅ TensorFlow version:", tf.__version__)
   print("✅ All dependencies loaded successfully!")

.. note::
   If you encounter import errors, double-check that you've activated your virtual environment and installed all requirements.

Your First Anomaly Detection Model
-----------------------------------

Let's create your first anomaly detection model using the LSTM AutoEncoder approach:

Step 1: Load the Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   from src.LSTM_AutoEncoder.data_preprocessor import DataPreprocessor
   
   # Initialize data loader
   loader = CMAPSSDataLoader()
   
   # Load FD001 dataset (single fault mode, single operating condition)
   print("🔄 Loading CMAPSS FD001 dataset...")
   train_data, test_data, rul_data = loader.load_dataset('FD001')
   
   print(f"✅ Training data shape: {train_data.shape}")
   print(f"✅ Test data shape: {test_data.shape}")
   print(f"✅ Loaded {len(train_data['unit'].unique())} training engines")

Step 2: Preprocess the Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize preprocessor
   preprocessor = DataPreprocessor()
   
   # Prepare sequences for LSTM training
   print("🔄 Preprocessing data...")
   X_train, y_train = preprocessor.create_sequences(
       train_data, 
       sequence_length=50,
       target_columns=['s1', 's2', 's3', 's4', 's5']  # Select key sensors
   )
   
   X_test, y_test = preprocessor.create_sequences(
       test_data,
       sequence_length=50,
       target_columns=['s1', 's2', 's3', 's4', 's5']
   )
   
   print(f"✅ Training sequences: {X_train.shape}")
   print(f"✅ Test sequences: {X_test.shape}")

Step 3: Build and Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
   
   # Initialize the LSTM AutoEncoder
   autoencoder = LSTMAutoEncoder(
       sequence_length=50,
       n_features=21,
       latent_dim=64
   )
   
   # Build the model architecture
   print("🔄 Building LSTM AutoEncoder...")
   autoencoder.build_model()
   
   # Train the model
   print("🔄 Training model (this may take a few minutes)...")
   history = autoencoder.train(
       X_train, 
       epochs=50,
       batch_size=32,
       validation_split=0.2,
       verbose=1
   )
   
   print("✅ Model training completed!")

Step 4: Detect Anomalies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.LSTM_AutoEncoder.anomaly_detector import AnomalyDetector
   
   # Initialize anomaly detector
   detector = AnomalyDetector(autoencoder)
   
   # Calculate reconstruction errors
   print("🔄 Calculating reconstruction errors...")
   train_errors = detector.calculate_reconstruction_error(X_train)
   test_errors = detector.calculate_reconstruction_error(X_test)
   
   # Set anomaly threshold (95th percentile of training errors)
   threshold = np.percentile(train_errors, 95)
   print(f"📊 Anomaly threshold: {threshold:.4f}")
   
   # Detect anomalies in test data
   anomalies = detector.detect_anomalies(test_errors, threshold)
   anomaly_rate = np.mean(anomalies) * 100
   
   print(f"🚨 Detected {anomaly_rate:.1f}% anomalous sequences in test data")

Step 5: Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.LSTM_AutoEncoder.visualizer import Visualizer
   import matplotlib.pyplot as plt
   
   # Initialize visualizer
   viz = Visualizer()
   
   # Plot training history
   viz.plot_training_history(history)
   plt.show()
   
   # Plot reconstruction error distribution
   viz.plot_error_distribution(train_errors, test_errors, threshold)
   plt.show()
   
   # Plot anomaly detection results
   viz.plot_anomaly_timeline(test_errors, anomalies, threshold)
   plt.show()

Complete Example Script
-----------------------

Here's the complete script that combines all the steps above:

.. code-block:: python

   """
   TurboGuard Quickstart Example
   Complete anomaly detection pipeline for turbofan engines
   """
   
   import numpy as np
   import matplotlib.pyplot as plt
   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   from src.LSTM_AutoEncoder.data_preprocessor import DataPreprocessor
   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
   from src.LSTM_AutoEncoder.anomaly_detector import AnomalyDetector
   from src.LSTM_AutoEncoder.visualizer import Visualizer
   
   def main():
       print("🚀 TurboGuard Quickstart Example")
       print("=" * 40)
       
       # Step 1: Load data
       loader = CMAPSSDataLoader()
       train_data, test_data, rul_data = loader.load_dataset('FD001')
       print(f"✅ Loaded {len(train_data)} training samples")
       
       # Step 2: Preprocess
       preprocessor = DataPreprocessor()
       X_train, _ = preprocessor.create_sequences(train_data, sequence_length=50)
       X_test, _ = preprocessor.create_sequences(test_data, sequence_length=50)
       print(f"✅ Created {len(X_train)} training sequences")
       
       # Step 3: Train model
       autoencoder = LSTMAutoEncoder(sequence_length=50, n_features=21)
       autoencoder.build_model()
       history = autoencoder.train(X_train, epochs=20, verbose=0)
       print("✅ Model training completed")
       
       # Step 4: Detect anomalies
       detector = AnomalyDetector(autoencoder)
       train_errors = detector.calculate_reconstruction_error(X_train)
       test_errors = detector.calculate_reconstruction_error(X_test)
       threshold = np.percentile(train_errors, 95)
       anomalies = detector.detect_anomalies(test_errors, threshold)
       
       print(f"📊 Anomaly threshold: {threshold:.4f}")
       print(f"🚨 Detected {np.mean(anomalies)*100:.1f}% anomalies")
       
       # Step 5: Visualize
       viz = Visualizer()
       viz.plot_training_history(history)
       viz.plot_error_distribution(train_errors, test_errors, threshold)
       plt.show()
       
       print("\n🎉 Quickstart completed successfully!")
       print("Next steps:")
       print("- Try the interactive dashboard: streamlit run app/app.py")
       print("- Explore advanced examples in the documentation")
       
   if __name__ == "__main__":
       main()

Running the Interactive Dashboard
---------------------------------

TurboGuard comes with a powerful Streamlit dashboard for interactive analysis:

.. code-block:: bash

   # Launch the main dashboard
   streamlit run app/app.py

This will open a web interface where you can:

🔧 **Configure Models**: Adjust hyperparameters and model architecture

📊 **Upload Data**: Use your own datasets or explore the CMAPSS data

📈 **Real-time Visualization**: See anomaly detection results in real-time

💾 **Export Results**: Download trained models and analysis reports

Dashboard Features
~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Data Explorer

      * Load and visualize CMAPSS datasets
      * Explore sensor readings and engine health metrics
      * Identify patterns and correlations

   .. tab:: Model Training

      * Configure AutoEncoder and Forecasting models
      * Monitor training progress in real-time
      * Compare different model architectures

   .. tab:: Anomaly Detection

      * Set custom thresholds and detection parameters
      * Visualize anomalies across different engines
      * Export detection results

   .. tab:: Performance Analysis

      * Evaluate model performance metrics
      * Compare different detection approaches
      * Generate comprehensive reports

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: ImportError: No module named 'src'
   :color: warning

   This usually means you're not running from the TurboGuard root directory:

   .. code-block:: bash

      cd /path/to/TurboGuard
      python your_script.py

.. dropdown:: CUDA out of memory error
   :color: warning

   Reduce batch size or sequence length:

   .. code-block:: python

      # Reduce batch size
      autoencoder.train(X_train, batch_size=16)  # Instead of 32
      
      # Or reduce sequence length
      X_train, _ = preprocessor.create_sequences(data, sequence_length=30)

.. dropdown:: Dashboard won't start
   :color: warning

   Check that Streamlit is installed and you're in the correct directory:

   .. code-block:: bash

      pip install streamlit
      cd TurboGuard
      streamlit run app/app.py

Getting Help
~~~~~~~~~~~~

If you encounter issues not covered here:

* 📖 Check the :doc:`../user_guide/index` for detailed documentation
* 💻 Browse :doc:`../examples/index` for more code examples
* 🐛 Report bugs on `GitHub Issues <https://github.com/mouradboutrid/TurboGuard/issues>`_
* 💬 Join discussions on our `GitHub Discussions <https://github.com/mouradboutrid/TurboGuard/discussions>`_

Next Steps
----------

Congratulations! You've successfully:

✅ Installed TurboGuard and its dependencies

✅ Loaded and preprocessed CMAPSS turbofan data

✅ Trained your first LSTM AutoEncoder model

✅ Detected anomalies in engine sensor data

✅ Visualized the results

Ready to dive deeper? Here's what to explore next:

.. container:: next-steps-grid

   .. container:: next-step-item

      📚 **Learn the Fundamentals**
      
      :doc:`../user_guide/data_preprocessing`
      
      Deep dive into data preprocessing techniques and feature engineering.

   .. container:: next-step-item

      🧠 **Advanced Modeling**
      
      :doc:`../user_guide/model_training`
      
      Explore advanced model architectures and training strategies.

   .. container:: next-step-item

      🔍 **Detection Methods**
      
      :doc:`../user_guide/anomaly_detection`
      
      Learn about different anomaly detection approaches and when to use them.

   .. container:: next-step-item

      📊 **Visualization**
      
      :doc:`../user_guide/visualization`
      
      Master the art of visualizing anomalies and model performance.

   .. container:: next-step-item

      💻 **Real Examples**
      
      :doc:`../examples/advanced_usage`
      
      See TurboGuard in action with real-world scenarios.

   .. container:: next-step-item

      🚀 **Production**
      
      :doc:`../examples/production_deployment`
      
      Learn how to deploy TurboGuard in production environments.