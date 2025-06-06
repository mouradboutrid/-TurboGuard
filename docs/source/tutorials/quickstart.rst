Quick Start
===========

Get TurboGuard up and running in just a few minutes! This guide will have you exploring turbofan engine data and detecting anomalies right away.

🚀 Launch in 3 Steps
--------------------

**Step 1: Start the Dashboard**

.. code-block:: bash

   cd TurboGuard
   streamlit run app/app.py

**Step 2: Open Your Browser**

Navigate to: http://localhost:8501

**Step 3: Explore!**

You should see the TurboGuard dashboard loading with sample data.

Dashboard Overview
------------------

The TurboGuard dashboard provides an intuitive interface for turbofan engine health monitoring:

📊 **Main Sections**

- **Data Overview**: Explore the CMAPSS dataset
- **Model Training**: Train LSTM AutoEncoder and Forecasting models  
- **Anomaly Detection**: Real-time anomaly detection and visualization
- **Health Monitoring**: Engine health metrics and RUL predictions
- **Settings**: Configure model parameters and thresholds

🎯 **Key Features**

- Interactive sensor data visualization  
- Real-time anomaly alerts
- Remaining Useful Life (RUL) predictions
- Model performance metrics
- Customizable detection thresholds

First Exploration
-----------------

Let's explore the dashboard step by step:

**1. Data Overview Tab**

.. code-block:: python

   # The dashboard automatically loads sample data
   # You'll see:
   # - 21 sensor channels from turbofan engines
   # - Multiple engine units with different operating conditions
   # - Time series plots of sensor readings

**Key Observations:**
- Sensor readings show different patterns over engine lifecycle
- Some sensors exhibit clear degradation trends
- Different fault modes create distinct signatures

**2. Model Training Tab**

The dashboard provides pre-configured model settings:

- **LSTM AutoEncoder**: 50 timesteps, 64 hidden units
- **Forecasting LSTM**: Multi-step ahead prediction
- **Training Parameters**: Adjustable epochs, batch size, learning rate

**3. Anomaly Detection Tab**

View real-time anomaly detection results:

- **Reconstruction Error**: AutoEncoder-based anomaly scores
- **Forecasting Deviation**: Prediction-based anomaly detection  
- **Combined Score**: Ensemble anomaly detection
- **Threshold Visualization**: Adjustable detection thresholds

Quick Data Analysis
-------------------

Let's run a quick analysis using the Python interface:

**Load Sample Data**

.. code-block:: python

   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   
   # Initialize data loader
   loader = CMAPSSDataLoader()
   
   # Load FD001 dataset (single fault mode, single operating condition)
   train_data, test_data = loader.load_dataset('FD001')
   
   print(f"Training engines: {len(train_data['unit_id'].unique())}")
   print(f"Test engines: {len(test_data['unit_id'].unique())}")
   print(f"Sensor columns: {train_data.columns.tolist()}")

**Expected Output:**

.. code-block:: text

   Training engines: 100
   Test engines: 100
   Sensor columns: ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3', 
                   's1', 's2', 's3', ..., 's21']

**Quick Visualization**

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot sensor data for first engine
   engine_1 = train_data[train_data['unit_id'] == 1]
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   axes = axes.flatten()
   
   sensors = ['s2', 's3', 's4', 's11']  # Key sensors
   for i, sensor in enumerate(sensors):
       axes[i].plot(engine_1['cycle'], engine_1[sensor])
       axes[i].set_title(f'Sensor {sensor}')
       axes[i].set_xlabel('Cycle')
       axes[i].set_ylabel('Value')
   
   plt.tight_layout()
   plt.show()

Train Your First Model
----------------------

**Quick AutoEncoder Training**

.. code-block:: python

   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
   
   # Initialize model
   model = LSTMAutoEncoder(
       sequence_length=50,
       n_features=21,
       encoding_dim=64
   )
   
   # Build model architecture
   model.build_model(input_shape=(50, 21))
   
   # Prepare training data
   X_train = loader.create_sequences(train_data, sequence_length=50)
   
   # Train model (quick training)
   history = model.train(
       X_train, 
       epochs=10,  # Use more epochs for better results
       batch_size=32,
       validation_split=0.2
   )
   
   print("✅ Model training completed!")

**Quick Anomaly Detection**

.. code-block:: python

   # Generate test sequences
   X_test = loader.create_sequences(test_data, sequence_length=50)
   
   # Detect anomalies
   reconstruction_errors = model.detect_anomalies(X_test)
   
   # Set threshold (can be optimized)
   threshold = np.percentile(reconstruction_errors, 95)
   anomalies = reconstruction_errors > threshold
   
   print(f"Detected {anomalies.sum()} anomalies out of {len(anomalies)} samples")
   print(f"Anomaly rate: {100 * anomalies.sum() / len(anomalies):.2f}%")

Interactive Dashboard Features
------------------------------

**Real-time Monitoring**

The dashboard updates in real-time as you:

- Upload new data files
- Adjust model parameters  
- Modify detection thresholds
- Select different engine units

**Key Interactive Elements**

- **Slider Controls**: Adjust thresholds and parameters
- **Dropdown Menus**: Select engines, sensors, and models
- **Interactive Plots**: Zoom, pan, and explore data
- **Real-time Updates**: See changes immediately

**Customization Options**

.. code-block:: python

   # Dashboard configuration (in app/config.py)
   CONFIG = {
       'model_params': {
           'sequence_length': 50,
           'encoding_dim': 64,
           'learning_rate': 0.001
       },
       'detection_params': {
           'threshold_percentile': 95,
           'window_size': 10
       },
       'visualization': {
           'plot_height': 400,
           'color_scheme': 'viridis'
       }
   }

Sample Results
--------------

After running the quick start, you should see:

**Performance Metrics**

.. code-block:: text

   AutoEncoder Performance:
   ├── Reconstruction MSE: 0.142
   ├── Detection F1-Score: 0.534
   ├── Precision: 0.423
   └── Recall: 0.721

   Forecasting Performance:
   ├── RUL RMSE: 14.2 cycles
   ├── Early Warning Rate: 67%
   └── False Positive Rate: 18%

**Visual Outputs**

- Sensor time series plots
- Anomaly detection charts  
- RUL prediction curves
- Model performance metrics

Troubleshooting
---------------

**Dashboard Won't Load**

.. code-block:: bash

   # Check if port is in use
   lsof -i :8501
   
   # Use different port
   streamlit run app/app.py --server.port 8502

**Memory Issues**

.. code-block:: python

   # Reduce batch size
   model.train(X_train, batch_size=16)  # Instead of 32
   
   # Use smaller sequence length
   sequence_length = 30  # Instead of 50

**Model Training Slow**

.. code-block:: python

   # Enable GPU if available
   import tensorflow as tf
   print("GPU Available:", tf.config.list_physical_devices('GPU'))
   
   # Reduce model complexity
   model = LSTMAutoEncoder(encoding_dim=32)  # Instead of 64

Next Steps
----------

Now that you have TurboGuard running:

1. 🎯 **Build your first complete model**: :doc:`first_model`
2. 📚 **Learn data preprocessing**: :doc:`../user_guide/data_preprocessing`
3. 🔧 **Explore advanced features**: :doc:`../examples/advanced_usage`
4. 📖 **Check API reference**: :doc:`../api/index`

Tips for Success
----------------

💡 **Best Practices**

- Start with the FD001 dataset (simplest case)
- Use the dashboard for initial exploration
- Experiment with different thresholds
- Monitor both reconstruction and forecasting errors

🎯 **Key Metrics to Watch**

- Reconstruction error trends
- RUL prediction accuracy  
- False positive rates
- Early warning performance

Congratulations! You're now ready to dive deeper into TurboGuard! 🎉
