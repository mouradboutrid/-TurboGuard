Your First Model
================

In this tutorial, you'll build and train your first complete TurboGuard model for turbofan engine anomaly detection. We'll cover both LSTM AutoEncoder and Forecasting LSTM approaches.

🎯 What You'll Build
--------------------

By the end of this tutorial, you'll have:

- ✅ A trained LSTM AutoEncoder for anomaly detection
- ✅ A Forecasting LSTM for RUL prediction  
- ✅ A complete evaluation pipeline
- ✅ Visualization of results

Let's get started! 🚀

Step 1: Data Preparation
------------------------

**Load and Explore the Dataset**

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from src.LSTM_AutoEncoder.data_loader import DataLoader
   
   # Initialize data loader
   loader = DataLoader()
   
   # Load FD001 dataset
   train_data, test_data = loader.load_dataset('FD001')
   
   print("Dataset Overview:")
   print(f"Training engines: {train_data['unit_id'].nunique()}")
   print(f"Test engines: {test_data['unit_id'].nunique()}")
   print(f"Training cycles: {len(train_data)}")
   print(f"Test cycles: {len(test_data)}")

**Expected Output:**

.. code-block:: text

   Dataset Overview:
   Training engines: 100
   Test engines: 100
   Training cycles: 20631
   Test cycles: 13096

**Explore Sensor Data**

.. code-block:: python

   # Look at sensor columns
   sensor_cols = [col for col in train_data.columns if col.startswith('s')]
   print(f"Available sensors: {len(sensor_cols)}")
   print(f"Sensor names: {sensor_cols}")
   
   # Check data types and missing values
   print("\nData Info:")
   print(train_data.info())

**Data Preprocessing**

.. code-block:: python

   from src.LSTM_AutoEncoder.preprocessor import DataPreprocessor
   
   # Initialize preprocessor
   preprocessor = DataPreprocessor()
   
   # Normalize the data
   train_normalized = preprocessor.fit_transform(train_data)
   test_normalized = preprocessor.transform(test_data)
   
   print("✅ Data preprocessing completed!")
   print(f"Normalized training shape: {train_normalized.shape}")

Step 2: Build LSTM AutoEncoder
------------------------------

**Model Architecture**

.. code-block:: python

   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
   
   
   # Initialize AutoEncoder
   autoencoder = LSTMAutoEncoder()
   
   # Build model architecture
   autoencoder.build_model(input_shape=(SEQUENCE_LENGTH, N_FEATURES))
   
   # Display model summary
   print("Model Architecture:")
   autoencoder.model.summary()

**Expected Architecture:**

.. code-block:: text

   Model: "lstm_autoencoder"
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #   
   =================================================================
   lstm_encoder (LSTM)          (None, 64)                22016     
   repeat_vector (RepeatVector) (None, 50, 64)            0         
   lstm_decoder (LSTM)          (None, 50, 64)            33024     
   time_distributed (TimeDistr) (None, 50, 21)            1365      
   =================================================================
   Total params: 56,405
   Trainable params: 56,405

**Prepare Training Sequences**

.. code-block:: python

   # Create sequences for training
   X_train = loader.create_sequences(
       train_normalized, 
       sequence_length=SEQUENCE_LENGTH
   )
   
   X_test = loader.create_sequences(
       test_normalized, 
       sequence_length=SEQUENCE_LENGTH
   )
   
   print(f"Training sequences: {X_train.shape}")
   print(f"Test sequences: {X_test.shape}")

Step 3: Train the AutoEncoder
-----------------------------

**Training Configuration**

.. code-block:: python

   # Training parameters
   EPOCHS = 50
   BATCH_SIZE = 32
   VALIDATION_SPLIT = 0.2
   
   # Train the model
   print("🚀 Starting AutoEncoder training...")
   
   history = autoencoder.train(
       X_train,
       epochs=EPOCHS,
       batch_size=BATCH_SIZE,
       validation_split=VALIDATION_SPLIT,
       verbose=0
   )
   
   print("✅ AutoEncoder training completed!")

**Monitor Training Progress**

.. code-block:: python

   # Plot training history
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 2, 1)
   plt.plot(history.history['loss'], label='Training Loss')
   plt.plot(history.history['val_loss'], label='Validation Loss')
   plt.title('Model Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   
   plt.subplot(1, 2, 2)
   plt.plot(history.history['mae'], label='Training MAE')
   plt.plot(history.history['val_mae'], label='Validation MAE')
   plt.title('Model MAE')
   plt.xlabel('Epoch')
   plt.ylabel('MAE')
   plt.legend()
   
   plt.tight_layout()
   plt.show()

Step 4: Anomaly Detection
-------------------------

**Generate Predictions**

.. code-block:: python

   # Get reconstructions for test data
   X_test_pred = autoencoder.model.predict(X_test)
   
   # Calculate reconstruction errors
   reconstruction_errors = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))
   
   print(f"Reconstruction errors shape: {reconstruction_errors.shape}")
   print(f"Mean reconstruction error: {reconstruction_errors.mean():.4f}")
   print(f"Std reconstruction error: {reconstruction_errors.std():.4f}")

**Set Anomaly Threshold**

.. code-block:: python

   # Calculate threshold using training data
   X_train_pred = autoencoder.model.predict(X_train)
   train_errors = np.mean(np.square(X_train - X_train_pred), axis=(1, 2))
   
   # Use 95th percentile as threshold
   threshold = np.percentile(train_errors, 95)
   
   print(f"Anomaly threshold: {threshold:.4f}")
   
   # Detect anomalies
   anomalies = reconstruction_errors > threshold
   anomaly_rate = anomalies.sum() / len(anomalies)
   
   print(f"Detected anomalies: {anomalies.sum()}/{len(anomalies)}")
   print(f"Anomaly rate: {anomaly_rate:.2%}")

**Visualize Anomaly Detection**

.. code-block:: python

   plt.figure(figsize=(15, 5))
   
   plt.subplot(1, 2, 1)
   plt.hist(train_errors, bins=50, alpha=0.7, label='Training Errors')
   plt.hist(reconstruction_errors, bins=50, alpha=0.7, label='Test Errors')
   plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
   plt.xlabel('Reconstruction Error')
   plt.ylabel('Frequency')
   plt.title('Error Distribution')
   plt.legend()
   
   plt.subplot(1, 2, 2)
   plt.plot(reconstruction_errors, alpha=0.7)
   plt.scatter(np.where(anomalies)[0], reconstruction_errors[anomalies], 
               color='red', s=10, label='Anomalies')
   plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
   plt.xlabel('Sample Index')
   plt.ylabel('Reconstruction Error')
   plt.title('Anomaly Detection Results')
   plt.legend()
   
   plt.tight_layout()
   plt.show()

Step 5: Build Forecasting LSTM
-------------------------------

**Forecasting Model Setup**

.. code-block:: python

   from src.Forecasting_LSTM.forecasting_lstm import PrognosticLSTMModel
   
   # Initialize forecasting model
   forecaster = PrognosticLSTMModel(
       n_features=N_FEATURES,
       sequence_length=SEQUENCE_LENGTH
   )
   
   # Build model
   forecaster.build_model(input_shape=(SEQUENCE_LENGTH, N_FEATURES))
   
   print("Forecasting Model Architecture:")
   forecaster.model.summary()

**Prepare Forecasting Data**

.. code-block:: python

   # Initialize your data processor and load normalized data
processor = DataProcessor()
df = processor.load_cmapss_data('/path/to/FD001.txt')

# Extract sensor columns and operational mode as numpy arrays
sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
data = df[sensor_cols].values
modes = df['op_mode'].values

# Parameters
SEQUENCE_LENGTH = 30

# Create sequences with modes using your model method
model = PrognosticLSTMModel(n_features=data.shape[1], sequence_length=SEQUENCE_LENGTH)
X, y, mode_seq = model.create_sequences(data, modes=modes)

print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Mode sequence shape: {mode_seq.shape}")

# Split train/val (example: 80% train)
split_idx = int(0.8 * len(X))
X_train, y_train, modes_train = X[:split_idx], y[:split_idx], mode_seq[:split_idx]
X_val, y_val, modes_val = X[split_idx:], y[split_idx:], mode_seq[split_idx:])

**Train Forecasting Model**

.. code-block:: python

   print("🚀 Starting Forecasting LSTM training...")
   
   model.build_model()
   history = model.train(
       X_train, y_train,
       X_val, y_val,
       epochs=30,
       batch_size=32,
       modes_train=modes_train,
       modes_val=modes_val
       )

print("Training completed!")


Step 6: Model Evaluation
------------------------

**Comprehensive Performance Metrics**

.. code-block:: python

   from sklearn.metrics import classification_report, confusion_matrix
   
   # For AutoEncoder anomaly detection
   # Create binary labels (assuming last 30% of engine life is anomalous)
   def create_binary_labels(data):
       labels = []
       for unit_id in data['unit_id'].unique():
           unit_data = data[data['unit_id'] == unit_id]
           unit_length = len(unit_data)
           # Last 30% cycles are considered anomalous
           anomaly_start = int(0.7 * unit_length)
           unit_labels = [0] * anomaly_start + [1] * (unit_length - anomaly_start)
           labels.extend(unit_labels)
       return np.array(labels)
   
   # Create ground truth labels for sequences
   test_labels = create_binary_labels(test_normalized)
   # Align with sequence data (simplified)
   sequence_labels = test_labels[SEQUENCE_LENGTH-1:][:len(anomalies)]
   
   # Classification report
   print("AutoEncoder Anomaly Detection Performance:")
   print(classification_report(sequence_labels, anomalies.astype(int)))

**Performance Summary**

.. code-block:: python

   # Create comprehensive performance summary
   performance_summary = {
       'AutoEncoder': {
           'Reconstruction MSE': np.mean(reconstruction_errors),
           'Detection Accuracy': np.mean(sequence_labels == anomalies.astype(int)),
           'Anomaly Rate': anomaly_rate,
           'Threshold': threshold
       },
       'Forecasting LSTM': {
           'RUL RMSE': rmse,
           'RUL MAE': mae,
           'Training Loss': forecast_history.history['loss'][-1],
           'Validation Loss': forecast_history.history['val_loss'][-1]
       }
   }
   
   print("\n" + "="*50)
   print("FINAL PERFORMANCE SUMMARY")
   print("="*50)
   
   for model_name, metrics in performance_summary.items():
       print(f"\n{model_name}:")
       for metric_name, value in metrics.items():
           if isinstance(value, float):
               print(f"  ├── {metric_name}: {value:.4f}")
           else:
               print(f"  ├── {metric_name}: {value}")

Step 7: Save Your Models
------------------------

**Save Trained Models**

.. code-block:: python

   import os
   from datetime import datetime
   
   # Create models directory
   os.makedirs('models/trained', exist_ok=True)
   
   # Generate timestamp for model versioning
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
   # Save AutoEncoder
   autoencoder_path = f'models/trained/autoencoder_FD001_{timestamp}.h5'
   autoencoder.model.save(autoencoder_path)
   print(f"✅ AutoEncoder saved to: {autoencoder_path}")
   
   # Save Forecasting LSTM
   forecaster_path = f'models/trained/forecaster_FD001_{timestamp}.h5'
   forecaster.model.save(forecaster_path)
   print(f"✅ Forecasting LSTM saved to: {forecaster_path}")
   
   # Save preprocessing parameters
   import pickle
   
   preprocessor_path = f'models/trained/preprocessor_FD001_{timestamp}.pkl'
   with open(preprocessor_path, 'wb') as f:
       pickle.dump(preprocessor, f)
   print(f"✅ Preprocessor saved to: {preprocessor_path}")

**Save Model Configuration**

.. code-block:: python

   import json
   
   # Model configuration
   model_config = {
       'dataset': 'FD001',
       'timestamp': timestamp,
       'autoencoder': {
           'sequence_length': SEQUENCE_LENGTH,
           'n_features': N_FEATURES,
           'encoding_dim': ENCODING_DIM,
           'epochs': EPOCHS,
           'batch_size': BATCH_SIZE,
           'threshold': float(threshold)
       },
       'forecaster': {
           'sequence_length': SEQUENCE_LENGTH,
           'n_features': N_FEATURES,
           'forecast_horizon': 10,
           'epochs': 30,
           'batch_size': 32
       },
       'performance': performance_summary
   }
   
   config_path = f'models/trained/config_FD001_{timestamp}.json'
   with open(config_path, 'w') as f:
       json.dump(model_config, f, indent=2)
   
   print(f"✅ Configuration saved to: {config_path}")

Step 8: Test Model Loading
--------------------------

**Load and Test Saved Models**

.. code-block:: python

   from tensorflow.keras.models import load_model
   
   # Load models
   loaded_autoencoder = load_model(autoencoder_path)
   loaded_forecaster = load_model(forecaster_path)
   
   # Load preprocessor
   with open(preprocessor_path, 'rb') as f:
       loaded_preprocessor = pickle.load(f)
   
   print("✅ All models loaded successfully!")
   
   # Test loaded models
   test_sample = X_test[:5]  # Test with 5 samples
   
   # Test AutoEncoder
   test_reconstruction = loaded_autoencoder.predict(test_sample)
   test_errors = np.mean(np.square(test_sample - test_reconstruction), axis=(1, 2))
   
   print(f"Test reconstruction errors: {test_errors}")
   
   # Test Forecaster
   test_forecast = loaded_forecaster.predict(test_sample)
   print(f"Test forecast shape: {test_forecast.shape}")


Step 9: Visualization Dashboard
--------------------------------

**Create Summary Visualization**

.. code-block:: python

   def create_model_dashboard(results, title="TurboGuard Model Results"):
       """Create comprehensive visualization dashboard"""
       
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       fig.suptitle(title, fontsize=16, fontweight='bold')
       
       # Plot 1: Reconstruction Errors
       axes[0, 0].plot(results['reconstruction_errors'])
       axes[0, 0].axhline(results['threshold'], color='red', linestyle='--', 
                         label=f'Threshold: {results["threshold"]:.4f}')
       axes[0, 0].scatter(np.where(results['anomalies'])[0], 
                         results['reconstruction_errors'][results['anomalies']], 
                         color='red', s=20, alpha=0.7, label='Anomalies')
       axes[0, 0].set_title('Reconstruction Error Timeline')
       axes[0, 0].set_xlabel('Sample Index')
       axes[0, 0].set_ylabel('Reconstruction Error')
       axes[0, 0].legend()
       
       # Plot 2: RUL Estimates
       axes[0, 1].plot(results['rul_estimates'])
       axes[0, 1].set_title('RUL Estimates Timeline')
       axes[0, 1].set_xlabel('Sample Index')
       axes[0, 1].set_ylabel('RUL (cycles)')
       
       # Plot 3: Error Distribution
       axes[0, 2].hist(results['reconstruction_errors'], bins=50, alpha=0.7)
       axes[0, 2].axvline(results['threshold'], color='red', linestyle='--', 
                         label='Threshold')
       axes[0, 2].set_title('Reconstruction Error Distribution')
       axes[0, 2].set_xlabel('Reconstruction Error')
       axes[0, 2].set_ylabel('Frequency')
       axes[0, 2].legend()
       
       # Plot 4: Anomaly Rate Over Time
       window_size = 100
       anomaly_rate_timeline = []
       for i in range(window_size, len(results['anomalies'])):
           window_anomalies = results['anomalies'][i-window_size:i]
           rate = window_anomalies.sum() / window_size
           anomaly_rate_timeline.append(rate)
       
       axes[1, 0].plot(anomaly_rate_timeline)
       axes[1, 0].set_title(f'Anomaly Rate (Rolling Window: {window_size})')
       axes[1, 0].set_xlabel('Sample Index')
       axes[1, 0].set_ylabel('Anomaly Rate')
       
       # Plot 5: RUL Distribution
       axes[1, 1].hist(results['rul_estimates'], bins=30, alpha=0.7)
       axes[1, 1].set_title('RUL Estimates Distribution')
       axes[1, 1].set_xlabel('RUL (cycles)')
       axes[1, 1].set_ylabel('Frequency')
       
       # Plot 6: Anomaly vs RUL Correlation
       normal_rul = results['rul_estimates'][~results['anomalies']]
       anomaly_rul = results['rul_estimates'][results['anomalies']]
       
       axes[1, 2].boxplot([normal_rul, anomaly_rul], labels=['Normal', 'Anomaly'])
       axes[1, 2].set_title('RUL Distribution: Normal vs Anomaly')
       axes[1, 2].set_ylabel('RUL (cycles)')
       
       plt.tight_layout()
       plt.show()
       
       return fig
   
   # Create dashboard for our results
   dashboard = create_model_dashboard(sample_results, "Your First TurboGuard Model Results")

Congratulations! 🎉
-------------------

You've successfully built your first complete TurboGuard model! Here's what you accomplished:

✅ **Data Loading & Preprocessing**
- Loaded CMAPSS FD001 dataset  
- Normalized sensor data
- Created sequential training data

✅ **LSTM AutoEncoder**
- Built dual LSTM architecture
- Trained for anomaly detection
- Achieved reconstruction-based anomaly detection

✅ **Forecasting LSTM**  
- Built forecasting model
- Trained for multi-step prediction

✅ **Model Evaluation**
- Comprehensive performance metrics
- Visualization dashboards
- Model saving and loading

✅ **Production Pipeline**
- Complete prediction function
- Model configuration management
- Reusable prediction pipeline

Key Takeaways
-------------

🎯 **Performance Insights**

- AutoEncoder effectively captures normal engine behavior patterns
- Reconstruction errors provide reliable anomaly indicators  
- Forecasting LSTM enables proactive maintenance planning
- Combined approach improves overall detection reliability

📊 **Best Practices Learned**

- Proper sequence length is crucial (50 timesteps works well)
- Threshold selection significantly impacts performance
- Model ensembling improves robustness
- Regular model retraining maintains accuracy

Next Steps
----------

Now that you have a working model, explore these advanced topics:

1. 🔧 **Hyperparameter Tuning**: :doc:`../user_guide/model_training`
2. 📊 **Advanced Visualization**: :doc:`../user_guide/visualization`  
3. 🚀 **Production Deployment**: :doc:`../examples/advanced_usage`
4. 📈 **Multi-Dataset Training**: Try FD002, FD003, FD004 datasets
5. 🎯 **Custom Thresholds**: Implement adaptive thresholding

Troubleshooting
---------------

**Common Issues and Solutions**

**Issue**: Model overfitting (training loss much lower than validation loss)
**Solution**: Add dropout layers, reduce model complexity, or increase data

**Issue**: Poor anomaly detection performance  
**Solution**: Adjust threshold, try different sequence lengths, or add more training data


**Issue**: Memory errors during training
**Solution**: Reduce batch size, use gradient accumulation, or train on smaller sequences

Resources
---------

📚 **Further Reading**
- :doc:`../user_guide/index` - Detailed user guide
- :doc:`../api/index` - Complete reference  
- :doc:`../examples/index` - More examples and use cases

🛠️ **Tools and Extensions**
- TensorBoard for training visualization
- MLflow for experiment tracking
- Docker for containerized deployment

You're now ready to build production-grade predictive maintenance systems with TurboGuard! 🚀
