# TimeSeries AutoEncoder Anomaly Detection Pipeline
## Project Structure and Workflow Documentation

---

## 📋 Project Overview

This project implements a comprehensive anomaly detection system for time series data using two complementary approaches:
1. **LSTM AutoEncoder** - Reconstruction-based anomaly detection
2. **Forecasting LSTM** - Prediction-based anomaly detection

The system is designed to work with the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset for aircraft engine health monitoring and failure prediction.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                │
├─────────────────────────────────────────────────────────────────┤
│  CMAPSSDataLoader → CMAPSSPreprocessor → Feature Selection      │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DUAL MODEL APPROACH                            │
├─────────────────────┬───────────────────────────────────────────┤
│  AUTOENCODER BRANCH │           FORECASTER BRANCH               │
│                     │                                           │
│  LSTMAutoencoder    │     PrognosticLSTMModel                   │
│  AnomalyDetector    │     AnomalyDetectionEngine                │
│  Visualizer         │     PrognosticVisualizationSuite          │
└─────────────────────┴───────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│              ANALYSIS & DEPLOYMENT                              │
├─────────────────────────────────────────────────────────────────┤
│  CMAPSSAnomalyAnalyzer → ModelManager → Streamlit App          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Workflow

### Phase 1: Data Ingestion and Preprocessing

#### 1.1 Data Loading (`CMAPSSDataLoader`)
- **Purpose**: Load and structure CMAPSS dataset files
- **Key Functions**:
  - `load_dataset()`: Load train/test/RUL files
  - `get_dataset()`: Retrieve processed datasets
- **Input**: Raw CMAPSS CSV files
- **Output**: Structured pandas DataFrames

#### 1.2 Data Preprocessing (`CMAPSSPreprocessor`)
- **Purpose**: Clean, normalize, and prepare data for modeling
- **Key Functions**:
  - `preprocess_data()`: Main preprocessing pipeline
  - `_normalize_data()`: Scale features using MinMax/Standard scaling
  - `split_by_engine()`: Separate data by engine units
  - `create_sequences()`: Generate time-windowed sequences
- **Transformations**:
  - Feature normalization
  - Sequence creation with sliding windows
  - Train/validation splits
  - Engine-wise data organization

### Phase 2: Feature Engineering and Selection

#### 2.1 Prognostic Feature Selection (`PrognosticFeatureSelector`)
- **Purpose**: Identify most relevant features for anomaly detection
- **Key Functions**:
  - `calculate_prognostic_relevance()`: Compute feature importance scores
- **Techniques**:
  - Correlation analysis with RUL (Remaining Useful Life)
  - Statistical significance testing
  - Domain knowledge integration

#### 2.2 Data Processing (`CMAPSSDataProcessor`)
- **Purpose**: Advanced data processing for forecasting models
- **Key Functions**:
  - `load_cmapss_data()`: Enhanced data loading
  - `remove_constant_sensors()`: Filter non-informative sensors
- **Features**:
  - Sensor data validation
  - Missing value handling
  - Outlier detection and treatment

### Phase 3: Model Training and Prediction

#### 3.1 AutoEncoder Branch

##### LSTMAutoencoder
- **Architecture**: Encoder-Decoder LSTM network
- **Key Functions**:
  - `build_model()`: Construct autoencoder architecture
  - `train()`: Train the reconstruction model
  - `predict()`: Generate reconstructions
  - `encode()`: Extract latent representations
- **Features**:
  - Multi-layer LSTM cells
  - Dropout regularization
  - Early stopping
  - Model checkpointing

##### AnomalyDetector
- **Purpose**: Multiple anomaly detection algorithms
- **Key Functions**:
  - `detect_lstm_anomalies()`: Reconstruction error-based detection
  - `detect_statistical_anomalies()`: Statistical outlier detection
  - `detect_wavelet_anomalies()`: Wavelet transform-based detection
- **Algorithms**:
  - Reconstruction error thresholding
  - Z-score statistical analysis
  - Wavelet coefficient analysis
  - Ensemble voting mechanism

#### 3.2 Forecasting Branch

##### PrognosticLSTMModel
- **Architecture**: Sequence-to-sequence LSTM for forecasting
- **Key Functions**:
  - `build_model()`: Create forecasting architecture
  - `create_sequences()`: Prepare sequential data
  - `train()`: Train forecasting model
  - `save_model()` / `load_model()`: Model persistence
- **Features**:
  - Multi-step ahead forecasting
  - Attention mechanisms
  - Bidirectional LSTM layers

##### AnomalyDetectionEngine
- **Purpose**: Forecast-based anomaly detection
- **Key Functions**:
  - `calculate_reconstruction_errors()`: Compute prediction errors
  - `update_threshold()`: Dynamic threshold adjustment
  - `detect_anomalies()`: Identify anomalous patterns
- **Techniques**:
  - Adaptive thresholding
  - Prediction confidence intervals
  - Temporal pattern analysis

### Phase 4: Analysis and Monitoring

#### 4.1 Health Monitoring (`CMAPSSPrognosticHealthMonitor`)
- **Purpose**: Comprehensive system health analysis
- **Key Functions**:
  - `prepare_sequence_data()`: Data preparation for analysis
  - `_create_sequences()`: Generate analysis sequences
  - `run_complete_analysis()`: Execute full health assessment
  - `_calculate_performance_metrics()`: Compute evaluation metrics
  - `_save_analysis_results()`: Persist analysis outputs
  - `load_trained_model()`: Load pre-trained models
  - `predict_anomalies()`: Generate anomaly predictions

#### 4.2 System Analysis (`CMAPSSAnomalyAnalyzer`)
- **Purpose**: High-level system analysis and comparison
- **Key Functions**:
  - `analyze_dataset()`: Comprehensive dataset analysis
  - `load_saved_model()`: Model loading and validation
  - `predict_anomalies()`: Anomaly prediction pipeline
  - `compare_all_datasets()`: Cross-dataset performance comparison
  - `print_performance_comparison()`: Performance reporting
  - `get_model_summary()`: Model architecture summary
  - `list_available_models()`: Available model inventory

### Phase 5: Visualization and Reporting

#### 5.1 AutoEncoder Visualization (`Visualizer`)
- **Key Functions**:
  - `plot_anomalies()`: Anomaly visualization with time series
  - `plot_training_history()`: Training metrics visualization
- **Visualizations**:
  - Time series with anomaly highlights
  - Reconstruction error plots
  - Training/validation loss curves
  - Feature importance heatmaps

#### 5.2 Prognostic Visualization (`PrognosticVisualizationSuite`)
- **Key Functions**:
  - `plot_dataset_overview()`: Dataset summary visualizations
  - `save_plot()`: Plot persistence utilities
  - `plot_training_progress()`: Training progress monitoring
  - `plot_anomaly_results()`: Anomaly detection results
- **Visualizations**:
  - Multi-sensor time series plots
  - Forecasting accuracy charts
  - Anomaly detection performance metrics
  - Health indicator trends

### Phase 6: Model Management and Deployment

#### 6.1 Model Persistence (`ModelManager`)
- **Key Functions**:
  - `save_model_package()`: Save complete model packages
  - `load_model_package()`: Load saved model packages
- **Features**:
  - Model versioning
  - Metadata preservation
  - Preprocessing pipeline serialization
  - Configuration management

#### 6.2 Testing and Validation (`CmapssAnomaliePredector`)
- **Purpose**: Model testing and validation utilities
- **Features**:
  - Batch prediction testing
  - Performance benchmarking
  - Model comparison utilities
  - Validation pipeline

#### 6.3 Web Application (`app.py`)
- **Platform**: Streamlit-based web interface
- **Features**:
  - Interactive data exploration
  - Real-time anomaly detection
  - Model comparison dashboard
  - Performance metrics visualization
  - Export capabilities

---

## 📊 Data Flow Diagram

```
Raw CMAPSS Data
      ↓
CMAPSSDataLoader
      ↓
CMAPSSPreprocessor
      ↓
┌─────────────────┬─────────────────┐
│   AutoEncoder   │   Forecaster    │
│     Branch      │     Branch      │
│        ↓        │        ↓        │
│ LSTMAutoencoder │ PrognosticLSTM  │
│        ↓        │        ↓        │
│ AnomalyDetector │ AnomalyEngine   │
│        ↓        │        ↓        │
│   Visualizer    │ PrognosticViz   │
└─────────────────┴─────────────────┘
              ↓
      ModelManager
              ↓
    CMAPSSAnomalyAnalyzer
              ↓
        Streamlit App
```

---

## 🎯 Key Features and Capabilities

### Dual Approach Strategy
- **Reconstruction-based**: Detects anomalies through reconstruction errors
- **Prediction-based**: Identifies anomalies through forecasting deviations
- **Ensemble Methods**: Combines multiple detection algorithms for robust results

### Advanced Analytics
- **Multi-sensor Analysis**: Handles 21 sensor readings simultaneously
- **Temporal Pattern Recognition**: Captures complex time-dependent relationships
- **Adaptive Thresholding**: Dynamic adjustment based on data characteristics
- **Performance Monitoring**: Comprehensive metrics and KPI tracking

### Scalability and Deployment
- **Modular Architecture**: Easy to extend and modify
- **Model Versioning**: Track and manage different model versions
- **Web Interface**: User-friendly dashboard for non-technical users
- **Batch Processing**: Handle large datasets efficiently

---

## 🚀 Usage Pipeline

### 1. Data Preparation
```python
# Load and preprocess data
loader = CMAPSSDataLoader(data_path)
preprocessor = CMAPSSPreprocessor()
data = loader.load_dataset("FD001", train_file, test_file)
processed_data = preprocessor.preprocess_data(data)
```

### 2. Model Training
```python
# AutoEncoder approach
autoencoder = LSTMAutoencoder()
autoencoder.train(processed_data)

# Forecasting approach
forecaster = PrognosticLSTMModel()
forecaster.train(processed_data)
```

### 3. Anomaly Detection
```python
# Detect anomalies using both approaches
detector = AnomalyDetector()
ae_anomalies = detector.detect_lstm_anomalies(data)

engine = AnomalyDetectionEngine()
forecast_anomalies = engine.detect_anomalies(data)
```

### 4. Analysis and Visualization
```python
# Comprehensive analysis
analyzer = CMAPSSAnomalyAnalyzer()
results = analyzer.run_complete_analysis(data)

# Visualization
visualizer = Visualizer()
visualizer.plot_anomalies(results)
```

### 5. Deployment
```python
# Save models
manager = ModelManager()
manager.save_model_package(models, "production_v1")

# Launch web app
streamlit run app.py
```

---

## 📈 Performance Metrics

### Model Evaluation
- **Precision/Recall/F1-Score**: Classification performance
- **ROC-AUC**: Binary classification effectiveness
- **Mean Absolute Error (MAE)**: Reconstruction/prediction accuracy
- **Root Mean Square Error (RMSE)**: Error magnitude assessment

### System Performance
- **Processing Speed**: Data throughput and latency
- **Memory Usage**: Resource efficiency monitoring
- **Scalability**: Performance under increasing data loads
- **Reliability**: System uptime and error rates

---

## 🔧 Configuration Management

### Model Hyperparameters
- **LSTM Units**: 50-200 units per layer
- **Sequence Length**: 30-50 time steps
- **Learning Rate**: 0.001-0.01
- **Batch Size**: 32-128 samples
- **Dropout Rate**: 0.2-0.5

### Detection Thresholds
- **Reconstruction Error**: Dynamic percentile-based
- **Statistical Outliers**: Z-score > 3.0
- **Wavelet Coefficients**: Adaptive threshold
- **Ensemble Voting**: Majority consensus

---

## 📚 Dependencies and Requirements

### Core Libraries
- TensorFlow/Keras: Deep learning framework
- NumPy/Pandas: Data manipulation
- Scikit-learn: Machine learning utilities
- Matplotlib/Seaborn: Visualization

### Specialized Libraries
- PyWavelets: Wavelet analysis
- Streamlit: Web application framework
- Joblib: Model serialization
- Plotly: Interactive visualizations

---

## 🎯 Future Enhancements

### Advanced Algorithms
- Transformer-based architectures
- Variational autoencoders
- Graph neural networks for sensor relationships
- Federated learning capabilities

### System Improvements
- Real-time streaming data processing
- Automated model retraining
- Advanced ensemble methods
- Cloud deployment optimization

---

## 📞 Project Structure Summary

This pipeline provides a complete end-to-end solution for time series anomaly detection with:
- **Comprehensive data processing** capabilities
- **Dual model approach** for robust detection
- **Advanced visualization** and analysis tools
- **Production-ready deployment** options
- **Extensive evaluation** and monitoring features

The modular design ensures easy maintenance, extensibility, and scalability for industrial applications in predictive maintenance and health monitoring systems.
