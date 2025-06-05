# 🔥 Turbofan Engine Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning framework for predictive maintenance and anomaly detection in turbofan engines using dual LSTM architectures and the CMAPSS dataset.

## 🎯 Overview

This project implements a comprehensive anomaly detection and forecasting system for turbofan engines by leveraging two complementary deep learning approaches:

- **LSTM AutoEncoder**: Reconstruction-based anomaly detection through sequence-to-sequence learning
- **Forecasting LSTM**: Next-step prediction for early fault detection.

The system provides robust, interpretable insights into engine health, enabling proactive maintenance strategies.

## ✨ Key Features

- **Dual Model Architecture**: Combines reconstruction and forecasting approaches for comprehensive anomaly detection
- **Real-time Monitoring**: Interactive Streamlit dashboard for live engine health visualization
- **Multivariate Analysis**: Handles 21 sensor channels with temporal dependencies and inter-correlations
- **Scalable Pipeline**: Modular design supporting both research and production deployment
- **Advanced Preprocessing**: Robust data normalization and sequence generation
- **Multiple Detection Methods**: LSTM-based, statistical anomaly detection
- **Performance Metrics**: Comprehensive evaluation with MSE, MAE, precision-recall, and accuracy

## 🏗️ Architecture

### LSTM AutoEncoder Model
```
CMAPSSDataLoader → CMAPSSPreprocessor → LSTMAutoEncoder → AnomalyDetector
                                                                ↓
                        ModelManager ← Visualizer ← CMAPSSAnomalyAnalyzer
```

### Forecasting LSTM Model
```
CMAPSSDataProcessor → PrognosticFeatureSelector → PrognosticLSTMModel
                                                         ↓
AnomalyDetectionEngine ← PrognosticVisualizationSuite ← CMAPSSPrognosticHealthMonitor
```

## 📁 Repository Structure

```
TurboGuard/
├── app/                       
│   ├──analyzer_app.py
│   ├──app.py
│   ├──autoencoder_anomaly_detector_app.py
│   ├──forecaster_anomaly_predictor_app.py
│   ├──loader_app.py         
│   └── preprocessor_app.py                                   
├── data/
│   ├──RUL_FD00X.txt
│   ├──test_FD00X.txt
│   ├──train_FD00X.txt        
│   └── readme.txt
├── data_overview/
├── results/
│   ├── autoencoder__/           
│   └── forecaster/        
├── src/                       
│   ├── LSTM_AutoEncoder/
│   │    ├──anomaly_analyzer.py
│   │    ├──anomaly_detector.py
│   │    ├──data_loader.py
│   │    ├──data_preprocessor.py
│   │    ├──lstm_autoencoder.py
│   │    ├──model_manager.py
│   │    └── visualizer.py
│   ├── Forecasting_LSTM/
│   │    ├──anomaly_detection_engine.py
│   │    ├──forecasting_data_processor.py
│   │    ├──main_training_.py
│   │    ├──prognostic_LSTMModel.py
│   │    ├──prognostic_feature_selector.py
│   │    ├──prognostic_health_monitor.py
│   │    └── prognostic_visualization_suite.py                    
│   ├── forecaster_anomaly_predictor_test.py       
│   └── autoencoder_anomaly_predictor_test.py                
├── trained_models/            
│   ├── autoencoder_models/               
│   └── forecaster_model/           
├── requirements.txt            
├── README.md
├── Damage Propagation Modeling.pdf                 
└── LICENSE                    
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mouradboutrid/-TurboGuard.git
   cd TurboGuard
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Launch Interactive Dashboard
```bash
streamlit run app.py
```

#### Train Models Programmatically
```python
from src.modeling import LSTMAutoencoder, PrognosticLSTMModel
from src.data_processing import CMAPSSDataLoader

# Load and preprocess data
loader = CMAPSSDataLoader()
data = loader.load_dataset('FD001')

# Train AutoEncoder
autoencoder = LSTMAutoencoder()
autoencoder.build_model(input_shape=(50, 21))
autoencoder.train(data)

# Detect anomalies
anomalies = autoencoder.detect_anomalies(test_data)
```

## 🛠️ Core Components

### Data Processing
- **DataLoader**: Handles CMAPSS dataset ingestion and management
- **DataPreprocessor**: Implements normalization, sequencing, and feature engineering
- **PrognosticFeatureSelector**: Calculates prognostic relevance for optimal feature selection

### Model Architectures
- **LSTMAutoencoder**: Deep LSTM encoder-decoder for reconstruction-based anomaly detection
- **PrognosticLSTMModel**: Multi-step forecasting LSTM with operational context integration

### Anomaly Detection
- **AnomalyDetector**: Multi-method anomaly detection (LSTM, statistical, wavelet)
- **AnomalyDetectionEngine**: Real-time anomaly scoring and threshold management

### Visualization & Analysis
- **Visualizer**: Comprehensive plotting utilities for anomalies and training metrics
- **PrognosticVisualizationSuite**: Advanced visualization for prognostic analysis
- **AnomalyAnalyzer**: Complete analysis pipeline with performance comparison

## 📊 Dataset Information

**CMAPSS Dataset** (Commercial Modular Aero-Propulsion System Simulation by NASA)

- **21 Sensor Channels**: Fan speed, core speed, turbine temperatures, pressures, fuel flow, vibrations
- **Multiple Operational Modes**: Various engine configurations and fault conditions
- **Variable Cycle Lengths**: Real-world variability in engine run-to-failure data
- **Benchmark Dataset**: Widely used in prognostics and health management research

## 📈 Performance Metrics

The system is evaluated on CMAPSS FD004 test sets using:

- **Reconstruction Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Anomaly Detection**: Precision, Recall, F1-Score, AUC-ROC
- **Robustness**: Performance across multiple fault modes and operational profiles

## 🔧 Advanced Features

### Multiple Anomaly Detection Methods
- **LSTM-based**: Reconstruction error analysis
- **Statistical**: Distribution-based anomaly scoring

### Model Management
- **ModelManager**: Serialization and deserialization of trained models
- **Version Control**: Track model performance across iterations
- **Ensemble Methods**: Combine multiple detection approaches

### Real-time Capabilities
- **Streaming Processing**: Handle live sensor data streams
- **Dynamic Thresholding**: Adaptive anomaly thresholds
- **Alert System**: Configurable anomaly notifications


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

**Boutrid Mourad & Kassimi Achraf** - AI Engineering Students

- 📧 Email: muurad.boutrid@gmail.com
- 📧 Email: ac.kassimi@edu.umi.ac.ma
- 🔗 LinkedIn: [https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/mourad-boutrid-981659336)
- 🔗 LinkedIn: [https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/achraf-kassimi-605418285)

## 🙏 Acknowledgments

- NASA for providing the CMAPSS dataset
- Contributors to the open-source libraries used in this project


⭐ **Star this repository if you find it helpful!**
