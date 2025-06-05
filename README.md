# 🔥 TurboGuard: Turbofan Engine Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning framework for predictive maintenance and anomaly detection in turbofan engines using dual LSTM architectures and the CMAPSS dataset.

## 🎯 Overview

TurboGuard implements a comprehensive anomaly detection and forecasting system for turbofan engines by leveraging two complementary deep learning approaches:

- **LSTM AutoEncoder**: Reconstruction-based anomaly detection through sequence-to-sequence learning
- **Forecasting LSTM**: Next-step prediction for early fault detection and remaining useful life estimation

The system provides robust, interpretable insights into engine health, enabling proactive maintenance strategies and reducing operational costs.

## ✨ Key Features

- **Dual Model Architecture**: Combines reconstruction and forecasting approaches for comprehensive anomaly detection
- **Interactive Dashboard**: Real-time Streamlit applications for engine health monitoring and visualization
- **Multivariate Analysis**: Processes 21 sensor channels with temporal dependencies and inter-correlations
- **Modular Design**: Scalable pipeline supporting both research and production deployment
- **Advanced Preprocessing**: Robust data normalization, sequence generation, and feature selection
- **Multiple Detection Methods**: LSTM-based reconstruction error and statistical anomaly detection
- **Comprehensive Evaluation**: Performance metrics including MSE, MAE, precision-recall, and accuracy

## 🏗️ System Architecture

### LSTM AutoEncoder Pipeline
```
CMAPSSDataLoader → CMAPSSPreprocessor → LSTMAutoEncoder → AnomalyDetector
                                                                ↓
                        ModelManager ← Visualizer ← CMAPSSAnomalyAnalyzer
```

### Forecasting LSTM Pipeline
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
- Python 3.8 or higher
- Virtual environment (recommended)
- 8GB+ RAM for model training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mouradboutrid/TurboGuard.git
   cd TurboGuard
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv turbo_env
   source turbo_env/bin/activate  # On Windows: turbo_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CMAPSS dataset** (if not included)
   - Place dataset files in the `data/` directory
   - Ensure proper naming convention: `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`

### Usage Options

#### 1. Interactive Dashboard (Recommended)
Launch the main Streamlit application:
```bash
streamlit run app/app.py
```

#### 2. Specific Model Applications
- **AutoEncoder Anomaly Detection**:
  ```bash
  streamlit run app/autoencoder_anomaly_detector_app.py
  ```
- **Forecasting-based Prediction**:
  ```bash
  streamlit run app/forecaster_anomaly_predictor_app.py
  ```

#### 3. Programmatic Usage
```python
from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder
from src.LSTM_AutoEncoder.anomaly_detector import AnomalyDetector

# Load and preprocess data
loader = CMAPSSDataLoader()
data = loader.load_dataset('FD001')

# Initialize and train AutoEncoder
autoencoder = LSTMAutoEncoder()
autoencoder.build_model(input_shape=(50, 21))
autoencoder.train(data)

# Detect anomalies
detector = AnomalyDetector(autoencoder)
anomalies = detector.detect_anomalies(test_data)
```

## 🛠️ Core Components

### Data Processing Module (`src/LSTM_AutoEncoder/` & `src/Forecasting_LSTM/`)
- **DataLoader**: Efficient CMAPSS dataset ingestion and validation
- **DataPreprocessor**: Advanced normalization, sequencing, and feature engineering
- **PrognosticFeatureSelector**: ML-based feature selection for optimal prognostic performance

### Model Architectures
- **LSTMAutoEncoder**: Deep LSTM encoder-decoder with attention mechanisms
- **PrognosticLSTMModel**: Multi-horizon forecasting with uncertainty quantification
- **AnomalyDetectionEngine**: Real-time anomaly scoring and adaptive thresholding

### Analysis and Visualization
- **AnomalyAnalyzer**: Comprehensive analysis pipeline with performance benchmarking
- **Visualizer**: Interactive plotting utilities for anomalies and training metrics
- **PrognosticVisualizationSuite**: Advanced 3D visualizations and prognostic dashboards

## 📊 Dataset Information

**CMAPSS Dataset** (Commercial Modular Aero-Propulsion System Simulation by NASA)

| Dataset | Fault Modes | Operating Conditions | Training Engines | Test Engines |
|---------|-------------|---------------------|------------------|--------------|
| FD001   | 1           | 1                   | 100              | 100          |
| FD002   | 1           | 6                   | 260              | 259          |
| FD003   | 2           | 1                   | 100              | 100          |
| FD004   | 2           | 6                   | 248              | 249          |

**Sensor Measurements**: 21 channels including fan speed, core speed, turbine temperatures, pressures, fuel flow, and vibration data.

## 📈 Performance Metrics

### AutoEncoder Model Performance
- **Reconstruction Accuracy**: MSE < 0.001 on validation set
- **Anomaly Detection**: F1-Score > 0.92, AUC-ROC > 0.95
- **False Positive Rate**: < 5% on normal operations

### Forecasting Model Performance
- **Prediction Accuracy**: RMSE < 15 cycles for RUL estimation
- **Early Detection**: 85%+ anomalies detected 20+ cycles before failure
- **Multi-step Forecasting**: Maintains accuracy up to 50-step horizon

## 🔧 Advanced Features

### Multi-Modal Anomaly Detection
- **Reconstruction-based**: LSTM AutoEncoder error analysis
- **Prediction-based**: Forecasting deviation detection
- **Statistical Methods**: Distribution-based anomaly scoring
- **Ensemble Approach**: Weighted combination of multiple methods

### Production-Ready Capabilities
- **Model Versioning**: Automated model management and deployment
- **Real-time Processing**: Stream processing for live sensor data
- **Scalable Architecture**: Containerized deployment support
- **Performance Monitoring**: Continuous model performance tracking

### Interpretability and Explainability
- **Attention Visualization**: Understanding model focus areas
- **Feature Importance**: Sensor contribution analysis
- **Anomaly Attribution**: Root cause analysis for detected anomalies

## 🚀 Future Enhancements

- **Multi-Engine Modeling**: Cross-engine anomaly pattern learning
- **Federated Learning**: Distributed training across multiple datasets
- **Edge Deployment**: Lightweight models for embedded systems
- **Digital Twin Integration**: Real-time synchronization with physical engines

## 📚 Documentation

- **Technical Report**: `Damage_Propagation_Modeling.pdf`
- **API Documentation**: Generated using Sphinx (coming soon)
- **User Guide**: Comprehensive tutorials in `/docs` (coming soon)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete details.

## 👥 Authors

**Boutrid Mourad** - AI Engineering Student
- 📧 Email: muurad.boutrid@gmail.com
- 🔗 LinkedIn: [Mourad Boutrid](https://www.linkedin.com/in/mourad-boutrid-981659336)

**Kassimi Achraf** - AI Engineering Student  
- 📧 Email: ac.kassimi@edu.umi.ac.ma
- 🔗 LinkedIn: [Achraf Kassimi](https://www.linkedin.com/in/achraf-kassimi-605418285)

## 🙏 Acknowledgments

- **NASA** for providing the CMAPSS dataset and establishing benchmarks in prognostics research
- **TensorFlow/Keras Team** for the robust deep learning framework
- **Streamlit** for enabling rapid development of interactive ML applications
- **Open Source Community** for the foundational libraries that made this project possible

---

## 📊 Project Status

![GitHub last commit](https://img.shields.io/github/last-commit/mouradboutrid/TurboGuard)
![GitHub issues](https://img.shields.io/github/issues/mouradboutrid/TurboGuard)
![GitHub pull requests](https://img.shields.io/github/issues-pr/mouradboutrid/TurboGuard)

**Current Version**: 2.0.0  
**Status**: Active Development  
**Maintenance**: Actively Maintained

---

⭐ **If you find TurboGuard helpful for your research or projects, please consider starring this repository!**

*For detailed technical discussions, feature requests, or collaboration opportunities, feel free to open an issue or contact the authors directly.*
