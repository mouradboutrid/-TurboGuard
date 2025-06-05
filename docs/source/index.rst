# TurboGuard Documentation

![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**TurboGuard** is an advanced deep learning framework engineered for predictive maintenance and real-time anomaly detection in turbofan engines. Built upon a sophisticated dual LSTM architecture, it leverages the comprehensive NASA CMAPSS dataset to deliver industrial-grade engine health monitoring solutions that predict failures before they occur.

## System Architecture

```mermaid
graph TD
    A[📊 DataLoader] --> B[🔄 DataPreprocessor]
    B --> C[🧠 LSTMAutoEncoder]
    C --> D[🚨 AnomalyDetector]
    D --> E[📈 Visualizer]
    E --> F[🔍 CMAPSSAnomalyAnalyzer]
    F --> G[💾 ModelManager]

    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#fff3e0
    style F fill:#e8f5e8
```

## Overview

TurboGuard revolutionizes turbofan engine health monitoring through an innovative dual-LSTM approach that combines two complementary detection methodologies:

- **LSTM AutoEncoder**: Employs unsupervised reconstruction learning to identify anomalous patterns by detecting deviations from normal operational sequences
- **Forecasting LSTM**: Utilizes supervised time-series prediction to anticipate future engine behavior, enabling early detection of degradation trends and accurate Remaining Useful Life (RUL) estimation

This synergistic architecture enables proactive maintenance strategies, substantially reduces unplanned downtime, and optimizes operational efficiency across turbofan engine fleets.

## Key Features

✨ **Hybrid Detection Architecture**  
Combines reconstruction-based anomaly detection with predictive forecasting for enhanced reliability and reduced false positives.

🎯 **Interactive Real-Time Dashboard**  
Streamlit-powered interface providing live engine health visualization, anomaly alerts, and maintenance scheduling insights.

📊 **Comprehensive Multivariate Analysis**  
Processes all 21 CMAPSS sensor channels simultaneously, capturing complex interdependencies and temporal patterns across engine subsystems.

🔧 **Production-Ready Modularity**  
Engineered with pluggable components and scalable architecture suitable for both research environments and industrial deployment.

⚡ **Intelligent Preprocessing Pipeline**  
Features advanced normalization techniques, adaptive sequence windowing, automated feature selection, and noise filtering.

🚨 **Multi-Modal Detection Strategies**  
Integrates LSTM reconstruction error analysis, forecasting deviation metrics, statistical threshold monitoring, and ensemble decision making.

📈 **Comprehensive Performance Analytics**  
Provides detailed evaluation through MSE, MAE, RMSE, F1-score, precision-recall curves, ROC analysis, and anomaly lead-time assessment.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mouradboutrid/TurboGuard.git
cd TurboGuard

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app/app.py
```

```python
# Minimal implementation example
from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder

# Initialize data pipeline
loader = CMAPSSDataLoader()
data = loader.load_dataset('FD001')

# Configure and train model
model = LSTMAutoEncoder()
model.build_model(input_shape=(50, 21))
model.train(data)

# Perform anomaly detection
anomalies = model.detect_anomalies(test_data)
```

## Performance Benchmarks

### AutoEncoder Model Performance
- **Reconstruction Precision**: MSE < 0.15 on validation datasets
- **Anomaly Detection F1-Score**: > 0.52 with balanced precision-recall
- **False Positive Rate**: < 20% during nominal operating conditions
- **Training Convergence**: Stable within 100 epochs

### Forecasting Model Performance
- **RUL Prediction Accuracy**: RMSE < 15 cycles across all fault modes
- **Early Warning Capability**: > 60% of critical anomalies detected ≥20 cycles before failure
- **Long-Horizon Forecasting**: Maintains prediction accuracy up to 50 time steps
- **Multi-Condition Robustness**: Consistent performance across varying operational environments

## Dataset Overview

**NASA CMAPSS Dataset** (Commercial Modular Aero-Propulsion System Simulation)

| Subset | Fault Modes | Operating Conditions | Training Units | Test Units |
|--------|-------------|---------------------|----------------|------------|
| FD001  | 1           | 1                   | 100            | 100        |
| FD002  | 1           | 6                   | 260            | 259        |
| FD003  | 2           | 1                   | 100            | 100        |
| FD004  | 2           | 6                   | 248            | 249        |

**Sensor Array**: 21-channel comprehensive monitoring including fan speed (N1), core speed (N2), exhaust gas temperature (EGT), fuel flow rate, compressor discharge temperature, turbine inlet temperature, and various pressure measurements across engine stages.

## Documentation Structure

### Getting Started
- [Installation Guide](tutorials/installation)
- [Quick Start Tutorial](tutorials/quickstart)
- [First Model Training](tutorials/first_model)

### User Guide
- [Data Preprocessing](user_guide/data_preprocessing)
- [Model Training & Optimization](user_guide/model_training)
- [Anomaly Detection Strategies](user_guide/anomaly_detection)
- [Visualization & Monitoring](user_guide/visualization)

### Examples
- [Basic Usage Patterns](examples/basic_usage)
- [Advanced Implementation](examples/advanced_usage)
- [Production Deployment](examples/production)

### API Reference
- [AutoEncoder API](api/autoencoder)
- [Forecasting API](api/forecasting)
- [Utilities & Helpers](api/utilities)

### Development
- [Contributing Guidelines](development/contributing)
- [System Architecture](development/architecture)
- [Testing Framework](development/testing)

## Authors

**Boutrid Mourad**  
*AI Engineering Student*  
📧 muurad.boutrid@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mourad-boutrid-981659336)

**Kassimi Achraf**  
*AI Engineering Student*  
📧 ac.kassimi@edu.umi.ac.ma  
🔗 [LinkedIn](https://www.linkedin.com/in/achraf-kassimi-605418285)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
