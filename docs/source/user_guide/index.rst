# User Guide

Welcome to the comprehensive User Guide for TurboGuard, your advanced deep learning toolkit for predictive maintenance and anomaly detection in turbofan engines. This guide provides detailed insights into leveraging TurboGuard's dual LSTM architecture for industrial-grade engine health monitoring and predictive analytics.

## Table of Contents

### Core Workflows
- [Data Preprocessing Pipeline](data_preprocessing)
- [Model Training & Optimization](model_training) 
- [Anomaly Detection Strategies](anomaly_detection)
- [Visualization & Monitoring](visualization)

### Advanced Topics
- [Multi-Modal Analysis](advanced/multi_modal)
- [Production Deployment](advanced/deployment)
- [Custom Model Development](advanced/custom_models)
- [Performance Tuning](advanced/optimization)

## Overview

TurboGuard empowers engineers and data scientists with a comprehensive suite of tools for turbofan engine health management:

### 🔧 **Industrial-Grade Data Processing**
Advanced preprocessing pipeline designed specifically for CMAPSS and similar aerospace datasets, featuring automated data quality assessment, sensor drift correction, and multi-condition normalization.

### 🧠 **Dual LSTM Architecture**
State-of-the-art neural network implementation combining:
- **Reconstruction-based anomaly detection** through LSTM AutoEncoders
- **Predictive forecasting** with sequence-to-sequence LSTM models
- **Ensemble decision making** for enhanced reliability

### 🚨 **Intelligent Anomaly Detection**
Multi-layered detection framework incorporating statistical analysis, deep learning inference, and domain-specific heuristics optimized for turbofan engine failure patterns.

### 📊 **Production-Ready Visualization**
Interactive dashboards and analytical tools designed for both technical analysis and executive reporting, featuring real-time monitoring capabilities and customizable alert systems.

## Prerequisites

### System Requirements
- **Python**: 3.9+ (recommended: 3.10+)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: 5GB+ free space for datasets and model artifacts
- **GPU**: Optional but recommended (CUDA-compatible for accelerated training)

### Dependencies
Ensure all dependencies are installed as specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dataset Access
Download the NASA CMAPSS dataset and place it in the designated data directory. The framework supports all four CMAPSS subsets (FD001-FD004) with automatic format detection.

## Core Components Deep Dive

### 🔄 **Data Preprocessing Pipeline**
Transform raw sensor readings into model-ready sequences through:

- **Multi-Sensor Data Integration**: Harmonize 21 sensor channels with different sampling rates and units
- **Temporal Sequence Construction**: Generate sliding windows optimized for LSTM input requirements  
- **Advanced Normalization**: Apply min-max, z-score, or robust scaling with outlier detection
- **Feature Engineering**: Extract domain-specific features like trend indicators and cycle-based statistics
- **Data Quality Assurance**: Automated detection and handling of missing values, sensor malfunctions, and data inconsistencies

*Learn more: [Data Preprocessing Guide](data_preprocessing)*

### 🎯 **Model Training & Optimization**
Configure and train sophisticated neural architectures:

- **LSTM AutoEncoder Training**: Unsupervised learning for normal operation pattern recognition
- **Forecasting Model Development**: Supervised training for RUL prediction and trend analysis
- **Hyperparameter Optimization**: Automated tuning using Bayesian optimization and cross-validation
- **Model Validation**: Comprehensive evaluation using industry-standard metrics and statistical tests
- **Transfer Learning**: Adapt pre-trained models across different operational conditions and fault modes

*Learn more: [Model Training Guide](model_training)*

### 🔍 **Multi-Strategy Anomaly Detection**
Deploy comprehensive detection algorithms:

- **Reconstruction Error Analysis**: Identify deviations from learned normal patterns using autoencoder outputs
- **Forecasting Residual Monitoring**: Detect anomalies through prediction error analysis and trend deviation
- **Statistical Threshold Management**: Dynamic threshold adjustment based on operational context and historical performance
- **Ensemble Scoring**: Combine multiple detection methods for improved accuracy and reduced false positives
- **Temporal Pattern Analysis**: Identify anomalous sequences and progressive degradation patterns

*Learn more: [Anomaly Detection Guide](anomaly_detection)*

### 📈 **Advanced Visualization & Monitoring**
Generate actionable insights through sophisticated visualization:

- **Real-Time Dashboards**: Live engine health monitoring with customizable KPI displays
- **Anomaly Heatmaps**: Temporal and sensor-wise anomaly visualization for root cause analysis
- **Predictive Analytics Charts**: RUL forecasting, degradation trending, and maintenance scheduling
- **Comparative Analysis Tools**: Multi-engine and multi-condition performance comparison
- **Executive Reporting**: Automated generation of maintenance reports and health summaries

*Learn more: [Visualization Guide](visualization)*

## Getting Started Workflow

### Step 1: Environment Setup
```bash
# Verify Python version
python --version  # Should be 3.9+

# Install TurboGuard
pip install -r requirements.txt

# Verify installation
python -c "import src.LSTM_AutoEncoder; print('Installation successful')"
```

### Step 2: Data Preparation
```python
from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader

# Initialize data loader
loader = CMAPSSDataLoader()

# Load and preprocess dataset
data = loader.load_dataset('FD001', normalize=True, sequence_length=50)
```

### Step 3: Model Training
```python
from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoEncoder

# Configure model
model = LSTMAutoEncoder(
    encoding_dim=50,
    dropout_rate=0.2,
    learning_rate=0.001
)

# Train model
model.fit(data['train'], validation_data=data['val'], epochs=100)
```

### Step 4: Anomaly Detection
```python
# Perform anomaly detection
anomalies = model.detect_anomalies(
    data['test'], 
    threshold_percentile=95,
    min_anomaly_duration=3
)

# Generate report
model.generate_anomaly_report(anomalies, output_path='results/')
```

## Best Practices

### Data Quality Management
- Always perform exploratory data analysis before preprocessing
- Validate sensor readings against known operational limits
- Monitor for concept drift in long-term deployments

### Model Development
- Start with baseline models before implementing complex architectures
- Use cross-validation for robust performance estimation
- Regularly retrain models with new operational data

### Production Deployment
- Implement proper logging and monitoring for deployed models
- Set up automated model performance tracking
- Establish clear escalation procedures for critical anomalies

## Troubleshooting & Support

### Common Issues
- **Memory errors**: Reduce batch size or sequence length
- **Convergence problems**: Adjust learning rate or model architecture
- **Performance degradation**: Check for data distribution shifts

### Getting Help
- **Documentation**: Comprehensive API reference and tutorials
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions and share experiences

### Contact
For enterprise support, custom development, or consulting services, contact the development team through the GitHub repository or LinkedIn profiles listed in the main documentation.

---
