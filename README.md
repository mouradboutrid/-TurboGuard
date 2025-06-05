# TimeSeries AutoEncoder Anomaly Detection Pipeline

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## 📋 Project Overview
This project implements a comprehensive anomaly detection system for turbofan engine data using two complementary deep learning approaches:

LSTM AutoEncoder — Reconstruction-based anomaly detection

Forecasting LSTM — Prediction-based anomaly detection

Designed specifically for the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset, this system enables aircraft engine health monitoring, anomaly detection, and early failure prediction.

The CMAPSS dataset consists of multi-sensor time series data collected from multiple turbofan engines operating under various conditions. Each engine’s operational cycle is represented by sensor readings capturing physical parameters such as temperature, pressure, fan speed, and vibration levels.

Number of Sensors: 21 sensor measurements per time step

Sensor Types: Include but are not limited to:

Total temperature at fan inlet

Total pressure at fan inlet

Fan speed

Core speed

Turbine inlet temperature

Fuel flow

Static pressure

Vibration measurements

Data Characteristics:

Variable-length sequences representing engine life cycles

Multiple operating conditions and fault modes

Sensor readings sampled at regular intervals

By leveraging these detailed sensor readings, the system captures complex temporal patterns and dependencies to accurately detect anomalies indicating potential engine degradation or failure.

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│         DataLoader → DataPreprocessor → Feature Selection       │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                     DUAL MODEL APPROACH                         │
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
│  AnomalyAnalyzer → ModelManager → Streamlit App          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Workflow

### Phase 1: Data Ingestion and Preprocessing

* **DataLoader**: Loads and structures the dataset.
* **DataPreprocessor**: Cleans and normalizes data; generates sequences.

### Phase 2: Feature Engineering and Selection

* **PrognosticFeatureSelector**: Identifies relevant sensors.
* **CMAPSSDataProcessor**: Removes constants, handles outliers and missing values.

### Phase 3: Model Training and Prediction

#### AutoEncoder Branch

* **LSTMAutoencoder**: Encoder-decoder model for reconstruction.
* **AnomalyDetector**: Detects anomalies via reconstruction error, Z-score, and wavelet transform.

#### Forecasting Branch

* **PrognosticLSTMModel**: Forecasts future values.
* **AnomalyDetectionEngine**: Detects anomalies based on forecast deviation.

### Phase 4: Analysis and Monitoring

* **PrognosticHealthMonitor**: Full health evaluation.
* **AnomalyAnalyzer**: Model testing, performance comparison.

### Phase 5: Visualization and Reporting

* **Visualizer**: Plots training and detection results.
* **PrognosticVisualizationSuite**: Forecast and anomaly insights.

### Phase 6: Model Management and Deployment

* **ModelManager**: Save/load models and configuration.
* **Streamlit App (`app.py`)**: Web UI for users to run the system interactively.

---

## 🎯 Key Features

* **Dual-Strategy Detection**: Combines reconstruction and forecasting.
* **Ensemble Analysis**: Multiple detection techniques for robustness.
* **Advanced Visuals**: Charts and dashboards for insight.
* **Modular Design**: Easy extension, testability, and scaling.
* **Web Interface**: Streamlit app for non-technical usage.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Model Testing 

```python
# AutoEncoder
python src/autoencoder_anomaly_predictor_test.py.py

# Forecaster
python forecaster_anomaly_predictor_test.py
```

### 3. Run Anomaly Detection

```python
python detect_anomalies.py
```

### 4. Launch Web App

```bash
streamlit run app.py
```

---

## 📈 Performance Metrics

### Model Evaluation

* **Precision, Recall, F1-Score**
* **ROC-AUC**
* **MAE / RMSE** for errors

### System Evaluation

* **Latency / Throughput**
* **Memory Use & Scalability**
* **Detection Reliability**

---

## 📊 Data Flow Diagram

```text
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

## ⚙️ Configuration

### Training

* LSTM Units: 50–200
* Sequence Length: 30–50
* Dropout: 0.2–0.5
* Batch Size: 32–128
* Learning Rate: 0.001–0.01

### Thresholds

* Z-score > 3.0
* Dynamic percentile for reconstruction
* Ensemble voting supported

---

## 📚 Dependencies

* `TensorFlow`, `Keras`
* `NumPy`, `Pandas`, `scikit-learn`
* `matplotlib`, `seaborn`, `plotly`
* `PyWavelets`, `joblib`, `Streamlit`

---

## 🧭 Future Enhancements

* Transformer & attention-based architectures
* Real-time and online detection
* Graph neural networks for sensor connectivity
* Cloud deployment and auto-retraining

---

## 📁 Repository Structure

```text
.
├── data/                # CMAPSS dataset files
├── models/              # Trained model artifacts
├── scripts/             # Training & detection scripts
├── app.py               # Streamlit app entry point
├── README.md            # Project overview
├── requirements.txt     # Python dependencies
└── utils/               # Preprocessing and feature engineering helpers
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙋 Contributing

Contributions are welcome! Please open an issue or submit a pull request with suggestions, improvements, or bug fixes.

---

## 📞 Contact

For questions or collaboration opportunities, please reach out via GitHub Issues or email.

