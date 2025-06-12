import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from autoencoder_anomaly_detector_app import AutoencoderAnomalyDetector
from preprocessor_app import DataPreprocessorApp
from data_loader import DataLoader

class AutoencoderAnomalyAnalyzer:
    """Main analyzer class for Model 2 with pre-trained models"""

    def __init__(self, dataset_name='FD001'):
        self.dataset_name = dataset_name
        self.detector = AutoencoderAnomalyDetector(dataset_name)
        self.preprocessor = None
        self.data = None
        self.sequences = None
        self.unit_ids = None
        self.sequence_indices = None
        self.models_loaded = False

    def load_models(self):
        """Load pre-trained models"""
        self.models_loaded = self.detector.load_models()
        if self.models_loaded and self.detector.config:
            self.preprocessor = DataPreprocessorApp(self.detector.config)
        return self.models_loaded

    def load_and_preprocess(self, data_path=None):
        """Load and preprocess dataset"""
        if not self.preprocessor:
            st.error("Models must be loaded first to get preprocessing configuration")
            return False

        self.data = DataLoader.load_dataset(self.dataset_name, data_path)
        if self.data is not None:
            self.data, sensor_columns = self.preprocessor.preprocess(self.data)
            self.sequences, self.unit_ids, self.sequence_indices = self.preprocessor.create_sequences(self.data, sensor_columns)

            return True
        return False

    def detect_anomalies(self, methods=['autoencoder', 'statistical', 'wavelet'], threshold_percentile=95):
        """Detect anomalies using specified methods"""
        if self.sequences is None:
            raise ValueError("No data loaded.")

        if not self.models_loaded:
            raise ValueError("Models not loaded.")

        return self.detector.ensemble_detection(self.sequences, methods, threshold_percentile=threshold_percentile)

    def analyze_unit_anomalies(self, anomaly_results, unit_ids, sequence_indices):
        """Analyze anomalies by unit for autoencoder model"""
        if 'ensemble' not in anomaly_results:
            return {}

        ensemble_anomalies = anomaly_results['ensemble']['anomalies']
        unit_analysis = {}

        for i, (unit_id, seq_idx) in enumerate(zip(unit_ids, sequence_indices)):
            if unit_id not in unit_analysis:
                unit_analysis[unit_id] = {
                    'total_sequences': 0,
                    'anomalies': 0,
                    'anomaly_rate': 0.0,
                    'risk_level': 'Low',
                    'anomaly_positions': []  # Fixed: Initialize anomaly_positions list
                }

            unit_analysis[unit_id]['total_sequences'] += 1
            if ensemble_anomalies[i]:
                unit_analysis[unit_id]['anomalies'] += 1
                unit_analysis[unit_id]['anomaly_positions'].append(seq_idx)

        for unit_id in unit_analysis:
            total = unit_analysis[unit_id]['total_sequences']
            anomalies = unit_analysis[unit_id]['anomalies']
            rate = (anomalies / total) * 100 if total > 0 else 0
            unit_analysis[unit_id]['anomaly_rate'] = rate

            # Enhanced risk classification
            if rate >= 75:
                unit_analysis[unit_id]['risk_level'] = 'Critical'
            elif rate >= 50:
                unit_analysis[unit_id]['risk_level'] = 'High'
            elif rate >= 25:
                unit_analysis[unit_id]['risk_level'] = 'Medium'
            else:
                unit_analysis[unit_id]['risk_level'] = 'Low'

        return unit_analysis

    def predict_and_analyze(self, data_path, methods=['autoencoder', 'statistical'], threshold_percentile=95):
        """Complete analysis pipeline for autoencoder model"""
        if not self.load_and_preprocess(data_path):
            return None

        anomaly_results = self.detect_anomalies(methods, threshold_percentile)
        if not anomaly_results:
            return None

        unit_analysis = self.analyze_unit_anomalies(anomaly_results, self.unit_ids, self.sequence_indices)

        return {
            'anomaly_results': anomaly_results,
            'unit_analysis': unit_analysis,
            'test_data': self.data,
            'sequences': self.sequences,
            'unit_ids': self.unit_ids,
            'sequence_indices': self.sequence_indices,
            'model_type': 'Autoencoder Ensemble'
        }
