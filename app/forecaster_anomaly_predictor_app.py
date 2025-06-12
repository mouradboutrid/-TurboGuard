import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import streamlit as st

class AnomalyPredictorApp:
    """Enhanced class for anomaly detection using saved CMAPSS forecasting model"""

    def __init__(self):
        self.model_path = MODEL_PATHS['forecasting']['model']
        self.config_path = MODEL_PATHS['forecasting']['config']
        self.model = None
        self.config = None
        self.selected_features = None
        self.sequence_length = None
        self.op_mode_scalers = {}

    def check_model_files(self):
        """Check if model files exist"""
        model_exists = os.path.exists(self.model_path)
        config_exists = os.path.exists(self.config_path)
        return model_exists and config_exists

    def load_model_and_config(self):
        try:
            if not self.check_model_files():
                st.error(f"❌ Model files not found at specified paths:\n- Model: {self.model_path}\n- Config: {self.config_path}")
                return False

            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            self.selected_features = self.config['selected_features']
            self.sequence_length = self.config['sequence_length']

            try:
                custom_objects = {
                    'mse': tf.keras.metrics.MeanSquaredError(),
                    'mae': tf.keras.metrics.MeanAbsoluteError(),
                    'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                    'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
                }
                self.model = load_model(self.model_path, custom_objects=custom_objects)
                st.success("✅ LSTM Forecasting Model loaded successfully!")

            except Exception as e1:
                try:
                    self.model = load_model(self.model_path, compile=False)
                    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
                    st.success("✅ LSTM Forecasting Model loaded successfully (recompiled)!")
                except Exception as e2:
                    st.error(f"Failed to load LSTM Forecasting Model: {e2}")
                    return False

            return True
        except Exception as e:
            st.error(f"Error loading LSTM Forecasting Model/config: {e}")
            return False

    def preprocess_test_data(self, test_filepath):
        sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        operational_columns = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        columns = ['unit_id', 'time_cycle'] + operational_columns + sensor_columns

        df = pd.read_csv(test_filepath, sep=' ', header=None)
        df = df.dropna(axis=1, how='all')
        df = df.iloc[:, :len(columns)]
        df.columns = columns

        op_data = df[operational_columns].values
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['op_mode'] = kmeans.fit_predict(op_data)

        normalized_dfs = []
        for mode in df['op_mode'].unique():
            mode_data = df[df['op_mode'] == mode].copy()
            scaler = MinMaxScaler()
            mode_data[sensor_columns] = scaler.fit_transform(mode_data[sensor_columns])
            self.op_mode_scalers[mode] = scaler
            normalized_dfs.append(mode_data)

        df = pd.concat(normalized_dfs)

        sensors_to_remove = []
        for sensor in sensor_columns:
            if sensor in df.columns and df[sensor].var() < 1e-6:
                sensors_to_remove.append(sensor)
        if sensors_to_remove:
            df = df.drop(columns=sensors_to_remove)

        return df

    def create_sequences(self, df):
        feature_data = df[self.selected_features].values
        modes = df['op_mode'].values
        unit_ids = df['unit_id'].values

        sequences = []
        mode_sequences = []
        unit_sequence_ids = []
        sequence_indices = []

        for unit_id in df['unit_id'].unique():
            unit_mask = df['unit_id'] == unit_id
            unit_data = feature_data[unit_mask]
            unit_modes = modes[unit_mask]

            if len(unit_data) > self.sequence_length:
                for i in range(len(unit_data) - self.sequence_length):
                    sequences.append(unit_data[i:i + self.sequence_length])
                    mode_sequences.append(unit_modes[i + self.sequence_length])
                    unit_sequence_ids.append(unit_id)
                    sequence_indices.append(i + self.sequence_length)

        return (np.array(sequences), np.array(mode_sequences),
                np.array(unit_sequence_ids), np.array(sequence_indices))

    def detect_anomalies(self, X_test, modes_test, threshold_percentile=95):
        try:
            predictions = self.model.predict([X_test, modes_test], verbose=0)
        except:
            try:
                predictions = self.model.predict(X_test, verbose=0)
            except Exception as e:
                st.error(f"LSTM Forecasting Model prediction failed: {e}")
                return None

        results = {}
        methods = ['mse', 'mae', 'max']

        for method in methods:
            if method == 'mse':
                errors = np.mean((X_test[:, -1, :] - predictions) ** 2, axis=1)
            elif method == 'mae':
                errors = np.mean(np.abs(X_test[:, -1, :] - predictions), axis=1)
            elif method == 'max':
                errors = np.max(np.abs(X_test[:, -1, :] - predictions), axis=1)

            threshold = np.percentile(errors, threshold_percentile)

            results[method] = {
                'errors': errors,
                'threshold': threshold,
                'anomalies': errors > threshold
            }

        ensemble_anomalies = (
            results['mse']['anomalies'].astype(int) +
            results['mae']['anomalies'].astype(int) +
            results['max']['anomalies'].astype(int)
        ) >= 2

        results['ensemble'] = {
            'anomalies': ensemble_anomalies,
            'threshold': None
        }

        return results

    def analyze_unit_anomalies(self, anomaly_results, unit_ids, sequence_indices):
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

    def predict_and_analyze(self, test_filepath, threshold_percentile=95):
        test_df = self.preprocess_test_data(test_filepath)
        X_test, modes_test, unit_ids, sequence_indices = self.create_sequences(test_df)

        anomaly_results = self.detect_anomalies(X_test, modes_test, threshold_percentile)
        if anomaly_results is None:
            return None

        unit_analysis = self.analyze_unit_anomalies(anomaly_results, unit_ids, sequence_indices)

        return {
            'anomaly_results': anomaly_results,
            'unit_analysis': unit_analysis,
            'test_data': test_df,
            'sequences': X_test,
            'unit_ids': unit_ids,
            'sequence_indices': sequence_indices,
            'model_type': 'LSTM Forecasting'
        }
