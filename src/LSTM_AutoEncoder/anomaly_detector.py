import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf


class AnomalyDetector:
    """Class for various anomaly detection methods"""

    def __init__(self):
        self.models = {}

    def detect_lstm_anomalies(self, X_data, autoencoder, threshold_multiplier=3.0):
        """
        Detect anomalies using LSTM Autoencoder

        Parameters:
        X_data (numpy.ndarray): Data sequences to evaluate
        autoencoder (LSTMAutoencoder): Trained autoencoder model
        threshold_multiplier (float): Multiplier for anomaly threshold

        Returns:
        tuple: (anomaly_scores, anomaly_flags, threshold)
        """
        # Calculate reconstruction error
        reconstructions = autoencoder.predict(X_data)
        mse = np.mean(np.square(X_data - reconstructions), axis=(1, 2))

        # Calculate threshold (mean + std * multiplier)
        threshold = np.mean(mse) + threshold_multiplier * np.std(mse)

        # Flag anomalies
        anomaly_flags = mse > threshold

        return mse, anomaly_flags, threshold

    def detect_statistical_anomalies(self, X_data, window_size=10, threshold_multiplier=3.0):
        """
        Detect anomalies using statistical methods (Z-score) on time series data
        """
        # Reshape data for processing
        n_samples, seq_len, n_features = X_data.shape

        # Calculate mean across features for each time step
        mean_features = np.mean(X_data, axis=2)

        # Calculate moving average and standard deviation
        anomaly_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Get current window
            start_idx = max(0, i - window_size)
            window_data = mean_features[start_idx:i+1]

            if len(window_data) > 1:
                # Calculate Z-score
                mean_val = np.mean(window_data)
                std_val = np.std(window_data) + 1e-10
                z_score = np.abs((mean_features[i] - mean_val) / std_val)
                anomaly_scores[i] = np.mean(z_score)

        # Calculate threshold
        threshold = np.mean(anomaly_scores) + threshold_multiplier * np.std(anomaly_scores)

        # Flag anomalies
        anomaly_flags = anomaly_scores > threshold

        return anomaly_scores, anomaly_flags, threshold

    def detect_wavelet_anomalies(self, X_data, threshold_multiplier=3.0):
        """
        Detect anomalies using wavelet-based decomposition
        (Simplified version without pywavelets)
        """
        # Reshape data for easier processing
        n_samples, seq_len, n_features = X_data.shape

        # Simplified approach: use moving average and residuals
        anomaly_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Calculate trend using moving average (simplified wavelet approximation)
            sequence = X_data[i]
            trend = np.mean(sequence, axis=0)

            # Calculate residuals (simplified wavelet details)
            residuals = np.abs(sequence - trend)

            # Use residual energy as anomaly score
            anomaly_scores[i] = np.mean(np.sum(residuals**2, axis=1))

        # Calculate threshold
        threshold = np.mean(anomaly_scores) + threshold_multiplier * np.std(anomaly_scores)

        # Flag anomalies
        anomaly_flags = anomaly_scores > threshold

        return anomaly_scores, anomaly_flags, threshold

    def ensemble_detection(self, X_data, autoencoder, threshold_multiplier=3.0, voting_threshold=2):
        """
        Ensemble anomaly detection combining multiple methods

        Parameters:
        X_data (numpy.ndarray): Data sequences to evaluate
        autoencoder (LSTMAutoencoder): Trained autoencoder model
        threshold_multiplier (float): Multiplier for anomaly threshold
        voting_threshold (int): Minimum votes needed for anomaly detection

        Returns:
        dict: Dictionary containing results from all methods
        """
        # Get results from all methods
        lstm_scores, lstm_flags, lstm_threshold = self.detect_lstm_anomalies(
            X_data, autoencoder, threshold_multiplier
        )

        stat_scores, stat_flags, stat_threshold = self.detect_statistical_anomalies(
            X_data, threshold_multiplier=threshold_multiplier
        )

        wav_scores, wav_flags, wav_threshold = self.detect_wavelet_anomalies(
            X_data, threshold_multiplier=threshold_multiplier
        )

        # Ensemble voting
        combined_flags = lstm_flags.astype(int) + stat_flags.astype(int) + wav_flags.astype(int)
        ensemble_flags = combined_flags >= voting_threshold

        return {
            'lstm': {
                'scores': lstm_scores,
                'flags': lstm_flags,
                'threshold': lstm_threshold
            },
            'statistical': {
                'scores': stat_scores,
                'flags': stat_flags,
                'threshold': stat_threshold
            },
            'wavelet': {
                'scores': wav_scores,
                'flags': wav_flags,
                'threshold': wav_threshold
            },
            'ensemble': {
                'flags': ensemble_flags,
                'votes': combined_flags
            }
        }