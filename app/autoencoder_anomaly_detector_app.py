import os
import numpy as np
import json
import pywt
from tensorflow.keras.models import load_model
from scipy.stats import zscore


MODEL_PATHS = {
    'forecasting': {
        'model': "trained_models/forecaster_model/saved_models/lstm_model_20250529_005930.h5",
        'config': "trained_models/forecaster_model/config/analysis_config_20250529_005930.json"
    },
    'autoencoder': {
        'FD001': {
            'autoencoder': "trained_models/trained_models/autoencoder_models/FD001/autoencoder.keras",
            'encoder': "trained_models/trained_models/autoencoder_models/FD001/encoder.keras",
            'config': "trained_models/trained_models/autoencoder_models/FD001/config.json"
        },
        'FD002': {
            'autoencoder': "trained_models/trained_models/autoencoder_models/FD002/autoencoder.keras",
            'encoder': "trained_models/trained_models/autoencoder_models/FD002/encoder.keras",
            'config': "trained_models/trained_models/autoencoder_models/FD002/config.json"
        },
        'FD003': {
            'autoencoder': "trained_models/trained_models/autoencoder_models/FD003/autoencoder.keras",
            'encoder': "trained_models/trained_models/autoencoder_models/FD003/encoder.keras",
            'config': "trained_models/trained_models/autoencoder_models/FD003/config.json"
        },
        'FD004': {
            'autoencoder': "trained_models/trained_models/autoencoder_models/FD004/autoencoder.keras",
            'encoder': "trained_models/trained_models/autoencoder_models/FD004/encoder.keras",
            'config': "trained_models/trained_models/autoencoder_models/FD004/config.json"
        }
    }
}

class AutoencoderAnomalyDetector:
    """Pre-trained anomaly detector using autoencoder models"""

    def __init__(self, dataset_name='FD001'):
        self.dataset_name = dataset_name
        self.autoencoder = None
        self.encoder = None
        self.config = None
        self.thresholds = {}

        if dataset_name in MODEL_PATHS['autoencoder']:
            self.model_paths = MODEL_PATHS['autoencoder'][dataset_name]
        else:
            raise ValueError(f"Dataset {dataset_name} not found in model paths")

    def check_model_files(self):
        """Check if all model files exist"""
        files_exist = []
        for key, path in self.model_paths.items():
            exists = os.path.exists(path)
            files_exist.append(exists)
            if not exists:
                st.warning(f"⚠️ {key.capitalize()} file not found: {path}")

        return all(files_exist)

    def load_models(self):
        """Load pre-trained autoencoder models and config"""
        try:
            if not self.check_model_files():
                st.error(f"❌ Some model files are missing for dataset {self.dataset_name}")
                return False

            with open(self.model_paths['config'], 'r') as f:
                self.config = json.load(f)

            self.autoencoder = load_model(self.model_paths['autoencoder'])
            self.encoder = load_model(self.model_paths['encoder'])
            self.thresholds['reconstruction'] = self.config.get('threshold', 0.1)

            if hasattr(self.autoencoder, 'input_shape'):
                expected_shape = self.autoencoder.input_shape

            st.success(f"✅ Autoencoder models loaded successfully for {self.dataset_name}!")
            return True

        except Exception as e:
            st.error(f"Error loading autoencoder models: {e}")
            return False

    def compute_reconstruction_error(self, X):
        """Compute reconstruction errors"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not loaded")

        try:
            X_pred = self.autoencoder.predict(X, verbose=0)
            return np.mean(np.square(X - X_pred), axis=(1, 2))
        except Exception as e:
            pass
            raise

    def detect_autoencoder_anomalies(self, X, threshold_percentile=95):
        """Detect anomalies using autoencoder reconstruction error"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not loaded")

        try:
            errors = self.compute_reconstruction_error(X)

            if 'threshold' in self.config:
                threshold = self.config['threshold']
            else:
                threshold = np.percentile(errors, threshold_percentile)

            anomalies = errors > threshold

            return {
                'anomalies': anomalies,
                'scores': errors,
                'threshold': threshold,
                'method': 'autoencoder'
            }
        except Exception as e:
            st.error(f"Autoencoder anomaly detection failed: {e}")
            return None

    def detect_statistical_anomalies(self, X, method='zscore', threshold=3):
        """Detect anomalies using statistical methods"""
        try:
            X_flat = X.reshape(X.shape[0], -1)

            if method == 'zscore':
                z_scores = np.abs(zscore(X_flat, axis=0))
                anomaly_scores = np.max(z_scores, axis=1)
                anomalies = anomaly_scores > threshold
            elif method == 'iqr':
                Q1 = np.percentile(X_flat, 25, axis=0)
                Q3 = np.percentile(X_flat, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (X_flat < lower_bound) | (X_flat > upper_bound)
                anomaly_scores = np.sum(outliers, axis=1) / X_flat.shape[1]
                anomalies = anomaly_scores > 0.1
            else:
                raise ValueError(f"Unknown statistical method: {method}")

            return {
                'anomalies': anomalies,
                'scores': anomaly_scores,
                'threshold': threshold,
                'method': method
            }
        except Exception as e:
            st.error(f"Statistical anomaly detection failed: {e}")
            return None

    def detect_wavelet_anomalies(self, X, wavelet='db4', threshold_factor=3):
        """Detect anomalies using wavelet analysis"""
        try:
            anomaly_scores = []

            for i in range(X.shape[0]):
                sequence = X[i]
                sensor_scores = []
                for j in range(sequence.shape[1]):
                    sensor_data = sequence[:, j]
                    coeffs = pywt.wavedec(sensor_data, wavelet, level=3)
                    detail_coeffs = coeffs[1:]
                    detail_energy = sum(np.sum(np.square(c)) for c in detail_coeffs)
                    sensor_scores.append(detail_energy)

                anomaly_scores.append(np.mean(sensor_scores))

            anomaly_scores = np.array(anomaly_scores)
            threshold = np.mean(anomaly_scores) + threshold_factor * np.std(anomaly_scores)
            anomalies = anomaly_scores > threshold

            return {
                'anomalies': anomalies,
                'scores': anomaly_scores,
                'threshold': threshold,
                'method': 'wavelet'
            }
        except Exception as e:
            st.error(f"Wavelet anomaly detection failed: {e}")
            return None

    def ensemble_detection(self, X, methods=['autoencoder', 'statistical', 'wavelet'], vote_threshold=2, threshold_percentile=95):
        """Ensemble anomaly detection with error handling"""
        results = {}
        votes = np.zeros(X.shape[0])
        successful_methods = 0

        for method in methods:
            try:
                if method == 'autoencoder':
                    result = self.detect_autoencoder_anomalies(X, threshold_percentile)
                elif method == 'statistical':
                    result = self.detect_statistical_anomalies(X)
                elif method == 'wavelet':
                    result = self.detect_wavelet_anomalies(X)
                else:
                    continue

                if result is not None:
                    results[method] = result
                    votes += result['anomalies'].astype(int)
                    successful_methods += 1
                else:
                    st.warning(f"⚠️ {method.title()} detection failed, skipping...")

            except Exception as e:
                st.error(f"Error in {method} detection: {e}")
                continue

        if successful_methods > 0:
            adjusted_threshold = min(vote_threshold, successful_methods)
            ensemble_anomalies = votes >= adjusted_threshold

            results['ensemble'] = {
                'anomalies': ensemble_anomalies,
                'votes': votes,
                'vote_threshold': adjusted_threshold,
                'successful_methods': successful_methods
            }
        else:
            st.error("❌ All detection methods failed!")
            results['ensemble'] = {
                'anomalies': np.zeros(X.shape[0], dtype=bool),
                'votes': votes,
                'vote_threshold': vote_threshold,
                'successful_methods': 0
            }

        return results
