import numpy as np


class AnomalyDetectionEngine:
    """From static to dynamic thresholding : our idea !!"""

    def __init__(self, model, threshold_percentile=95, smoothing_factor=0.1):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.smoothing_factor = smoothing_factor
        self.error_history = []

    def calculate_reconstruction_errors(self, X_test, method='mse', modes=None):
        """Calculate reconstruction errors with optional modes"""
        if modes is not None:
            predictions = self.model.predict([X_test, modes], verbose=0)
        else:
            predictions = self.model.predict(X_test, verbose=0)

        if method == 'mse':
            errors = np.mean((X_test[:, -1, :] - predictions) ** 2, axis=1)
        elif method == 'mae':
            errors = np.mean(np.abs(X_test[:, -1, :] - predictions), axis=1)
        elif method == 'max':
            errors = np.max(np.abs(X_test[:, -1, :] - predictions), axis=1)
        else:
            raise ValueError("Method must be 'mse', 'mae', or 'max'")

        # Update threshold dynamically
        self.update_threshold(errors)
        return errors

    def update_threshold(self, errors):
        """Dynamically adjust threshold using exponential smoothing"""
        if len(self.error_history) == 0:
            self.error_history = errors
        else:
            self.error_history = (self.smoothing_factor * errors +
                                (1 - self.smoothing_factor) * np.array(self.error_history))

    def detect_anomalies(self, X_test, threshold_percentile=None, modes=None):
        """Detect anomalies with optional mode support"""
        if threshold_percentile is not None:
            self.threshold_percentile = threshold_percentile

        results = {}
        methods = ['mse', 'mae', 'max']

        for method in methods:
            errors = self.calculate_reconstruction_errors(X_test, method, modes)
            threshold = np.percentile(errors, self.threshold_percentile)

            results[method] = {
                'errors': errors,
                'threshold': threshold,
                'anomalies': errors > threshold
            }

        # Ensemble approach
        ensemble_anomalies = (
            results['mse']['anomalies'].astype(int) +
            results['mae']['anomalies'].astype(int) +
            results['max']['anomalies'].astype(int)
        ) >= 2

        results['ensemble'] = {
            'anomalies': ensemble_anomalies,
            'threshold': None  # No single threshold for ensemble
        }

        return results