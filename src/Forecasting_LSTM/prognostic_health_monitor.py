import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from src.Forecasting_LSTM.prognostic_visualization_suite import PrognosticVisualizationSuite
from src.Forecasting_LSTM.prognostic_LSTMModel import PrognosticLSTMModel
from src.Forecasting_LSTM.anomaly_detection_engine import AnomalyDetectionEngine
from src.Forecasting_LSTM.prognostic_feature_selector import PrognosticFeatureSelector
from src.Forecasting_LSTM.forecasting_data_processor import DataProcessor

class PrognosticHealthMonitor:
    """Heal monitor with operational mode support"""

    def __init__(self, sequence_length=30, n_features=10, save_base_path=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.save_base_path = save_base_path
        self.selected_features = None

        # Initialize components
        self.data_processor = DataProcessor()
        self.feature_selector = PrognosticFeatureSelector()
        self.model = None
        self.anomaly_detector = None

        # Set up save paths
        if save_base_path:
            self.model_save_path = os.path.join(save_base_path, 'saved_models')
            self.plots_save_path = os.path.join(save_base_path, 'plots')
            self.config_save_path = os.path.join(save_base_path, 'config')
            os.makedirs(self.model_save_path, exist_ok=True)
            os.makedirs(self.plots_save_path, exist_ok=True)
            os.makedirs(self.config_save_path, exist_ok=True)
            self.visualizer = PrognosticVisualizationSuite(self.plots_save_path)
        else:
            self.visualizer = PrognosticVisualizationSuite()

    def prepare_sequence_data(self, df):
        """Enhanced sequence preparation with operational modes"""
        feature_data = df[self.selected_features].values
        modes = df['op_mode'].values if 'op_mode' in df.columns else None

        sequences = []
        targets = []
        mode_sequences = [] if modes is not None else None

        for unit_id in df['unit_id'].unique():
            unit_mask = df['unit_id'] == unit_id
            unit_data = feature_data[unit_mask]
            unit_modes = modes[unit_mask] if modes is not None else None

            if len(unit_data) > self.sequence_length:
                if unit_modes is not None:
                    seq, tgt, mds = self._create_sequences(unit_data, unit_modes)
                    sequences.extend(seq)
                    targets.extend(tgt)
                    mode_sequences.extend(mds)
                else:
                    seq, tgt = self._create_sequences(unit_data)
                    sequences.extend(seq)
                    targets.extend(tgt)

        if mode_sequences is not None:
            return np.array(sequences), np.array(targets), np.array(mode_sequences)
        return np.array(sequences), np.array(targets)

    def _create_sequences(self, data, modes=None):
        """Enhanced sequence creation with optional modes"""
        sequences = []
        targets = []
        mode_seqs = [] if modes is not None else None

        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(data[i + self.sequence_length])
            if modes is not None:
                mode_seqs.append(modes[i + self.sequence_length])

        if modes is not None:
            return np.array(sequences), np.array(targets), np.array(mode_seqs)
        return np.array(sequences), np.array(targets)

    def run_complete_analysis(self, data_filepaths, use_combined=True, test_size=0.2, epochs=50):
        """Enhanced analysis with operational mode support"""
        print("Starting complete Data prognostic analysis...")

        # Load data
        self.individual_datasets, self.combined_df = self.data_processor.load_multiple_datasets(data_filepaths)

        if self.combined_df is None or len(self.combined_df) == 0:
            raise ValueError("No data loaded successfully")

        analysis_df = self.combined_df if use_combined else max(self.individual_datasets.values(), key=len)

        # Show dataset overview
        self.visualizer.plot_dataset_overview(self.individual_datasets, self.combined_df)

        # Feature selection
        self.selected_features = self.feature_selector.calculate_prognostic_relevance(
            analysis_df, top_k=self.n_features
        )

        # Prepare sequences
        if 'op_mode' in analysis_df.columns:
            X, y, modes = self.prepare_sequence_data(analysis_df)
            X_train, X_test, y_train, y_test, modes_train, modes_test = train_test_split(
                X, y, modes, test_size=test_size, random_state=42
            )
        else:
            X, y = self.prepare_sequence_data(analysis_df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            modes_train = modes_test = None

        print(f"Prepared {len(X_train)} training sequences and {len(X_test)} test sequences")

        # Build and train model
        self.model = PrognosticLSTMModel(
            n_features=len(self.selected_features),
            sequence_length=self.sequence_length
        )
        self.model.build_model()

        print("Training model...")
        history = self.model.train(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            modes_train=modes_train,
            modes_val=modes_test,
            epochs=epochs
        )

        # Show training progress
        self.visualizer.plot_training_progress(history)

        # Anomaly detection
        print("Performing anomaly detection...")
        self.anomaly_detector = AnomalyDetectionEngine(self.model.model)

        if modes_test is not None:
            anomaly_results = self.anomaly_detector.detect_anomalies(X_test, modes=modes_test)
        else:
            anomaly_results = self.anomaly_detector.detect_anomalies(X_test)

        # Show anomaly results
        self.visualizer.plot_anomaly_results(anomaly_results, modes_test)

        # Calculate and display performance metrics
        self._calculate_performance_metrics(X_test, y_test, modes_test)

        # Save model and configuration
        if self.save_base_path:
            self._save_analysis_results(analysis_df, anomaly_results)

        print("Complete analysis finished successfully!")
        return {
            'model': self.model,
            'anomaly_detector': self.anomaly_detector,
            'anomaly_results': anomaly_results,
            'selected_features': self.selected_features,
            'datasets': self.individual_datasets,
            'combined_data': self.combined_df
        }

    def _calculate_performance_metrics(self, X_test, y_test, modes_test=None):
        """Calculate and display model performance metrics"""
        print("\nCalculating performance metrics...")

        # Get predictions
        if modes_test is not None:
            y_pred = self.model.model.predict([X_test, modes_test], verbose=0)
        else:
            y_pred = self.model.model.predict(X_test, verbose=0)

        # Calculate metrics for each feature
        feature_metrics = {}
        for i, feature in enumerate(self.selected_features):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])

            feature_metrics[feature] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            }

        # Display results
        print("\nFeature Reconstruction Performance:")
        print("-" * 50)
        for feature, metrics in feature_metrics.items():
            print(f"{feature:>12}: MSE={metrics['MSE']:.6f}, MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.4f}")

        # Overall metrics
        overall_mse = np.mean([m['MSE'] for m in feature_metrics.values()])
        overall_mae = np.mean([m['MAE'] for m in feature_metrics.values()])
        overall_r2 = np.mean([m['R2'] for m in feature_metrics.values()])

        print("-" * 50)
        print(f"{'Overall':>12}: MSE={overall_mse:.6f}, MAE={overall_mae:.6f}, R²={overall_r2:.4f}")

    def _save_analysis_results(self, analysis_df, anomaly_results):
        """Save analysis results and configuration"""
        print("Saving analysis results...")

        # Save model
        model_path = os.path.join(self.model_save_path, f'lstm_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        self.model.save_model(model_path)

        # Save configuration
        config = {
            'selected_features': self.selected_features,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'feature_scores': self.feature_selector.feature_scores,
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(analysis_df),
                'total_units': analysis_df['unit_id'].nunique(),
                'operational_modes': analysis_df['op_mode'].nunique() if 'op_mode' in analysis_df.columns else 1
            }
        }

        config_path = os.path.join(self.config_save_path, f'analysis_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save anomaly results summary
        anomaly_summary = {
            'total_samples': len(anomaly_results['ensemble']['anomalies']),
            'anomalies_detected': int(anomaly_results['ensemble']['anomalies'].sum()),
            'anomaly_rate': float(anomaly_results['ensemble']['anomalies'].mean() * 100),
            'method_comparison': {
                'mse': int(anomaly_results['mse']['anomalies'].sum()),
                'mae': int(anomaly_results['mae']['anomalies'].sum()),
                'max': int(anomaly_results['max']['anomalies'].sum())
            }
        }

        anomaly_path = os.path.join(self.config_save_path, f'anomaly_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(anomaly_path, 'w') as f:
            json.dump(anomaly_summary, f, indent=2)

        print(f"Results saved to: {self.save_base_path}")

    def load_trained_model(self, model_path, config_path):
        """Load a previously trained model and configuration"""
        print(f"Loading model from: {model_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.selected_features = config['selected_features']
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']

        # Initialize and load model
        self.model = PrognosticLSTMModel(
            n_features=self.n_features,
            sequence_length=self.sequence_length
        )
        self.model.load_model(model_path)

        print("Model and configuration loaded successfully!")
        return config

    def predict_anomalies(self, new_data, threshold_percentile=95):
        """Predict anomalies on new data using trained model"""
        if self.model is None:
            raise ValueError("No trained model available. Train a model first or load a pre-trained one.")

        # Prepare sequences from new data
        if 'op_mode' in new_data.columns:
            X_new, _, modes_new = self.prepare_sequence_data(new_data)
        else:
            X_new, _ = self.prepare_sequence_data(new_data)
            modes_new = None

        # Initialize anomaly detector if not exists
        if self.anomaly_detector is None:
            self.anomaly_detector = AnomalyDetectionEngine(self.model.model)

        # Detect anomalies
        if modes_new is not None:
            results = self.anomaly_detector.detect_anomalies(
                X_new,
                threshold_percentile=threshold_percentile,
                modes=modes_new
            )
        else:
            results = self.anomaly_detector.detect_anomalies(
                X_new,
                threshold_percentile=threshold_percentile
            )

        return results
