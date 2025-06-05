import numpy as np
import matplotlib.pyplot as plt

from src.LSTM_AutoEncoder.data_loader import DataLoader
from src.LSTM_AutoEncoder.data_preprocessor import DataPreprocessor
from src.DataPreprocessor.lstm_autoencoder import LSTMAutoencoder
from src.LSTMAutoencoder.anomaly_detector import AnomalyDetector
from src.LSTMAutoencoder.model_manager import ModelManager
from src.LSTMAutoencoder.visualizer import Visualizer


class AnomalyAnalyzer:
    """Main class that orchestrates the entire anomaly detection pipeline"""

    def __init__(self, data_dir='./', models_dir='./models'):
        self.data_loader = DataLoader(data_dir)
        self.preprocessor = DataPreprocessor()
        self.autoencoder = None
        self.anomaly_detector = AnomalyDetector()
        self.model_manager = ModelManager(models_dir)
        self.visualizer = Visualizer()

        # Analysis results storage
        self.results = {}
        self.config = {}

    def analyze_dataset(self, dataset_id='FD004', sequence_length=30,
                       sensors_to_drop=None, epochs=30, save_model=True):
        """
        Complete analysis pipeline for a CMAPSS dataset

        Parameters:
        dataset_id (str): Dataset ID ('FD001', 'FD002', 'FD003', or 'FD004')
        sequence_length (int): Length of sequences for LSTM
        sensors_to_drop (list): List of sensor indices to drop
        epochs (int): Number of training epochs
        save_model (bool): Whether to save the trained model

        Returns:
        dict: Dictionary containing analysis results
        """
        print(f"Processing {dataset_id} dataset...")

        # Set default sensors to drop if not provided
        if sensors_to_drop is None:
            sensors_to_drop = [1, 5, 10, 16, 18, 19]

        # Store configuration
        self.config = {
            'dataset_id': dataset_id,
            'sequence_length': sequence_length,
            'sensors_to_drop': sensors_to_drop,
            'epochs': epochs
        }

        # Load data
        data = self.data_loader.load_dataset(dataset_id)

        # Preprocess data
        train_data = self.preprocessor.preprocess_data(
            data['train'],
            calculate_rul=True,
            normalize=True,
            drop_sensors=sensors_to_drop
        )

        # Split by engine
        engine_data = self.preprocessor.split_by_engine(train_data)

        # Create sequences for engines
        print("Creating sequences...")
        engine_sequences = {}
        for engine_id in list(engine_data.keys())[:5]:  # Demo with 5 engines
            sequences = self.preprocessor.create_sequences(
                engine_data[engine_id], sequence_length
            )
            engine_sequences[engine_id] = sequences

        # Prepare training data
        train_engines = list(engine_sequences.keys())[:3]
        X_train = np.concatenate([engine_sequences[engine_id] for engine_id in train_engines])

        # Train autoencoder
        print("Training LSTM Autoencoder...")
        self.autoencoder = LSTMAutoencoder(encoding_dim=16)
        history = self.autoencoder.train(X_train, epochs=epochs, batch_size=32)

        # Save model if requested
        if save_model:
            model_dir = self.model_manager.save_model_package(
                dataset_id, self.autoencoder, self.preprocessor, self.config
            )

        # Analyze test engines
        print("Detecting anomalies...")
        results = {}
        test_engines = list(engine_sequences.keys())[3:]

        for engine_id in test_engines:
            X_test = engine_sequences[engine_id]

            # Ensemble anomaly detection
            engine_results = self.anomaly_detector.ensemble_detection(
                X_test, self.autoencoder, threshold_multiplier=3.0
            )

            results[engine_id] = engine_results

            # Visualize results
            fig = self.visualizer.plot_anomalies(
                engine_data[engine_id],
                engine_results['lstm']['scores'],
                engine_results['lstm']['flags'],
                engine_results['lstm']['threshold'],
                sequence_length=sequence_length,
                title=f"LSTM Anomaly Detection - Engine {engine_id} ({dataset_id})"
            )
            plt.show()

        # Store results
        self.results = {
            'data': data,
            'processed_data': train_data,
            'engine_data': engine_data,
            'engine_sequences': engine_sequences,
            'autoencoder': self.autoencoder,
            'history': history,
            'anomaly_results': results,
            'config': self.config
        }

        print("Analysis completed!")
        return self.results

    def load_saved_model(self, dataset_id):
        """Load a previously saved model"""
        self.autoencoder, self.preprocessor, self.config = self.model_manager.load_model_package(dataset_id)
        return self.autoencoder, self.preprocessor, self.config

    def predict_anomalies(self, data, dataset_id=None):
        """
        Predict anomalies on new data using loaded model

        Parameters:
        data (DataFrame): New data to analyze
        dataset_id (str): Dataset ID for loading model (if not already loaded)

        Returns:
        dict: Anomaly detection results
        """
        if self.autoencoder is None:
            if dataset_id is None:
                raise ValueError("No model loaded and no dataset_id provided")
            self.load_saved_model(dataset_id)

        # Preprocess new data
        processed_data = self.preprocessor.preprocess_data(
            data,
            calculate_rul=self.preprocessor.preprocessing_params.get('calculate_rul', True),
            normalize=self.preprocessor.preprocessing_params.get('normalize', True),
            drop_sensors=self.preprocessor.preprocessing_params.get('drop_sensors', None)
        )

        # Split by engine
        engine_data = self.preprocessor.split_by_engine(processed_data)

        # Create sequences and detect anomalies
        results = {}
        sequence_length = self.config.get('sequence_length', 30)

        for engine_id, engine_df in engine_data.items():
            # Create sequences
            X_sequences = self.preprocessor.create_sequences(engine_df, sequence_length)

            # Detect anomalies
            engine_results = self.anomaly_detector.ensemble_detection(
                X_sequences, self.autoencoder, threshold_multiplier=3.0
            )

            results[engine_id] = engine_results

        return results

    def compare_all_datasets(self):
        """
        Compare anomaly detection performance across all CMAPSS datasets

        Returns:
        dict: Performance metrics for each dataset
        """
        results = {}

        for dataset_id in ['FD001', 'FD002', 'FD003', 'FD004']:
            print(f"\n===== Analyzing {dataset_id} =====")
            analysis_results = self.analyze_dataset(dataset_id=dataset_id)
            results[dataset_id] = analysis_results

            # Calculate metrics
            metrics = {}
            for engine_id, engine_results in analysis_results['anomaly_results'].items():
                lstm_anomalies = np.sum(engine_results['lstm']['flags'])
                stat_anomalies = np.sum(engine_results['statistical']['flags'])
                wav_anomalies = np.sum(engine_results['wavelet']['flags'])
                ensemble_anomalies = np.sum(engine_results['ensemble']['flags'])
                total_points = len(engine_results['lstm']['flags'])

                metrics[engine_id] = {
                    'lstm_anomaly_rate': lstm_anomalies / total_points,
                    'statistical_anomaly_rate': stat_anomalies / total_points,
                    'wavelet_anomaly_rate': wav_anomalies / total_points,
                    'ensemble_anomaly_rate': ensemble_anomalies / total_points,
                    'total_points': total_points
                }

            results[dataset_id]['metrics'] = metrics

        # Print performance comparison
        self._print_performance_comparison(results)

        return results

    def _print_performance_comparison(self, results):
        """Print performance comparison across datasets"""
        print("\n===== Performance Comparison =====")
        print("Detailed per-engine anomaly detection performance:")

        for dataset_id in ['FD001', 'FD002', 'FD003', 'FD004']:
            if dataset_id not in results:
                continue

            metrics = results[dataset_id]['metrics']
            print(f"\n{dataset_id}:")

            # Calculate averages per anomaly type
            lstm_rates = []
            statistical_rates = []
            wavelet_rates = []
            ensemble_rates = []

            for engine_id, engine_metrics in sorted(metrics.items()):
                print(f"  Engine {engine_id}:")
                print(f"    LSTM anomaly rate:       {engine_metrics['lstm_anomaly_rate']:.4f}")
                print(f"    Statistical anomaly rate: {engine_metrics['statistical_anomaly_rate']:.4f}")
                print(f"    Wavelet anomaly rate:     {engine_metrics['wavelet_anomaly_rate']:.4f}")
                print(f"    Ensemble anomaly rate:    {engine_metrics['ensemble_anomaly_rate']:.4f}")

                lstm_rates.append(engine_metrics['lstm_anomaly_rate'])
                statistical_rates.append(engine_metrics['statistical_anomaly_rate'])
                wavelet_rates.append(engine_metrics['wavelet_anomaly_rate'])
                ensemble_rates.append(engine_metrics['ensemble_anomaly_rate'])

            # Print average anomaly rates per dataset
            if lstm_rates:  # Check for the data
                print(f"\n  Average anomaly rates for {dataset_id}:")
                print(f"    LSTM anomaly rate:       {sum(lstm_rates) / len(lstm_rates):.4f}")
                print(f"    Statistical anomaly rate: {sum(statistical_rates) / len(statistical_rates):.4f}")
                print(f"    Wavelet anomaly rate:     {sum(wavelet_rates) / len(wavelet_rates):.4f}")
                print(f"    Ensemble anomaly rate:    {sum(ensemble_rates) / len(ensemble_rates):.4f}")

    def get_model_summary(self):
        """Get summary of the current model"""
        if self.autoencoder is None:
            return "No model loaded"

        summary = {
            'input_shape': self.autoencoder.input_shape,
            'encoding_dim': self.autoencoder.encoding_dim,
            'config': self.config,
            'preprocessing_params': self.preprocessor.preprocessing_params
        }

        return summary

    def list_available_models(self):
        """List all available saved models"""
        return self.model_manager.list_available_models()
