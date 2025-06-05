import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class AnomalyPredictorTest:
    """Focused class for anomaly detection using saved model (forecasting)"""

    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.selected_features = None
        self.sequence_length = None
        self.op_mode_scalers = {}

    def load_model_and_config(self):
        print(f"Loading model from: {self.model_path}")
        print(f"Loading config from: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self.selected_features = self.config['selected_features']
        self.sequence_length = self.config['sequence_length']

        # Different approaches to load the model
        try:
            # Load with custom objects
            custom_objects = {
                'mse': tf.keras.metrics.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
                'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
            }
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            print("Model loaded successfully with custom objects!")

        except Exception as e1:
            print(f"First attempt failed: {e1}")
            try:
                # Load without compiling
                self.model = load_model(self.model_path, compile=False)
                print("Model loaded successfully without compilation!")

                # Recompile the model with proper metrics
                self.model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae', 'mse']
                )
                print("Model recompiled successfully!")

            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                try:
                    # Load with TensorFlow's SavedModel format 
                    import tensorflow.keras.utils as utils
                    self.model = tf.keras.models.load_model(
                        self.model_path,
                        custom_objects={'mse': 'mean_squared_error', 'mae': 'mean_absolute_error'}
                    )
                    print("Model loaded with TensorFlow SavedModel format!")

                except Exception as e3:
                    print(f"All loading attempts failed:")
                    print(f"Error 1: {e1}")
                    print(f"Error 2: {e2}")
                    print(f"Error 3: {e3}")
                    raise Exception("Could not load the model with any method")

        print("Model and configuration loaded successfully!")
        print(f"Selected features: {self.selected_features}")
        print(f"Sequence length: {self.sequence_length}")

        # Print model summary to verify it loaded correctly
        try:
            print("\nModel architecture:")
            self.model.summary()
        except:
            print("Could not display model summary, but model loaded successfully")

    def preprocess_test_data(self, test_filepath):
        print(f"Loading test data from: {test_filepath}")

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
            print(f"Removed constant sensors: {sensors_to_remove}")

        print(f"Test data loaded: {len(df)} samples, {df['unit_id'].nunique()} units")
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
        print("Detecting anomalies...")

        # Handle prediction based on model input requirements
        try:
            # Try with both inputs (sequence data and modes)
            predictions = self.model.predict([X_test, modes_test], verbose=0)
        except Exception as e:
            print(f"Multi-input prediction failed: {e}")
            try:
                # Try with just sequence data
                predictions = self.model.predict(X_test, verbose=0)
            except Exception as e2:
                print(f"Single input prediction also failed: {e2}")
                raise Exception("Could not make predictions with the model")

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
                    'anomaly_positions': [],
                    'anomaly_rate': 0.0
                }

            unit_analysis[unit_id]['total_sequences'] += 1
            if ensemble_anomalies[i]:
                unit_analysis[unit_id]['anomalies'] += 1
                unit_analysis[unit_id]['anomaly_positions'].append(seq_idx)

        for unit_id in unit_analysis:
            total = unit_analysis[unit_id]['total_sequences']
            anomalies = unit_analysis[unit_id]['anomalies']
            unit_analysis[unit_id]['anomaly_rate'] = (anomalies / total) * 100 if total > 0 else 0

        return unit_analysis

    def visualize_anomaly_results(self, anomaly_results, unit_analysis, modes_test):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        methods = ['MSE', 'MAE', 'Max Error', 'Ensemble']
        counts = [
            anomaly_results['mse']['anomalies'].sum(),
            anomaly_results['mae']['anomalies'].sum(),
            anomaly_results['max']['anomalies'].sum(),
            anomaly_results['ensemble']['anomalies'].sum()
        ]

        bars = axes[0, 0].bar(methods, counts, color=['blue', 'green', 'orange', 'red'])
        axes[0, 0].set_title('Anomalies Detected by Method')
        axes[0, 0].set_ylabel('Number of Anomalies')

        for bar, count in zip(bars, counts):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           str(count), ha='center', va='bottom')

        for method, color in zip(['mse', 'mae', 'max'], ['blue', 'green', 'orange']):
            axes[0, 1].hist(anomaly_results[method]['errors'], bins=50, alpha=0.5,
                           label=method.upper(), color=color)
            axes[0, 1].axvline(anomaly_results[method]['threshold'], color=color, linestyle='--')
        axes[0, 1].set_title('Error Distributions with Thresholds')
        axes[0, 1].set_xlabel('Error Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        unit_rates = [(uid, data['anomaly_rate']) for uid, data in unit_analysis.items()]
        unit_rates.sort(key=lambda x: x[1], reverse=True)
        top_units = unit_rates[:20]

        if top_units:
            unit_ids, rates = zip(*top_units)
            axes[1, 0].bar(range(len(unit_ids)), rates, color='red', alpha=0.7)
            axes[1, 0].set_title('Top 20 Units by Anomaly Rate')
            axes[1, 0].set_xlabel('Unit ID')
            axes[1, 0].set_ylabel('Anomaly Rate (%)')
            axes[1, 0].set_xticks(range(len(unit_ids)))
            axes[1, 0].set_xticklabels([f'Unit {uid}' for uid in unit_ids], rotation=45)

        mode_anomalies = {}
        for mode in np.unique(modes_test):
            mask = modes_test == mode
            mode_anomalies[f'Mode {mode}'] = anomaly_results['ensemble']['anomalies'][mask].sum()

        axes[1, 1].bar(mode_anomalies.keys(), mode_anomalies.values(),
                      color=['blue', 'green', 'orange'])
        axes[1, 1].set_title('Anomalies by Operational Mode')
        axes[1, 1].set_ylabel('Number of Anomalies')

        plt.tight_layout()
        plt.show()

    def plot_sensor_anomalies(self, test_df, anomaly_results, unit_ids, sequence_indices, top_n_units=5):
        """Plot sensor values over time with anomaly locations highlighted"""

        ensemble_anomalies = anomaly_results['ensemble']['anomalies']

        # Get top anomalous units for detailed plotting
        unit_anomaly_counts = {}
        for i, unit_id in enumerate(unit_ids):
            if ensemble_anomalies[i]:
                unit_anomaly_counts[unit_id] = unit_anomaly_counts.get(unit_id, 0) + 1

        top_anomalous_units = sorted(unit_anomaly_counts.items(),
                                   key=lambda x: x[1], reverse=True)[:top_n_units]

        if not top_anomalous_units:
            print("No anomalies found to plot")
            return

        print(f"Plotting sensor data for top {len(top_anomalous_units)} most anomalous units...")

        # Create subplots for each unit
        fig, axes = plt.subplots(len(top_anomalous_units), 1,
                                figsize=(20, 4 * len(top_anomalous_units)))
        if len(top_anomalous_units) == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.selected_features)))

        for idx, (unit_id, anomaly_count) in enumerate(top_anomalous_units):
            # Get unit data
            unit_data = test_df[test_df['unit_id'] == unit_id].copy()
            unit_data = unit_data.sort_values('time_cycle')

            # Get anomaly positions for this unit
            unit_anomaly_positions = []
            for i, (seq_unit_id, seq_idx) in enumerate(zip(unit_ids, sequence_indices)):
                if seq_unit_id == unit_id and ensemble_anomalies[i]:
                    unit_anomaly_positions.append(seq_idx)

            ax = axes[idx]

            # Plot each sensor
            for j, sensor in enumerate(self.selected_features):
                if sensor in unit_data.columns:
                    ax.plot(unit_data['time_cycle'], unit_data[sensor],
                           color=colors[j], alpha=0.7, linewidth=1, label=sensor)

            # Highlight anomaly positions
            for pos in unit_anomaly_positions:
                ax.axvline(x=pos, color='red', linestyle='--', alpha=0.8, linewidth=2)

            # Add red markers at anomaly points
            if unit_anomaly_positions:
                for sensor in self.selected_features:
                    if sensor in unit_data.columns:
                        anomaly_values = []
                        anomaly_cycles = []
                        for pos in unit_anomaly_positions:
                            if pos in unit_data['time_cycle'].values:
                                val = unit_data[unit_data['time_cycle'] == pos][sensor].iloc[0]
                                anomaly_values.append(val)
                                anomaly_cycles.append(pos)
                        if anomaly_cycles:
                            ax.scatter(anomaly_cycles, anomaly_values,
                                     color='red', s=50, alpha=0.8, zorder=5)

            ax.set_title(f'Unit {unit_id} - Sensor Values Over Time ({anomaly_count} anomalies)')
            ax.set_xlabel('Time Cycle')
            ax.set_ylabel('Normalized Sensor Value')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

        # Additional plot: Sensor heatmap showing anomaly intensity
        self.plot_sensor_anomaly_heatmap(test_df, anomaly_results, unit_ids, sequence_indices)

    def plot_sensor_anomaly_heatmap(self, test_df, anomaly_results, unit_ids, sequence_indices):
        """Create a heatmap showing which sensors are most anomalous"""

        ensemble_anomalies = anomaly_results['ensemble']['anomalies']

        # Calculate anomaly rates per sensor
        sensor_anomaly_data = {sensor: [] for sensor in self.selected_features}

        for i, (unit_id, seq_idx) in enumerate(zip(unit_ids, sequence_indices)):
            if ensemble_anomalies[i]:
                # Get sensor values at anomaly point
                unit_data = test_df[(test_df['unit_id'] == unit_id) &
                                  (test_df['time_cycle'] == seq_idx)]
                if not unit_data.empty:
                    for sensor in self.selected_features:
                        if sensor in unit_data.columns:
                            sensor_anomaly_data[sensor].append(unit_data[sensor].iloc[0])

        # Create box plot of anomalous sensor values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Box plot of sensor values during anomalies
        anomaly_data = [sensor_anomaly_data[sensor] for sensor in self.selected_features
                       if sensor_anomaly_data[sensor]]
        sensor_labels = [sensor for sensor in self.selected_features
                        if sensor_anomaly_data[sensor]]

        if anomaly_data:
            bp = ax1.boxplot(anomaly_data, labels=sensor_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)

        ax1.set_title('Sensor Value Distributions During Anomalies')
        ax1.set_xlabel('Sensors')
        ax1.set_ylabel('Normalized Sensor Value')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Count anomalies per sensor (how often each sensor contributes to anomalies)
        sensor_anomaly_counts = {sensor: len(values) for sensor, values in sensor_anomaly_data.items()}

        sensors = list(sensor_anomaly_counts.keys())
        counts = list(sensor_anomaly_counts.values())

        bars = ax2.bar(sensors, counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Anomaly Frequency by Sensor')
        ax2.set_xlabel('Sensors')
        ax2.set_ylabel('Number of Anomalous Readings')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_anomaly_timeline(self, test_df, anomaly_results, unit_ids, sequence_indices, top_n_units=3):
        """Plot timeline view showing when anomalies occur for top units"""

        ensemble_anomalies = anomaly_results['ensemble']['anomalies']

        # Get units with most anomalies
        unit_anomaly_counts = {}
        for i, unit_id in enumerate(unit_ids):
            if ensemble_anomalies[i]:
                unit_anomaly_counts[unit_id] = unit_anomaly_counts.get(unit_id, 0) + 1

        top_units = sorted(unit_anomaly_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_units]

        if not top_units:
            print("No anomalies found for timeline plot")
            return

        fig, axes = plt.subplots(len(top_units), 1, figsize=(16, 3 * len(top_units)))
        if len(top_units) == 1:
            axes = [axes]

        for idx, (unit_id, anomaly_count) in enumerate(top_units):
            unit_data = test_df[test_df['unit_id'] == unit_id].copy()
            unit_data = unit_data.sort_values('time_cycle')

            # Create binary anomaly timeline
            anomaly_timeline = np.zeros(len(unit_data))
            unit_cycles = unit_data['time_cycle'].values

            # Mark anomalous time points
            for i, (seq_unit_id, seq_idx) in enumerate(zip(unit_ids, sequence_indices)):
                if seq_unit_id == unit_id and ensemble_anomalies[i]:
                    cycle_idx = np.where(unit_cycles == seq_idx)[0]
                    if len(cycle_idx) > 0:
                        anomaly_timeline[cycle_idx[0]] = 1

            ax = axes[idx]

            # Plot timeline as scatter
            normal_mask = anomaly_timeline == 0
            anomaly_mask = anomaly_timeline == 1

            ax.scatter(unit_cycles[normal_mask], np.zeros(np.sum(normal_mask)),
                      c='green', alpha=0.6, s=20, label='Normal')
            ax.scatter(unit_cycles[anomaly_mask], np.zeros(np.sum(anomaly_mask)),
                      c='red', alpha=0.8, s=60, marker='X', label='Anomaly')

            ax.set_title(f'Unit {unit_id} - Anomaly Timeline ({anomaly_count} total anomalies)')
            ax.set_xlabel('Time Cycle')
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add failure progression indication
            max_cycle = unit_cycles.max()
            ax.axvline(x=max_cycle, color='black', linestyle=':', alpha=0.5,
                      label=f'End of Life (Cycle {max_cycle})')

        plt.tight_layout()
        plt.show()

    def print_anomaly_summary(self, anomaly_results, unit_analysis):
        total_sequences = len(anomaly_results['ensemble']['anomalies'])
        total_anomalies = anomaly_results['ensemble']['anomalies'].sum()
        anomaly_rate = (total_anomalies / total_sequences) * 100

        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Total sequences analyzed: {total_sequences:,}")
        print(f"Total anomalies detected: {total_anomalies:,}")
        print(f"Overall anomaly rate: {anomaly_rate:.2f}%")

        print(f"\nDetection by method:")
        for method in ['mse', 'mae', 'max']:
            count = anomaly_results[method]['anomalies'].sum()
            rate = (count / total_sequences) * 100
            print(f"  {method.upper()}: {count:,} ({rate:.2f}%)")

        units_with_anomalies = sum(1 for data in unit_analysis.values() if data['anomalies'] > 0)
        total_units = len(unit_analysis)

        print(f"\nUnit-level analysis:")
        print(f"  Total units: {total_units}")
        print(f"  Units with anomalies: {units_with_anomalies}")
        print(f"  Units without anomalies: {total_units - units_with_anomalies}")

        unit_rates = [(uid, data['anomaly_rate'], data['anomalies'])
                      for uid, data in unit_analysis.items() if data['anomalies'] > 0]
        unit_rates.sort(key=lambda x: x[1], reverse=True)

        if unit_rates:
            print(f"\nTop 10 most anomalous units:")
            for i, (unit_id, rate, count) in enumerate(unit_rates[:10], 1):
                print(f"  {i:2d}. Unit {unit_id:3d}: {rate:5.1f}% ({count:3d} anomalies)")

    def predict_and_analyze(self, test_filepath, threshold_percentile=95):
        test_df = self.preprocess_test_data(test_filepath)
        X_test, modes_test, unit_ids, sequence_indices = self.create_sequences(test_df)
        print(f"Created {len(X_test)} sequences for anomaly detection")

        anomaly_results = self.detect_anomalies(X_test, modes_test, threshold_percentile)
        unit_analysis = self.analyze_unit_anomalies(anomaly_results, unit_ids, sequence_indices)

        self.print_anomaly_summary(anomaly_results, unit_analysis)

        # Original visualization
        self.visualize_anomaly_results(anomaly_results, unit_analysis, modes_test)

        # New detailed sensor plots
        print("\nGenerating detailed sensor anomaly plots...")
        self.plot_sensor_anomalies(test_df, anomaly_results, unit_ids, sequence_indices, top_n_units=5)
        self.plot_anomaly_timeline(test_df, anomaly_results, unit_ids, sequence_indices, top_n_units=3)

        return {
            'anomaly_results': anomaly_results,
            'unit_analysis': unit_analysis,
            'test_data': test_df,
            'sequences': X_test,
            'unit_ids': unit_ids,
            'sequence_indices': sequence_indices
        }


def main():
    model_path = '/content/cmapss_analysis_results/saved_models/lstm_model_20250529_005930.h5'
    config_path = '/content/cmapss_analysis_results/config/analysis_config_20250529_005930.json'
    test_data_path = '/content/drive/MyDrive/CMAPSSData/test_FD004.txt'

    try:
        predictor = AnomalyPredictorTest(model_path, config_path)
        predictor.load_model_and_config()
        results = predictor.predict_and_analyze(
            test_filepath=test_data_path,
            threshold_percentile=95
        )

        print("\nAnomaly detection completed successfully!")
        print("Check the visualization plots above for detailed analysis.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check the file paths and ensure all files exist.")
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        import traceback
        traceback.print_exc()

main()
