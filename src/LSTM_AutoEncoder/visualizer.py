import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """Class for visualization methods"""

    @staticmethod
    def plot_anomalies(data, anomaly_scores, anomaly_flags, threshold,
                      sequence_length=30, title="Anomaly Detection Results"):
        """
        Visualize anomaly detection results

        Parameters:
        data (DataFrame): Original data with time information
        anomaly_scores (numpy.ndarray): Anomaly scores
        anomaly_flags (numpy.ndarray): Boolean flags for anomalies
        threshold (float): Threshold for anomaly detection
        sequence_length (int): Length of each sequence
        title (str): Plot title

        Returns:
        matplotlib.figure.Figure: Figure with anomaly visualization
        """
        # Create figure and axes
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot reconstruction error or anomaly score
        axes[0].plot(anomaly_scores, label='Anomaly Score')
        axes[0].axhline(y=threshold, color='r', linestyle='--',
                        label=f'Threshold: {threshold:.4f}')
        axes[0].set_ylabel('Anomaly Score')
        axes[0].set_title(title)
        axes[0].legend()

        # Highlight anomalies
        anomaly_idx = np.where(anomaly_flags)[0]
        axes[0].scatter(anomaly_idx, anomaly_scores[anomaly_idx],
                       color='red', label='Anomalies', s=50, alpha=0.7)

        # Plot original sensor data with anomalies highlighted
        sensor_cols = [col for col in data.columns if 'sensor_' in col][:3]

        for i, sensor in enumerate(sensor_cols):
            sensor_data = data[sensor].values
            axes[1].plot(sensor_data, label=sensor, alpha=0.7)

        # Highlight anomalies on sensor data
        anomaly_regions = np.zeros(len(data))
        for idx in anomaly_idx:
            if idx + sequence_length < len(anomaly_regions):
                anomaly_regions[idx + sequence_length - 1] = 1

        # Find contiguous anomaly regions
        anomaly_starts = np.where(np.diff(np.pad(anomaly_regions, (1, 0), 'constant')) == 1)[0]
        anomaly_ends = np.where(np.diff(np.pad(anomaly_regions, (0, 1), 'constant')) == -1)[0]

        # Highlight anomaly regions
        for start, end in zip(anomaly_starts, anomaly_ends):
            axes[1].axvspan(start, end, color='red', alpha=0.2)

        axes[1].set_xlabel('Cycles')
        axes[1].set_ylabel('Sensor Values')
        axes[1].set_title('Sensor Data with Anomalies Highlighted')
        axes[1].legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_training_history(history):
        """Plot training history"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig