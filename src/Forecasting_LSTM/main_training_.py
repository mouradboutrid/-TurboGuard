import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt

# Example usage and demonstration
def main():
    """Example usage of the complete CMAPSS prognostic system"""

    # Define data file paths 
    data_filepaths = {
        'FD001': '/content/drive/MyDrive/CMAPSSData/train_FD001.txt',
        'FD002': '/content/drive/MyDrive/CMAPSSData/train_FD002.txt',
        'FD003': '/content/drive/MyDrive/CMAPSSData/train_FD003.txt',
        'FD004': '/content/drive/MyDrive/CMAPSSData/train_FD004.txt'
    }

    # Initialize the health monitor
    monitor = PrognosticHealthMonitor(
        sequence_length=30,
        n_features=10,
        save_base_path='./cmapss_analysis_results'
    )

    try:
        # Run complete analysis
        results = monitor.run_complete_analysis(
            data_filepaths=data_filepaths,
            use_combined=True,
            test_size=0.2,
            epochs=25
        )

        print("\nAnalysis Summary:")
        print(f"Selected Features: {len(results['selected_features'])}")
        print(f"Anomalies Detected: {results['anomaly_results']['ensemble']['anomalies'].sum()}")
        print(f"Total Test Samples: {len(results['anomaly_results']['ensemble']['anomalies'])}")

        # Example of using trained model for new predictions
        # new_data = pd.read_csv('new_sensor_data.csv')  # Your new data
        # new_anomalies = monitor.predict_anomalies(new_data)

    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please ensure C-MAPSS dataset files are in the correct location")
    except Exception as e:
        print(f"Error during analysis: {e}")


main()
