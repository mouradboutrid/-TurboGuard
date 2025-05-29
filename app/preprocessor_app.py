import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessorApp:
    """Preprocess data with model configuration matching"""

    def __init__(self, model_config=None):
        if model_config:
            self.sequence_length = model_config.get('sequence_length', 30)
            self.n_features = model_config.get('n_features', 19)
            self.sensors_to_drop = model_config.get('sensors_to_drop', [])
            self.selected_sensors = model_config.get('selected_sensors', [])
        else:
            self.sequence_length = 30
            self.n_features = 19
            self.sensors_to_drop = []
            self.selected_sensors = []

        self.scalers = {}

    def preprocess(self, df):
        """Preprocess the dataset to match model requirements"""
        sensor_columns = [col for col in df.columns if col.startswith('sensor_')]

        if not self.selected_sensors:
            for sensor in sensor_columns:
                if df[sensor].var() < 1e-6:
                    self.sensors_to_drop.append(sensor)

        df = df.drop(columns=self.sensors_to_drop, errors='ignore')
        sensor_columns = [col for col in df.columns if col.startswith('sensor_')]

        if self.selected_sensors:
            available_selected = [s for s in self.selected_sensors if s in sensor_columns]
            if available_selected:
                sensor_columns = available_selected
            else:
                st.warning("None of the selected sensors found in data. Using all available sensors.")

        if len(sensor_columns) > self.n_features:
            sensor_columns = sensor_columns[:self.n_features]
            st.info(f"Using first {self.n_features} sensors to match model requirements")
        elif len(sensor_columns) < self.n_features:
            st.warning(f"Data has {len(sensor_columns)} sensors but model expects {self.n_features}")
            for i in range(len(sensor_columns), self.n_features):
                df[f'sensor_pad_{i}'] = 0.0
                sensor_columns.append(f'sensor_pad_{i}')

        scaler = StandardScaler()
        df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
        self.scalers['sensors'] = scaler

        return df, sensor_columns

    def create_sequences(self, df, sensor_columns):
        """Create sequences for LSTM with correct dimensions"""
        sequences = []
        unit_ids = []
        sequence_indices = []

        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].sort_values('time_cycle')
            unit_sensors = unit_data[sensor_columns].values

            if len(unit_sensors) >= self.sequence_length:
                for i in range(len(unit_sensors) - self.sequence_length + 1):
                    sequences.append(unit_sensors[i:i + self.sequence_length])
                    unit_ids.append(unit_id)
                    sequence_indices.append(i + self.sequence_length)

        sequences = np.array(sequences)

        if sequences.ndim == 3:
            expected_shape = (sequences.shape[0], self.sequence_length, self.n_features)
            if sequences.shape != expected_shape:
                st.info(f"Reshaping sequences from {sequences.shape} to match model input {expected_shape}")
                if sequences.shape[2] > self.n_features:
                    sequences = sequences[:, :, :self.n_features]
                elif sequences.shape[2] < self.n_features:
                    padding = np.zeros((sequences.shape[0], sequences.shape[1],
                                     self.n_features - sequences.shape[2]))
                    sequences = np.concatenate([sequences, padding], axis=2)

        return sequences, np.array(unit_ids), np.array(sequence_indices)