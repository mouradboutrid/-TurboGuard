import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Class for preprocessing data"""

    def __init__(self):
        self.scalers = {}
        self.preprocessing_params = {}

    def preprocess_data(self, data, calculate_rul=True, normalize=True, drop_sensors=None):
        """
        Preprocess the C-MAPSS data

        Parameters:
        data (DataFrame): Raw data from C-MAPSS dataset
        calculate_rul (bool): Whether to calculate RUL
        normalize (bool): Whether to normalize features
        drop_sensors (list): List of sensor indices to drop (if any)

        Returns:
        DataFrame: Preprocessed data
        """
        df = data.copy()

        # Store preprocessing parameters
        self.preprocessing_params = {
            'calculate_rul': calculate_rul,
            'normalize': normalize,
            'drop_sensors': drop_sensors
        }

        # Drop sensors that might not be useful
        if drop_sensors:
            sensors_to_drop = [f'sensor_{i}' for i in drop_sensors]
            df.drop(columns=sensors_to_drop, inplace=True, errors='ignore')

        # Calculate RUL (Remaining Useful Life)
        if calculate_rul:
            # Group by engine_id and get max cycle for each engine
            max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
            max_cycles.columns = ['engine_id', 'max_cycle']

            # Merge with original data
            df = df.merge(max_cycles, on='engine_id', how='left')

            # Calculate RUL
            df['RUL'] = df['max_cycle'] - df['cycle']
            df.drop('max_cycle', axis=1, inplace=True)

        # Normalize data if requested
        if normalize:
            df = self._normalize_data(df)

        return df

    def _normalize_data(self, df):
        """Internal method to normalize data"""
        # Identify non-feature columns that shouldn't be normalized
        non_feature_cols = ['engine_id', 'cycle']
        if 'RUL' in df.columns:
            non_feature_cols.append('RUL')

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in non_feature_cols]

        # Group by operating conditions for FD002 and FD004
        if 'op_setting_1' in df.columns and df['op_setting_1'].nunique() > 1:
            # If multiple operating conditions exist
            df['op_condition'] = df.apply(
                lambda x: f"{x['op_setting_1']}_{x['op_setting_2']}_{x['op_setting_3']}",
                axis=1
            )

            # Normalize within each operating condition
            normalized_dfs = []

            for condition, group in df.groupby('op_condition'):
                group_copy = group.copy()
                scaler = StandardScaler()
                group_copy[feature_cols] = scaler.fit_transform(group_copy[feature_cols])
                self.scalers[condition] = scaler
                normalized_dfs.append(group_copy)

            df = pd.concat(normalized_dfs, axis=0)

            # Drop the temporary condition column
            if 'op_condition' in df.columns:
                df.drop('op_condition', axis=1, inplace=True)
        else:
            # Simple normalization for single operating condition
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers['default'] = scaler

        return df

    def split_by_engine(self, data):
        """Split the data by engine_id"""
        engine_data = {}
        for engine_id, group in data.groupby('engine_id'):
            engine_data[engine_id] = group.sort_values('cycle').reset_index(drop=True)
        return engine_data

    def create_sequences(self, df, sequence_length=30, step=1, target_col=None):
        """
        Create sequences for multivariate time series modeling

        Parameters:
        df (DataFrame): Input data
        sequence_length (int): Length of each sequence
        step (int): Step size between sequences
        target_col (str): Target column name (if applicable)

        Returns:
        numpy arrays: X sequences and y targets (if target_col is provided)
        """
        # Remove non-feature columns
        feature_df = df.drop(['engine_id', 'cycle'], axis=1, errors='ignore')

        if target_col and target_col in feature_df.columns:
            # Split features and target
            y_data = feature_df[target_col].values
            X_data = feature_df.drop(target_col, axis=1).values

            X_sequences = []
            y_sequences = []

            for i in range(0, len(X_data) - sequence_length, step):
                X_sequences.append(X_data[i:i+sequence_length])
                y_sequences.append(y_data[i+sequence_length])

            return np.array(X_sequences), np.array(y_sequences)
        else:
            # Just features, no target
            X_data = feature_df.values
            X_sequences = []

            for i in range(0, len(X_data) - sequence_length + 1, step):
                X_sequences.append(X_data[i:i+sequence_length])

            return np.array(X_sequences)