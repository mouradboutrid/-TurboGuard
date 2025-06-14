import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class DataProcessor:
    """Data processor with operational mode clustering"""

    def __init__(self):
        self.sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        self.operational_columns = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        self.scaler = MinMaxScaler()
        self.op_mode_scalers = {}  # Store scalers per operational mode

    def load_cmapss_data(self, filepath):
        """Load and preprocess C-MAPSS dataset with operational mode handling"""
        print(f"Loading C-MAPSS data from: {filepath}")
        columns = ['unit_id', 'time_cycle'] + self.operational_columns + self.sensor_columns

        try:
            df = pd.read_csv(filepath, sep=' ', header=None)
            df = df.dropna(axis=1, how='all')
            df = df.iloc[:, :len(columns)]
            df.columns = columns

            # Calculate RUL
            df = self.calculate_rul(df)

            # Cluster operational modes
            op_data = df[self.operational_columns].values
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['op_mode'] = kmeans.fit_predict(op_data)

            # Normalize within each operational mode
            normalized_dfs = []
            for mode in df['op_mode'].unique():
                mode_data = df[df['op_mode'] == mode].copy()
                scaler = MinMaxScaler()
                mode_data[self.sensor_columns] = scaler.fit_transform(mode_data[self.sensor_columns])
                self.op_mode_scalers[mode] = scaler
                normalized_dfs.append(mode_data)

            df = pd.concat(normalized_dfs)
            df = self.remove_constant_sensors(df)

            print(f"Data loaded successfully: {len(df)} samples, {df['unit_id'].nunique()} units")
            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def load_multiple_datasets(self, filepaths):
        """Load multiple C-MAPSS datasets and return individual and combined DataFrames"""
        individual_datasets = {}
        combined_dfs = []

        for name, path in filepaths.items():
            try:
                df = self.load_cmapss_data(path)
                individual_datasets[name] = df
                combined_dfs.append(df)
            except Exception as e:
                print(f"Error loading dataset {name}: {e}")
                continue

        combined_df = pd.concat(combined_dfs, ignore_index=True) if combined_dfs else None
        return individual_datasets, combined_df

    def calculate_rul(self, df):
        """Calculate Remaining Useful Life for each unit"""
        max_cycles = df.groupby('unit_id')['time_cycle'].max()
        df['max_cycle'] = df['unit_id'].map(max_cycles)
        df['RUL'] = df['max_cycle'] - df['time_cycle']
        df = df.drop('max_cycle', axis=1)
        return df

    def remove_constant_sensors(self, df, variance_threshold=1e-6):
        """Remove sensors with very low variance"""
        sensors_to_remove = []
        for sensor in self.sensor_columns:
            if sensor in df.columns:
                if df[sensor].var() < variance_threshold:
                    sensors_to_remove.append(sensor)
        if sensors_to_remove:
            df = df.drop(columns=sensors_to_remove)
            print(f"Removed constant sensors: {sensors_to_remove}")
        return df
