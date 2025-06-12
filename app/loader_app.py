import pandas as pd
import os


class DataLoader:
    """Load CMAPSS datasets"""

    @staticmethod
    def load_dataset(dataset_name, data_path=None):
        """Load CMAPSS dataset by name (FD001, FD002, FD003, FD004)"""
        sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        operational_columns = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        columns = ['unit_id', 'time_cycle'] + operational_columns + sensor_columns

        if data_path and os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path, sep=' ', header=None)
                df = df.dropna(axis=1, how='all')
                df = df.iloc[:, :len(columns)]
                df.columns = columns
                return df
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return None
        else:
            st.warning(f"Dataset {dataset_name} not found. Please upload the file.")
            return None
