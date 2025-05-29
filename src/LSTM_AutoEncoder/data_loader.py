import os
import pandas as pd


class DataLoader:
    """Class for loading and managing the dataset"""

    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.datasets = {}

    def load_dataset(self, dataset_id='FD001'):
        """
        Load training, test and RUL data for a specific engine dataset

        Parameters:
        dataset_id (str): Dataset ID ('FD001', 'FD002', 'FD003', or 'FD004')

        Returns:
        dict: Dictionary containing train, test and RUL data
        """
        # Define file paths
        train_file = os.path.join(self.data_dir, f'/content/drive/MyDrive/CMAPSSData/train_{dataset_id}.txt')
        test_file = os.path.join(self.data_dir, f'/content/drive/MyDrive//CMAPSSData/test_{dataset_id}.txt')
        rul_file = os.path.join(self.data_dir, f'/content/drive/MyDrive/CMAPSSData/RUL_{dataset_id}.txt')

        # Define column names
        sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        op_setting_columns = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        columns = ['engine_id', 'cycle'] + op_setting_columns + sensor_columns

        # Load train data
        train_data = pd.read_csv(train_file, sep=' ', header=None)
        train_data.drop(columns=[26, 27], inplace=True)  # Remove NaN columns
        train_data.columns = columns

        # Load test data
        test_data = pd.read_csv(test_file, sep=' ', header=None)
        test_data.drop(columns=[26, 27], inplace=True)  # Remove NaN columns
        test_data.columns = columns

        # Load RUL data
        rul_data = pd.read_csv(rul_file, sep=' ', header=None)
        rul_data.drop(columns=[1], inplace=True)  # Remove NaN column
        rul_data.columns = ['RUL']

        dataset = {
            'train': train_data,
            'test': test_data,
            'rul': rul_data
        }

        self.datasets[dataset_id] = dataset
        return dataset

    def get_dataset(self, dataset_id):
        """Get previously loaded dataset"""
        if dataset_id not in self.datasets:
            return self.load_dataset(dataset_id)
        return self.datasets[dataset_id]