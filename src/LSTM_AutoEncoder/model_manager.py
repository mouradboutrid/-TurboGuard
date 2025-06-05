import os
import pickle
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from src.LSTM_Autoencoder.lstm_autoencoder import LSTMAutoencoder  
from src.LSTM_Autoencoder.data_preprocessor import DataPreprocessor 


class ModelManager:
    """Class for saving and loading models and configurations"""

    def __init__(self, base_dir='./models'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_model_package(self, dataset_id, autoencoder, preprocessor, config):
        """
        Save complete model package including models, preprocessor, and configuration

        Parameters:
        dataset_id (str): Dataset identifier
        autoencoder (LSTMAutoencoder): Trained autoencoder
        preprocessor (DataPreprocessor): Fitted preprocessor
        config (dict): Configuration parameters
        """
        model_dir = os.path.join(self.base_dir, dataset_id)
        os.makedirs(model_dir, exist_ok=True)

        # Save models
        encoder_path = os.path.join(model_dir, 'encoder.keras')
        autoencoder_path = os.path.join(model_dir, 'autoencoder.keras')

        autoencoder.encoder.save(encoder_path)
        autoencoder.autoencoder.save(autoencoder_path)

        # Save preprocessor scalers
        scaler_path = os.path.join(model_dir, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(preprocessor.scalers, f)

        # Save preprocessing parameters
        preprocessing_path = os.path.join(model_dir, 'preprocessing_params.json')
        with open(preprocessing_path, 'w') as f:
            json.dump(preprocessor.preprocessing_params, f, indent=2)

        # Save configuration
        config_path = os.path.join(model_dir, 'config.json')
        config['timestamp'] = datetime.now().isoformat()
        config['input_shape'] = autoencoder.input_shape
        config['encoding_dim'] = autoencoder.encoding_dim

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Complete model package saved to: {model_dir}")
        return model_dir

    def load_model_package(self, dataset_id):
        """
        Load complete model package

        Parameters:
        dataset_id (str): Dataset identifier

        Returns:
        tuple: (autoencoder, preprocessor, config)
        """
        model_dir = os.path.join(self.base_dir, dataset_id)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model package not found: {model_dir}")

        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load models
        encoder_path = os.path.join(model_dir, 'encoder.keras')
        autoencoder_path = os.path.join(model_dir, 'autoencoder.keras')

        # Create autoencoder instance
        autoencoder = LSTMAutoencoder(encoding_dim=config['encoding_dim'])
        autoencoder.encoder = load_model(encoder_path)
        autoencoder.autoencoder = load_model(autoencoder_path)
        autoencoder.input_shape = tuple(config['input_shape'])

        # Load preprocessor
        preprocessor = DataPreprocessor()

        # Load scalers
        scaler_path = os.path.join(model_dir, 'scalers.pkl')
        with open(scaler_path, 'rb') as f:
            preprocessor.scalers = pickle.load(f)

        # Load preprocessing parameters
        preprocessing_path = os.path.join(model_dir, 'preprocessing_params.json')
        with open(preprocessing_path, 'r') as f:
            preprocessor.preprocessing_params = json.load(f)

        print(f"Model package loaded from: {model_dir}")
        return autoencoder, preprocessor, config

    def list_available_models(self):
        """List all available model packages"""
        models = []
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                model_dir = os.path.join(self.base_dir, item)
                if os.path.isdir(model_dir):
                    config_path = os.path.join(model_dir, 'config.json')
                    if os.path.exists(config_path):
                        models.append(item)
        return models
