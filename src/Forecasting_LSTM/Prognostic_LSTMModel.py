import numpy as np
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class PrognosticLSTMModel:
    """Enhanced LSTM model with operational mode support"""

    def __init__(self, n_features, sequence_length=30):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.model = None
        self.history = None

    def build_model(self, lstm_units=64, dropout_rate=0.3, learning_rate=0.001):
        """Build mode-aware LSTM model"""
        # Sensor data input
        sensor_input = Input(shape=(self.sequence_length, self.n_features), name='sensor_input')
        lstm1 = LSTM(lstm_units, return_sequences=True)(sensor_input)
        bn1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(dropout_rate)(bn1)

        lstm2 = LSTM(lstm_units//2, return_sequences=False)(dropout1)
        bn2 = BatchNormalization()(lstm2)

        # Operational mode input
        mode_input = Input(shape=(1,), name='mode_input')
        mode_embed = Dense(lstm_units//4, activation='relu')(mode_input)
        combined = Concatenate()([bn2, mode_embed])

        # Output layers
        dense1 = Dense(lstm_units//2, activation='relu')(combined)
        dropout2 = Dropout(dropout_rate/2)(dense1)
        output = Dense(self.n_features, activation='linear')(dropout2)

        self.model = Model(inputs=[sensor_input, mode_input], outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return self.model

    def create_sequences(self, data, modes=None):
        """Create sequences with optional mode information"""
        sequences = []
        targets = []
        mode_seqs = [] if modes is not None else None

        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(data[i + self.sequence_length])
            if modes is not None:
                mode_seqs.append(modes[i + self.sequence_length])

        if modes is not None:
            return np.array(sequences), np.array(targets), np.array(mode_seqs)
        return np.array(sequences), np.array(targets)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32,
              modes_train=None, modes_val=None):
        """Train with optional mode information"""
        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                            patience=5, factor=0.5, min_lr=1e-6)
        ]

        if modes_train is not None:
            # Mode-aware training
            val_data = ([X_val, modes_val], y_val) if X_val is not None else None
            self.history = self.model.fit(
                [X_train, modes_train], y_train,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Original training without modes
            val_data = (X_val, y_val) if X_val is not None else None
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        return self.history

    def save_model(self, save_path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, model_path):
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = load_model(model_path)
        print(f"Model loaded from: {model_path}")
        return self.model