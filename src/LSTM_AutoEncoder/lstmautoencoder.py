import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping


class LSTMAutoencoder:
    """Class for LSTM Autoencoder model"""

    def __init__(self, encoding_dim=16):
        self.encoding_dim = encoding_dim
        self.encoder = None
        self.autoencoder = None
        self.history = None
        self.input_shape = None

    def build_model(self, input_shape):
        """
        Build LSTM Autoencoder for anomaly detection
        """
        self.input_shape = input_shape

        # Define input layer
        inputs = Input(shape=input_shape)

        # Encoder
        encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
        encoded = LSTM(self.encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = RepeatVector(input_shape[0])(encoded)
        decoded = LSTM(self.encoding_dim, activation='relu', return_sequences=True)(decoded)
        decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(input_shape[1]))(decoded)

        # Autoencoder model
        self.autoencoder = Model(inputs, decoded)

        # Encoder model
        self.encoder = Model(inputs, encoded)

        # Compile the model
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train(self, X_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train LSTM Autoencoder model"""
        if self.autoencoder is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the autoencoder
        self.history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )

        return self.history

    def predict(self, X_data):
        """Get reconstructions from the autoencoder"""
        return self.autoencoder.predict(X_data)

    def encode(self, X_data):
        """Get encoded representations"""
        return self.encoder.predict(X_data)