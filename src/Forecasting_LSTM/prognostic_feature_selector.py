import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
import pickle
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Input, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model


class PrognosticFeatureSelector:
    """Feature selector for prognostic models with various feature selection methods"""

    def __init__(self):
        self.feature_scores = None
        self.selected_features = None

    def calculate_prognostic_relevance(self, df, top_k=10, method='spearman'):
        """Calculate feature relevance for RUL prediction using specified method"""
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        rul = df['RUL'].values

        # Calculate feature scores based on selected method
        if method == 'spearman':
            # Use Spearman correlation with RUL
            corrs = []
            for sensor in sensor_cols:
                corr, _ = spearmanr(df[sensor].values, rul)
                corrs.append(abs(corr) if not np.isnan(corr) else 0)
            self.feature_scores = dict(zip(sensor_cols, corrs))

        elif method == 'variance':
            # Use variance of each sensor
            variances = [df[sensor].var() for sensor in sensor_cols]
            self.feature_scores = dict(zip(sensor_cols, variances))

        elif method == 'hybrid':
            # Combine Spearman correlation and variance
            corrs = []
            for sensor in sensor_cols:
                corr, _ = spearmanr(df[sensor].values, rul)
                corrs.append(abs(corr) if not np.isnan(corr) else 0)
            corr_scores = np.array(corrs)

            variances = np.array([df[sensor].var() for sensor in sensor_cols])

            # Normalize both metrics to 0-1 range
            if corr_scores.max() > corr_scores.min():
                corr_norm = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min())
            else:
                corr_norm = corr_scores

            if variances.max() > variances.min():
                var_norm = (variances - variances.min()) / (variances.max() - variances.min())
            else:
                var_norm = variances

            # Combine with equal weight
            combined_scores = 0.5 * corr_norm + 0.5 * var_norm
            self.feature_scores = dict(zip(sensor_cols, combined_scores))

        else:
            raise ValueError("Method must be 'spearman', 'variance', or 'hybrid'")

        # Select top k features
        sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [feat[0] for feat in sorted_features[:top_k]]

        print("\nSelected features:")
        for i, (feat, score) in enumerate(sorted_features[:top_k], 1):
            print(f"{i}. {feat}: {score:.4f}")

        return self.selected_features