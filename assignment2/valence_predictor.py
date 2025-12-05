"""
ValencePredictor: Trains linear regression and evaluates predictions.
"""
from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


class ValencePredictor:
    """Trains and evaluates a linear regression model for valence prediction."""

    def __init__(self):
        """
        Initialize the ValencePredictor.

        Args:
            None

        Returns:
            None
        """
        self.model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            None (fits the internal model)
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict valence scores.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Numpy array of predicted valence scores of shape (n_samples,)
        """
        return self.model.predict(X)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Calculate MSE and Pearson correlation.

        Args:
            y_true: Ground truth valence scores
            y_pred: Predicted valence scores

        Returns:
            Tuple of (MSE, Pearson correlation coefficient)
        """
        mse = mean_squared_error(y_true, y_pred)
        corr, _ = pearsonr(y_true, y_pred)
        return mse, corr
