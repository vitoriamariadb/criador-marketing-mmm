"""Modelos de regressao para Marketing Mix Modeling."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_TARGET_COLUMN


class MMMRegressor:
    """Modelo de regressao linear para atribuicao de canais de marketing."""

    def __init__(
        self,
        target_column: str = DEFAULT_TARGET_COLUMN,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.target_column: str = target_column
        self.test_size: float = test_size
        self.random_state: int = random_state

        self.model: Optional[LinearRegression] = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepara dados para treino e teste."""
        if self.target_column not in df.columns:
            raise ValueError(f"Coluna target nao encontrada: {self.target_column}")

        if feature_columns is None:
            exclude: set = {self.target_column, "date"}
            feature_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        self.feature_names = feature_columns
        X: np.ndarray = df[feature_columns].values
        y: np.ndarray = df[self.target_column].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        logger.info(
            "Dados preparados: treino={} amostras, teste={} amostras, {} features",
            len(X_train),
            len(X_test),
            len(feature_columns),
        )
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        normalize: bool = True,
    ) -> "MMMRegressor":
        """Treina modelo de regressao linear."""
        if normalize:
            X_train = self.scaler.fit_transform(X_train)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        logger.info("Modelo LinearRegression treinado com {} amostras", len(X_train))
        return self

    def predict(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Gera predicoes com o modelo treinado."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo nao foi treinado. Execute fit() primeiro.")

        if normalize:
            X = self.scaler.transform(X)

        predictions: np.ndarray = self.model.predict(X)
        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """Avalia performance do modelo no conjunto de teste."""
        predictions: np.ndarray = self.predict(X_test, normalize)

        self.metrics = {
            "r2": float(r2_score(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mape": float(np.mean(np.abs((y_test - predictions) / y_test)) * 100),
        }

        logger.info(
            "Avaliacao: R2={:.4f}, RMSE={:.2f}, MAE={:.2f}, MAPE={:.2f}%",
            self.metrics["r2"],
            self.metrics["rmse"],
            self.metrics["mae"],
            self.metrics["mape"],
        )
        return self.metrics

    def get_coefficients(self) -> pd.DataFrame:
        """Retorna coeficientes do modelo com nomes das features."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo nao foi treinado.")

        coefs: pd.DataFrame = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_,
            "abs_coefficient": np.abs(self.model.coef_),
        })
        coefs = coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        coefs["intercept"] = self.model.intercept_

        return coefs

    def get_contribution(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calcula contribuicao de cada canal para a variavel target."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo nao foi treinado.")

        cols: List[str] = feature_columns or self.feature_names
        X: np.ndarray = self.scaler.transform(df[cols].values)

        contributions: Dict[str, np.ndarray] = {}
        for i, col in enumerate(cols):
            contributions[col] = X[:, i] * self.model.coef_[i]

        contributions["intercept"] = np.full(len(df), self.model.intercept_)

        result: pd.DataFrame = pd.DataFrame(contributions)
        logger.info("Contribuicoes calculadas para {} features", len(cols))
        return result

    def summary(self) -> Dict[str, Any]:
        """Retorna resumo completo do modelo."""
        if not self.is_fitted:
            return {"status": "modelo nao treinado"}

        return {
            "model_type": "LinearRegression",
            "n_features": len(self.feature_names),
            "features": self.feature_names,
            "metrics": self.metrics,
            "intercept": float(self.model.intercept_) if self.model else None,
        }


# "Todos os modelos estao errados, mas alguns sao uteis." - George Box

