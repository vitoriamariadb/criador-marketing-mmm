"""Modelos regularizados (Ridge, Lasso, ElasticNet) para MMM."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_TARGET_COLUMN, MODEL_PARAMS


class ModelType(str, Enum):
    """Tipos de modelo de regressao regularizada."""
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTICNET = "elasticnet"


class RegularizedMMM:
    """Modelos regularizados para Marketing Mix Modeling.

    Ridge: penaliza L2 (soma dos quadrados dos coeficientes).
    Lasso: penaliza L1 (soma dos valores absolutos) - promove sparsidade.
    ElasticNet: combina L1 e L2 - equilibrio entre selecao e estabilidade.
    """

    MODEL_CLASSES = {
        ModelType.RIDGE: Ridge,
        ModelType.LASSO: Lasso,
        ModelType.ELASTICNET: ElasticNet,
    }

    def __init__(
        self,
        model_type: str = "ridge",
        target_column: str = DEFAULT_TARGET_COLUMN,
        test_size: float = 0.2,
        random_state: int = 42,
        **model_kwargs: Any,
    ) -> None:
        self.model_type: ModelType = ModelType(model_type)
        self.target_column: str = target_column
        self.test_size: float = test_size
        self.random_state: int = random_state

        default_params: dict = MODEL_PARAMS.get(model_type, {})
        self.model_params: dict = {**default_params, **model_kwargs}

        self.model: Optional[Any] = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepara e divide dados em treino e teste."""
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
            "Dados preparados ({}): treino={}, teste={}, features={}",
            self.model_type.value,
            len(X_train),
            len(X_test),
            len(feature_columns),
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RegularizedMMM":
        """Treina modelo regularizado."""
        X_scaled: np.ndarray = self.scaler.fit_transform(X_train)

        model_class = self.MODEL_CLASSES[self.model_type]
        self.model = model_class(**self.model_params)
        self.model.fit(X_scaled, y_train)

        self._X_train = X_scaled
        self._y_train = y_train
        self.is_fitted = True

        n_nonzero: int = int(np.sum(np.abs(self.model.coef_) > 1e-10))
        logger.info(
            "Modelo {} treinado: {} coeficientes nao-zero de {}",
            self.model_type.value,
            n_nonzero,
            len(self.feature_names),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera predicoes."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo nao treinado.")

        X_scaled: np.ndarray = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Avalia modelo no conjunto de teste."""
        predictions: np.ndarray = self.predict(X_test)

        self.metrics = {
            "r2": float(r2_score(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mape": float(np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100),
            "adj_r2": self._adjusted_r2(y_test, predictions, X_test.shape[1]),
        }

        logger.info(
            "Avaliacao {}: R2={:.4f}, Adj-R2={:.4f}, RMSE={:.2f}",
            self.model_type.value,
            self.metrics["r2"],
            self.metrics["adj_r2"],
            self.metrics["rmse"],
        )
        return self.metrics

    def _adjusted_r2(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int
    ) -> float:
        """Calcula R-quadrado ajustado."""
        n: int = len(y_true)
        r2: float = r2_score(y_true, y_pred)

        if n <= n_features + 1:
            return r2

        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    def get_coefficients(self) -> pd.DataFrame:
        """Retorna coeficientes com importancia relativa."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo nao treinado.")

        coefs: np.ndarray = self.model.coef_
        abs_coefs: np.ndarray = np.abs(coefs)
        total: float = float(abs_coefs.sum()) or 1.0

        result: pd.DataFrame = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefs,
            "abs_coefficient": abs_coefs,
            "relative_importance": abs_coefs / total * 100,
            "is_active": abs_coefs > 1e-10,
        })

        return result.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    def get_channel_contribution(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calcula contribuicao absoluta de cada feature."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo nao treinado.")

        cols: List[str] = feature_columns or self.feature_names
        X_scaled: np.ndarray = self.scaler.transform(df[cols].values)

        contributions: Dict[str, np.ndarray] = {}
        for i, col in enumerate(cols):
            contributions[col] = X_scaled[:, i] * self.model.coef_[i]

        contributions["base"] = np.full(len(df), self.model.intercept_)
        result: pd.DataFrame = pd.DataFrame(contributions, index=df.index)

        logger.info("Contribuicoes calculadas ({}) para {} features", self.model_type.value, len(cols))
        return result

    def summary(self) -> Dict[str, Any]:
        """Resumo do modelo."""
        if not self.is_fitted:
            return {"status": "nao treinado", "type": self.model_type.value}

        return {
            "model_type": self.model_type.value,
            "params": self.model_params,
            "n_features": len(self.feature_names),
            "n_active_features": int(np.sum(np.abs(self.model.coef_) > 1e-10)),
            "metrics": self.metrics,
            "intercept": float(self.model.intercept_),
        }


def compare_models(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Compara Ridge, Lasso e ElasticNet no mesmo dataset."""
    results: List[Dict[str, Any]] = []

    for model_type in ModelType:
        mmm = RegularizedMMM(model_type=model_type.value, target_column=target_column, test_size=test_size)
        X_train, X_test, y_train, y_test = mmm.prepare_data(df, feature_columns)
        mmm.fit(X_train, y_train)
        metrics: Dict[str, float] = mmm.evaluate(X_test, y_test)

        results.append({
            "model": model_type.value,
            "r2": metrics["r2"],
            "adj_r2": metrics["adj_r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "n_active": int(np.sum(np.abs(mmm.model.coef_) > 1e-10)),
        })

    comparison: pd.DataFrame = pd.DataFrame(results)
    logger.info("Comparacao concluida entre {} modelos", len(results))
    return comparison


# "A perfeicao nao e atingida quando nao ha mais nada a adicionar, mas quando nao ha mais nada a remover." - Antoine de Saint-Exupery

