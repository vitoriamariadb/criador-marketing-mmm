"""Validacao cruzada adaptada para series temporais de marketing."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class TimeSeriesCV:
    """Validacao cruzada temporal para modelos de Marketing Mix."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ) -> None:
        self.n_splits: int = n_splits
        self.test_size: Optional[int] = test_size
        self.gap: int = gap
        self.cv_results: List[Dict[str, Any]] = []

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class: type,
        model_params: Optional[dict] = None,
        scale: bool = True,
    ) -> Dict[str, Any]:
        """Executa validacao cruzada temporal."""
        params: dict = model_params or {}
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=self.gap,
        )

        self.cv_results = []
        fold_metrics: List[Dict[str, float]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            if scale:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred: np.ndarray = model.predict(X_test)

            metrics: Dict[str, float] = {
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "mape": float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100),
            }

            fold_metrics.append(metrics)
            logger.debug("Fold {}: R2={:.4f}, RMSE={:.2f}", fold, metrics["r2"], metrics["rmse"])

        self.cv_results = fold_metrics

        summary: Dict[str, Any] = self._compute_summary(fold_metrics)
        logger.info(
            "CV concluida ({} folds): R2 medio={:.4f} (+/- {:.4f})",
            self.n_splits,
            summary["mean_r2"],
            summary["std_r2"],
        )
        return summary

    def _compute_summary(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Computa estatisticas resumo da validacao cruzada."""
        metrics_df: pd.DataFrame = pd.DataFrame(fold_metrics)

        summary: Dict[str, Any] = {
            "n_folds": self.n_splits,
            "mean_r2": float(metrics_df["r2"].mean()),
            "std_r2": float(metrics_df["r2"].std()),
            "mean_rmse": float(metrics_df["rmse"].mean()),
            "std_rmse": float(metrics_df["rmse"].std()),
            "mean_mae": float(metrics_df["mae"].mean()),
            "std_mae": float(metrics_df["mae"].std()),
            "mean_mape": float(metrics_df["mape"].mean()),
            "std_mape": float(metrics_df["mape"].std()),
            "fold_details": fold_metrics,
        }

        return summary

    def get_results_dataframe(self) -> pd.DataFrame:
        """Retorna resultados de cada fold como DataFrame."""
        if not self.cv_results:
            raise RuntimeError("Validacao cruzada nao foi executada.")
        return pd.DataFrame(self.cv_results)


class HyperparameterSearch:
    """Busca de hiperparametros com validacao cruzada temporal."""

    MODEL_MAP = {
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet,
    }

    def __init__(
        self,
        model_type: str = "ridge",
        n_splits: int = 5,
        metric: str = "r2",
    ) -> None:
        self.model_type: str = model_type
        self.n_splits: int = n_splits
        self.metric: str = metric
        self.search_results: List[Dict[str, Any]] = []

    def grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Executa grid search com validacao cruzada temporal."""
        model_class = self.MODEL_MAP.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Tipo de modelo desconhecido: {self.model_type}")

        param_combinations: List[dict] = self._generate_combinations(param_grid)
        self.search_results = []

        logger.info(
            "Iniciando grid search: {} combinacoes x {} folds",
            len(param_combinations),
            self.n_splits,
        )

        best_score: float = -np.inf
        best_params: dict = {}

        for i, params in enumerate(param_combinations):
            cv = TimeSeriesCV(n_splits=self.n_splits)
            summary: Dict[str, Any] = cv.validate(X, y, model_class, params)

            score: float = summary[f"mean_{self.metric}"]
            result_entry: Dict[str, Any] = {
                "params": params,
                "mean_score": score,
                "std_score": summary[f"std_{self.metric}"],
                "mean_rmse": summary["mean_rmse"],
            }
            self.search_results.append(result_entry)

            if score > best_score:
                best_score = score
                best_params = params

            if (i + 1) % 10 == 0:
                logger.info("  Progresso: {}/{} combinacoes", i + 1, len(param_combinations))

        logger.info(
            "Grid search concluido. Melhor {}: {:.4f} com params={}",
            self.metric,
            best_score,
            best_params,
        )

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_combinations": len(param_combinations),
            "all_results": self.search_results,
        }

    def _generate_combinations(self, param_grid: Dict[str, List[Any]]) -> List[dict]:
        """Gera todas as combinacoes de parametros."""
        keys: List[str] = list(param_grid.keys())
        values: List[List[Any]] = list(param_grid.values())

        combinations: List[dict] = []
        self._recursive_combine(keys, values, 0, {}, combinations)
        return combinations

    def _recursive_combine(
        self,
        keys: List[str],
        values: List[List[Any]],
        depth: int,
        current: dict,
        result: List[dict],
    ) -> None:
        """Gera combinacoes recursivamente."""
        if depth == len(keys):
            result.append(current.copy())
            return

        for val in values[depth]:
            current[keys[depth]] = val
            self._recursive_combine(keys, values, depth + 1, current, result)

    def get_results_dataframe(self) -> pd.DataFrame:
        """Retorna resultados da busca como DataFrame."""
        if not self.search_results:
            raise RuntimeError("Grid search nao foi executado.")

        rows: List[dict] = []
        for r in self.search_results:
            row: dict = {**r["params"], "mean_score": r["mean_score"], "std_score": r["std_score"]}
            rows.append(row)

        return pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)


def get_default_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Retorna grids de hiperparametros padrao para cada tipo de modelo."""
    return {
        "ridge": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "lasso": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        },
        "elasticnet": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
    }


# "A experiencia sem teoria e cega, a teoria sem experiencia e mero jogo intelectual." - Immanuel Kant
