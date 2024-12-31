"""Analise de importancia de features e decomposicao de contribuicao."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_TARGET_COLUMN


class FeatureImportanceAnalyzer:
    """Analisa importancia de features usando multiplos metodos."""

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        self.feature_names: List[str] = feature_names or []
        self.importance_results: Dict[str, pd.DataFrame] = {}

    def coefficient_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Calcula importancia baseada nos coeficientes do modelo."""
        names: List[str] = feature_names or self.feature_names

        if not hasattr(model, "coef_"):
            raise ValueError("Modelo nao possui atributo coef_")

        coefs: np.ndarray = model.coef_
        abs_coefs: np.ndarray = np.abs(coefs)
        total: float = float(abs_coefs.sum()) or 1.0

        result: pd.DataFrame = pd.DataFrame({
            "feature": names,
            "coefficient": coefs,
            "abs_importance": abs_coefs,
            "relative_pct": (abs_coefs / total) * 100,
            "direction": ["positivo" if c > 0 else "negativo" for c in coefs],
        })

        result = result.sort_values("abs_importance", ascending=False).reset_index(drop=True)
        self.importance_results["coefficient"] = result
        logger.info("Importancia por coeficientes calculada para {} features", len(names))
        return result

    def permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Calcula importancia por permutacao."""
        names: List[str] = feature_names or self.feature_names

        perm_result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="r2"
        )

        result: pd.DataFrame = pd.DataFrame({
            "feature": names,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        })

        result = result.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        self.importance_results["permutation"] = result
        logger.info(
            "Importancia por permutacao calculada ({} repeticoes) para {} features",
            n_repeats,
            len(names),
        )
        return result

    def decompose_revenue(
        self,
        model: Any,
        df: pd.DataFrame,
        feature_columns: List[str],
        scaler: Optional[StandardScaler] = None,
        target_column: str = DEFAULT_TARGET_COLUMN,
    ) -> pd.DataFrame:
        """Decompoe a receita em contribuicoes por canal."""
        X: np.ndarray = df[feature_columns].values
        if scaler is not None:
            X = scaler.transform(X)

        contributions: Dict[str, np.ndarray] = {}
        for i, col in enumerate(feature_columns):
            contributions[col] = X[:, i] * model.coef_[i]

        contributions["base_intercept"] = np.full(len(df), model.intercept_)

        result: pd.DataFrame = pd.DataFrame(contributions, index=df.index)

        total_contribution: pd.Series = result.sum(axis=1)
        for col in result.columns:
            result[f"{col}_pct"] = result[col] / total_contribution * 100

        self.importance_results["decomposition"] = result
        logger.info("Decomposicao de receita calculada para {} features", len(feature_columns))
        return result

    def roi_analysis(
        self,
        df: pd.DataFrame,
        contributions: pd.DataFrame,
        spend_columns: List[str],
    ) -> pd.DataFrame:
        """Calcula ROI e ROAS de cada canal de midia."""
        roi_data: List[Dict[str, Any]] = []

        for spend_col in spend_columns:
            if spend_col not in df.columns:
                continue

            contrib_candidates: List[str] = [
                c for c in contributions.columns
                if spend_col in c and "_pct" not in c
            ]

            if not contrib_candidates:
                continue

            contrib_col: str = contrib_candidates[0]
            total_spend: float = float(df[spend_col].sum())
            total_contribution: float = float(contributions[contrib_col].sum())

            roas: float = total_contribution / total_spend if total_spend > 0 else 0.0
            roi: float = (total_contribution - total_spend) / total_spend if total_spend > 0 else 0.0

            channel_name: str = spend_col.replace("_spend", "").replace("_adstock", "").replace("_saturated", "")
            roi_data.append({
                "channel": channel_name,
                "total_spend": total_spend,
                "total_contribution": total_contribution,
                "roas": roas,
                "roi_pct": roi * 100,
                "cpa": total_spend / max(total_contribution, 1),
            })

        result: pd.DataFrame = pd.DataFrame(roi_data)
        if not result.empty:
            result = result.sort_values("roas", ascending=False).reset_index(drop=True)

        self.importance_results["roi"] = result
        logger.info("Analise de ROI calculada para {} canais", len(roi_data))
        return result

    def get_top_features(self, method: str = "coefficient", top_n: int = 10) -> pd.DataFrame:
        """Retorna top N features por metodo de importancia."""
        if method not in self.importance_results:
            raise ValueError(f"Metodo '{method}' nao calculado. Execute primeiro.")

        return self.importance_results[method].head(top_n)

    def summary(self) -> Dict[str, int]:
        """Resumo dos metodos de importancia calculados."""
        return {method: len(df) for method, df in self.importance_results.items()}


# "A medida do que somos e o que fazemos com o que temos." - Vince Lombardi

