"""Otimizacao de alocacao de budget entre canais de marketing."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

from src.config import MEDIA_CHANNELS


class BudgetOptimizer:
    """Otimiza a distribuicao do orcamento de marketing entre canais.

    Utiliza otimizacao numerica (scipy.optimize) para encontrar a alocacao
    que maximiza o retorno total, respeitando restricoes de budget e limites por canal.
    """

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_names: List[str],
        media_channels: Optional[List[str]] = None,
    ) -> None:
        self.model: Any = model
        self.scaler: Any = scaler
        self.feature_names: List[str] = feature_names
        self.media_channels: List[str] = media_channels or MEDIA_CHANNELS
        self.channel_indices: Dict[str, int] = {
            ch: self.feature_names.index(ch)
            for ch in self.media_channels
            if ch in self.feature_names
        }
        self.optimization_history: List[Dict[str, Any]] = []

    def _predict_revenue(
        self,
        allocation: np.ndarray,
        base_features: np.ndarray,
    ) -> float:
        """Prediz receita para uma alocacao de budget."""
        features: np.ndarray = base_features.copy()

        for i, channel in enumerate(self.channel_indices.keys()):
            idx: int = self.channel_indices[channel]
            features[idx] = allocation[i]

        features_scaled: np.ndarray = self.scaler.transform(features.reshape(1, -1))
        prediction: float = float(self.model.predict(features_scaled)[0])
        return prediction

    def _objective(
        self,
        allocation: np.ndarray,
        base_features: np.ndarray,
    ) -> float:
        """Funcao objetivo a ser minimizada (negativa da receita)."""
        return -self._predict_revenue(allocation, base_features)

    def optimize(
        self,
        total_budget: float,
        base_features: np.ndarray,
        min_allocation: Optional[Dict[str, float]] = None,
        max_allocation: Optional[Dict[str, float]] = None,
        method: str = "SLSQP",
    ) -> Dict[str, Any]:
        """Encontra alocacao otima de budget entre canais.

        Args:
            total_budget: Orcamento total disponivel.
            base_features: Features base (media ou ultimo periodo).
            min_allocation: Investimento minimo por canal.
            max_allocation: Investimento maximo por canal.
            method: Algoritmo de otimizacao scipy.

        Returns:
            Dicionario com alocacao otima e receita prevista.
        """
        channels: List[str] = list(self.channel_indices.keys())
        n_channels: int = len(channels)

        if n_channels == 0:
            raise ValueError("Nenhum canal de midia encontrado nas features do modelo.")

        initial_allocation: np.ndarray = np.full(n_channels, total_budget / n_channels)

        bounds: List[Tuple[float, float]] = []
        for ch in channels:
            lower: float = min_allocation.get(ch, 0) if min_allocation else 0
            upper: float = max_allocation.get(ch, total_budget) if max_allocation else total_budget
            bounds.append((lower, upper))

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - total_budget},
        ]

        result = minimize(
            self._objective,
            initial_allocation,
            args=(base_features,),
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        optimal_allocation: Dict[str, float] = {}
        for i, ch in enumerate(channels):
            optimal_allocation[ch] = float(result.x[i])

        optimal_revenue: float = -float(result.fun)
        current_revenue: float = self._predict_revenue(initial_allocation, base_features)
        improvement: float = ((optimal_revenue - current_revenue) / abs(current_revenue) * 100) if current_revenue != 0 else 0

        output: Dict[str, Any] = {
            "total_budget": total_budget,
            "optimal_allocation": optimal_allocation,
            "predicted_revenue": optimal_revenue,
            "baseline_revenue": current_revenue,
            "improvement_pct": improvement,
            "converged": result.success,
            "iterations": result.nit,
        }

        self.optimization_history.append(output)
        logger.info(
            "Otimizacao concluida: budget={:.0f}, receita prevista={:.0f} (+{:.2f}%)",
            total_budget,
            optimal_revenue,
            improvement,
        )
        return output

    def optimize_range(
        self,
        budget_range: np.ndarray,
        base_features: np.ndarray,
        min_allocation: Optional[Dict[str, float]] = None,
        max_allocation: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Otimiza para uma faixa de budgets."""
        results: List[Dict[str, Any]] = []

        for budget in budget_range:
            result: Dict[str, Any] = self.optimize(
                total_budget=float(budget),
                base_features=base_features,
                min_allocation=min_allocation,
                max_allocation=max_allocation,
            )
            flat: Dict[str, Any] = {
                "budget": budget,
                "predicted_revenue": result["predicted_revenue"],
                "improvement_pct": result["improvement_pct"],
            }
            for ch, val in result["optimal_allocation"].items():
                flat[f"alloc_{ch}"] = val
            results.append(flat)

        df: pd.DataFrame = pd.DataFrame(results)
        logger.info("Otimizacao de faixa concluida: {} niveis de budget", len(budget_range))
        return df

    def marginal_roas(
        self,
        base_features: np.ndarray,
        channel: str,
        increments: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Calcula ROAS marginal para incrementos no investimento de um canal."""
        if channel not in self.channel_indices:
            raise ValueError(f"Canal nao encontrado: {channel}")

        if increments is None:
            increments = np.arange(1000, 50001, 1000)

        idx: int = self.channel_indices[channel]
        base_revenue: float = self._predict_revenue(
            np.array([base_features[self.channel_indices[ch]] for ch in self.channel_indices]),
            base_features,
        )

        results: List[Dict[str, float]] = []
        for inc in increments:
            modified_features: np.ndarray = base_features.copy()
            modified_features[idx] += inc

            new_revenue: float = float(
                self.model.predict(self.scaler.transform(modified_features.reshape(1, -1)))[0]
            )
            marginal: float = (new_revenue - base_revenue) / inc if inc > 0 else 0

            results.append({
                "channel": channel,
                "increment": float(inc),
                "base_revenue": base_revenue,
                "new_revenue": new_revenue,
                "marginal_revenue": new_revenue - base_revenue,
                "marginal_roas": marginal,
            })

        df: pd.DataFrame = pd.DataFrame(results)
        logger.info("ROAS marginal calculado para {} com {} incrementos", channel, len(increments))
        return df

    def get_allocation_summary(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Formata resultado da otimizacao como DataFrame."""
        allocation: Dict[str, float] = result["optimal_allocation"]
        total: float = sum(allocation.values())

        rows: List[Dict[str, Any]] = []
        for ch, value in allocation.items():
            rows.append({
                "canal": ch.replace("_spend", ""),
                "investimento": value,
                "percentual": (value / total * 100) if total > 0 else 0,
            })

        return pd.DataFrame(rows).sort_values("investimento", ascending=False).reset_index(drop=True)


# "O orcamento nao e apenas um documento contabil. E a expressao de nossos valores." - Jacob Lew
