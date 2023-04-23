"""Planejamento de cenarios para simulacao de estrategias de marketing."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Scenario:
    """Representa um cenario de alocacao de marketing."""
    name: str
    description: str
    allocation: Dict[str, float]
    predicted_revenue: float = 0.0
    predicted_roas: float = 0.0
    total_spend: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScenarioPlanner:
    """Planeja e compara cenarios de alocacao de marketing."""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_names: List[str],
        media_channels: List[str],
    ) -> None:
        self.model: Any = model
        self.scaler: Any = scaler
        self.feature_names: List[str] = feature_names
        self.media_channels: List[str] = media_channels
        self.channel_indices: Dict[str, int] = {
            ch: feature_names.index(ch)
            for ch in media_channels
            if ch in feature_names
        }
        self.scenarios: List[Scenario] = []

    def _predict_for_allocation(
        self, allocation: Dict[str, float], base_features: np.ndarray
    ) -> float:
        """Prediz receita para uma alocacao especifica."""
        features: np.ndarray = base_features.copy()

        for channel, value in allocation.items():
            if channel in self.channel_indices:
                features[self.channel_indices[channel]] = value

        features_scaled: np.ndarray = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(features_scaled)[0])

    def create_scenario(
        self,
        name: str,
        description: str,
        allocation: Dict[str, float],
        base_features: np.ndarray,
    ) -> Scenario:
        """Cria e avalia um cenario."""
        predicted_revenue: float = self._predict_for_allocation(allocation, base_features)
        total_spend: float = sum(allocation.values())
        roas: float = predicted_revenue / total_spend if total_spend > 0 else 0

        scenario = Scenario(
            name=name,
            description=description,
            allocation=allocation,
            predicted_revenue=predicted_revenue,
            predicted_roas=roas,
            total_spend=total_spend,
        )

        self.scenarios.append(scenario)
        logger.info(
            "Cenario '{}' criado: spend={:.0f}, receita={:.0f}, ROAS={:.2f}",
            name,
            total_spend,
            predicted_revenue,
            roas,
        )
        return scenario

    def create_baseline(
        self, base_features: np.ndarray, name: str = "Baseline"
    ) -> Scenario:
        """Cria cenario baseline com alocacao atual."""
        current_allocation: Dict[str, float] = {}
        for ch in self.channel_indices:
            current_allocation[ch] = float(base_features[self.channel_indices[ch]])

        return self.create_scenario(
            name=name,
            description="Cenario base com alocacao atual",
            allocation=current_allocation,
            base_features=base_features,
        )

    def create_proportional_scenario(
        self,
        base_features: np.ndarray,
        multiplier: float,
        name: Optional[str] = None,
    ) -> Scenario:
        """Cria cenario com aumento/reducao proporcional em todos os canais."""
        allocation: Dict[str, float] = {}
        for ch in self.channel_indices:
            allocation[ch] = float(base_features[self.channel_indices[ch]]) * multiplier

        label: str = name or f"Proporcional x{multiplier:.1f}"
        direction: str = "aumento" if multiplier > 1 else "reducao"
        pct: float = abs(multiplier - 1) * 100

        return self.create_scenario(
            name=label,
            description=f"{direction} proporcional de {pct:.0f}% em todos os canais",
            allocation=allocation,
            base_features=base_features,
        )

    def create_channel_focus_scenario(
        self,
        base_features: np.ndarray,
        focus_channel: str,
        focus_multiplier: float,
        other_multiplier: float = 1.0,
        name: Optional[str] = None,
    ) -> Scenario:
        """Cria cenario com foco em um canal especifico."""
        allocation: Dict[str, float] = {}
        for ch in self.channel_indices:
            if ch == focus_channel:
                allocation[ch] = float(base_features[self.channel_indices[ch]]) * focus_multiplier
            else:
                allocation[ch] = float(base_features[self.channel_indices[ch]]) * other_multiplier

        channel_name: str = focus_channel.replace("_spend", "")
        label: str = name or f"Foco {channel_name} x{focus_multiplier:.1f}"

        return self.create_scenario(
            name=label,
            description=f"Foco em {channel_name} com multiplicador {focus_multiplier:.1f}",
            allocation=allocation,
            base_features=base_features,
        )

    def create_reallocation_scenario(
        self,
        base_features: np.ndarray,
        from_channel: str,
        to_channel: str,
        transfer_pct: float,
        name: Optional[str] = None,
    ) -> Scenario:
        """Cria cenario de realocacao de budget de um canal para outro."""
        allocation: Dict[str, float] = {}
        for ch in self.channel_indices:
            allocation[ch] = float(base_features[self.channel_indices[ch]])

        transfer_amount: float = allocation.get(from_channel, 0) * transfer_pct
        allocation[from_channel] = allocation.get(from_channel, 0) - transfer_amount
        allocation[to_channel] = allocation.get(to_channel, 0) + transfer_amount

        from_name: str = from_channel.replace("_spend", "")
        to_name: str = to_channel.replace("_spend", "")
        label: str = name or f"Realocar {from_name} -> {to_name} ({transfer_pct*100:.0f}%)"

        return self.create_scenario(
            name=label,
            description=f"Transferencia de {transfer_pct*100:.0f}% de {from_name} para {to_name}",
            allocation=allocation,
            base_features=base_features,
        )

    def compare_scenarios(self) -> pd.DataFrame:
        """Compara todos os cenarios criados."""
        if not self.scenarios:
            raise RuntimeError("Nenhum cenario criado.")

        data: List[Dict[str, Any]] = []
        baseline_revenue: float = self.scenarios[0].predicted_revenue if self.scenarios else 0

        for sc in self.scenarios:
            delta_pct: float = (
                (sc.predicted_revenue - baseline_revenue) / abs(baseline_revenue) * 100
                if baseline_revenue != 0
                else 0
            )
            row: Dict[str, Any] = {
                "cenario": sc.name,
                "descricao": sc.description,
                "investimento_total": sc.total_spend,
                "receita_prevista": sc.predicted_revenue,
                "roas": sc.predicted_roas,
                "delta_vs_baseline_pct": delta_pct,
            }
            for ch, val in sc.allocation.items():
                row[ch.replace("_spend", "")] = val
            data.append(row)

        comparison: pd.DataFrame = pd.DataFrame(data)
        logger.info("Comparacao de {} cenarios gerada", len(self.scenarios))
        return comparison

    def sensitivity_analysis(
        self,
        base_features: np.ndarray,
        channel: str,
        multiplier_range: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Analise de sensibilidade para um canal especifico."""
        if multiplier_range is None:
            multiplier_range = np.arange(0.5, 2.1, 0.1)

        results: List[Dict[str, float]] = []
        base_alloc: float = float(base_features[self.channel_indices[channel]])

        for mult in multiplier_range:
            allocation: Dict[str, float] = {}
            for ch in self.channel_indices:
                if ch == channel:
                    allocation[ch] = base_alloc * mult
                else:
                    allocation[ch] = float(base_features[self.channel_indices[ch]])

            revenue: float = self._predict_for_allocation(allocation, base_features)
            results.append({
                "multiplier": float(mult),
                "spend": base_alloc * mult,
                "predicted_revenue": revenue,
            })

        df: pd.DataFrame = pd.DataFrame(results)
        logger.info(
            "Analise de sensibilidade para {}: {} pontos",
            channel,
            len(multiplier_range),
        )
        return df

    def clear_scenarios(self) -> None:
        """Remove todos os cenarios."""
        count: int = len(self.scenarios)
        self.scenarios = []
        logger.info("{} cenarios removidos", count)


# "Planeje o trabalho e trabalhe o plano." - Napoleon Hill
