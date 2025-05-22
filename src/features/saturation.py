"""Curvas de saturacao para modelar retornos decrescentes de investimento em midia."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import MEDIA_CHANNELS, SATURATION_DEFAULTS


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Aplica funcao de saturacao Hill (sigmoide).

    A funcao Hill modela retornos decrescentes: y = x^alpha / (x^alpha + gamma^alpha)

    Args:
        x: Valores de investimento (normalizados entre 0 e 1 recomendado).
        alpha: Controla a inclinacao da curva (steepness).
        gamma: Ponto de inflexao (half-saturation point).

    Returns:
        Valores transformados entre 0 e 1.
    """
    if alpha <= 0:
        raise ValueError(f"alpha deve ser positivo, recebido: {alpha}")
    if gamma <= 0:
        raise ValueError(f"gamma deve ser positivo, recebido: {gamma}")

    x_safe: np.ndarray = np.maximum(x, 0)
    return x_safe**alpha / (x_safe**alpha + gamma**alpha)


def exponential_saturation(x: np.ndarray, lambd: float) -> np.ndarray:
    """Aplica funcao de saturacao exponencial negativa.

    Modela retornos decrescentes: y = 1 - exp(-lambda * x)

    Args:
        x: Valores de investimento.
        lambd: Taxa de saturacao (quanto maior, mais rapida a saturacao).

    Returns:
        Valores transformados entre 0 e 1.
    """
    if lambd <= 0:
        raise ValueError(f"lambda deve ser positivo, recebido: {lambd}")

    x_safe: np.ndarray = np.maximum(x, 0)
    return 1.0 - np.exp(-lambd * x_safe)


def logistic_saturation(
    x: np.ndarray, midpoint: float, steepness: float = 1.0
) -> np.ndarray:
    """Aplica funcao logistica de saturacao.

    Curva S classica: y = 1 / (1 + exp(-steepness * (x - midpoint)))

    Args:
        x: Valores de investimento.
        midpoint: Ponto medio da curva (onde y = 0.5).
        steepness: Inclinacao da curva no ponto medio.

    Returns:
        Valores transformados entre 0 e 1.
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def power_saturation(x: np.ndarray, exponent: float) -> np.ndarray:
    """Aplica funcao de potencia para saturacao.

    Modelo simples: y = x^exponent, onde exponent < 1 produz retornos decrescentes.

    Args:
        x: Valores de investimento (devem ser nao-negativos).
        exponent: Expoente de potencia (0 < exponent < 1 para retornos decrescentes).

    Returns:
        Valores transformados.
    """
    if exponent <= 0 or exponent > 1:
        raise ValueError(f"exponent deve estar entre 0 e 1, recebido: {exponent}")

    x_safe: np.ndarray = np.maximum(x, 0)
    return np.power(x_safe, exponent)


class SaturationTransformer:
    """Aplica transformacoes de saturacao em canais de midia."""

    METHODS = {
        "hill": "hill_saturation",
        "exponential": "exponential_saturation",
        "logistic": "logistic_saturation",
        "power": "power_saturation",
    }

    def __init__(
        self,
        saturation_params: Optional[Dict[str, float]] = None,
        media_channels: Optional[List[str]] = None,
        method: str = "exponential",
    ) -> None:
        self.saturation_params: Dict[str, float] = saturation_params or SATURATION_DEFAULTS
        self.media_channels: List[str] = media_channels or MEDIA_CHANNELS
        self.method: str = method
        self._scalers: Dict[str, Dict[str, float]] = {}

    def _normalize(self, series: pd.Series) -> pd.Series:
        """Normaliza serie entre 0 e 1 para aplicacao da curva de saturacao."""
        min_val: float = float(series.min())
        max_val: float = float(series.max())

        if max_val == min_val:
            return pd.Series(np.zeros(len(series)), index=series.index)

        self._scalers[series.name] = {"min": min_val, "max": max_val}
        return (series - min_val) / (max_val - min_val)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica curva de saturacao nos canais de midia."""
        result: pd.DataFrame = df.copy()
        transformed_count: int = 0

        for channel in self.media_channels:
            col_name: str = f"{channel}_adstock" if f"{channel}_adstock" in result.columns else channel

            if col_name not in result.columns:
                logger.warning("Canal nao encontrado: {}", col_name)
                continue

            normalized: pd.Series = self._normalize(result[col_name])
            param: float = self.saturation_params.get(channel, 0.0003)

            if self.method == "exponential":
                saturated: np.ndarray = exponential_saturation(normalized.values, param * 10000)
            elif self.method == "hill":
                saturated = hill_saturation(normalized.values, alpha=2.0, gamma=param * 10000)
            elif self.method == "logistic":
                saturated = logistic_saturation(normalized.values, midpoint=0.5, steepness=param * 50000)
            elif self.method == "power":
                saturated = power_saturation(normalized.values, exponent=min(param * 5000, 0.99))
            else:
                raise ValueError(f"Metodo desconhecido: {self.method}")

            result[f"{channel}_saturated"] = saturated
            transformed_count += 1

        logger.info(
            "Saturacao ({}) aplicada em {} canais", self.method, transformed_count
        )
        return result

    def compute_marginal_returns(
        self, df: pd.DataFrame, channel: str, pct_increase: float = 0.1
    ) -> Dict[str, float]:
        """Calcula retorno marginal para um incremento percentual no investimento."""
        col_name: str = f"{channel}_adstock" if f"{channel}_adstock" in df.columns else channel

        if col_name not in df.columns:
            raise ValueError(f"Canal nao encontrado: {col_name}")

        current: np.ndarray = df[col_name].values
        increased: np.ndarray = current * (1 + pct_increase)

        normalized_current = self._normalize(pd.Series(current, name=col_name))
        param: float = self.saturation_params.get(channel, 0.0003)

        sat_current: np.ndarray = exponential_saturation(
            normalized_current.values, param * 10000
        )
        sat_increased: np.ndarray = exponential_saturation(
            increased / (self._scalers.get(col_name, {}).get("max", 1) or 1),
            param * 10000,
        )

        marginal_return: float = float(np.mean(sat_increased - sat_current))
        efficiency: float = marginal_return / pct_increase if pct_increase > 0 else 0.0

        logger.info(
            "Retorno marginal {}: {:.4f} para +{:.0f}% investimento",
            channel,
            marginal_return,
            pct_increase * 100,
        )
        return {
            "channel": channel,
            "pct_increase": pct_increase,
            "marginal_return": marginal_return,
            "efficiency": efficiency,
        }

    def get_saturation_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula nivel de saturacao atual de cada canal (0 = nada saturado, 1 = totalmente)."""
        levels: Dict[str, float] = {}

        for channel in self.media_channels:
            col: str = f"{channel}_saturated"
            if col in df.columns:
                levels[channel] = float(df[col].mean())

        logger.info("Niveis de saturacao calculados para {} canais", len(levels))
        return levels


# "Metade do dinheiro que gasto em publicidade e desperdicado; o problema e que nao sei qual metade." - John Wanamaker
