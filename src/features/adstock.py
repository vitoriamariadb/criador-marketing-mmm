"""Transformacao de adstock para modelagem de efeitos carry-over de midia."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import ADSTOCK_DEFAULTS, MEDIA_CHANNELS


def geometric_adstock(series: pd.Series, decay_rate: float, max_lag: int = 12) -> pd.Series:
    """Aplica adstock geometrico a uma serie temporal.

    O adstock geometrico modela o efeito residual da publicidade ao longo do tempo.
    Em cada periodo, o efeito acumulado e: adstock_t = x_t + decay * adstock_{t-1}

    Args:
        series: Serie temporal de investimento em midia.
        decay_rate: Taxa de decaimento entre 0 e 1 (quanto maior, mais duradouro o efeito).
        max_lag: Numero maximo de periodos de carry-over.

    Returns:
        Serie com transformacao adstock aplicada.
    """
    if not 0 <= decay_rate <= 1:
        raise ValueError(f"decay_rate deve estar entre 0 e 1, recebido: {decay_rate}")

    values: np.ndarray = series.values.astype(float)
    adstocked: np.ndarray = np.zeros_like(values)
    adstocked[0] = values[0]

    for t in range(1, len(values)):
        adstocked[t] = values[t] + decay_rate * adstocked[t - 1]

    return pd.Series(adstocked, index=series.index, name=f"{series.name}_adstock")


def weibull_adstock(
    series: pd.Series, shape: float, scale: float, max_lag: int = 12
) -> pd.Series:
    """Aplica adstock Weibull que permite formas flexiveis de decaimento.

    A distribuicao Weibull permite modelar efeitos que podem ter pico
    apos o investimento (delayed effect), nao apenas decaimento imediato.

    Args:
        series: Serie temporal de investimento.
        shape: Parametro de forma (k > 1 = pico atrasado, k < 1 = decaimento rapido).
        scale: Parametro de escala (controla velocidade do decaimento).
        max_lag: Numero maximo de periodos de carry-over.

    Returns:
        Serie com transformacao adstock Weibull aplicada.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError(f"shape e scale devem ser positivos: shape={shape}, scale={scale}")

    values: np.ndarray = series.values.astype(float)
    n: int = len(values)

    lags: np.ndarray = np.arange(max_lag + 1)
    weights: np.ndarray = (shape / scale) * (lags / scale) ** (shape - 1) * np.exp(
        -((lags / scale) ** shape)
    )
    weights = weights / weights.sum()

    adstocked: np.ndarray = np.zeros(n)
    for t in range(n):
        for lag in range(min(max_lag + 1, t + 1)):
            adstocked[t] += weights[lag] * values[t - lag]

    return pd.Series(adstocked, index=series.index, name=f"{series.name}_weibull")


class AdstockTransformer:
    """Aplica transformacoes de adstock em multiplos canais de midia."""

    def __init__(
        self,
        decay_rates: Optional[Dict[str, float]] = None,
        media_channels: Optional[List[str]] = None,
        method: str = "geometric",
        max_lag: int = 12,
    ) -> None:
        self.decay_rates: Dict[str, float] = decay_rates or ADSTOCK_DEFAULTS
        self.media_channels: List[str] = media_channels or MEDIA_CHANNELS
        self.method: str = method
        self.max_lag: int = max_lag

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica adstock em todos os canais de midia configurados."""
        result: pd.DataFrame = df.copy()
        transformed_count: int = 0

        for channel in self.media_channels:
            if channel not in result.columns:
                logger.warning("Canal nao encontrado no DataFrame: {}", channel)
                continue

            decay: float = self.decay_rates.get(channel, 0.5)

            if self.method == "geometric":
                adstocked: pd.Series = geometric_adstock(
                    result[channel], decay, self.max_lag
                )
            elif self.method == "weibull":
                adstocked = weibull_adstock(
                    result[channel], shape=2.0, scale=decay * 10, max_lag=self.max_lag
                )
            else:
                raise ValueError(f"Metodo de adstock desconhecido: {self.method}")

            result[f"{channel}_adstock"] = adstocked
            transformed_count += 1

        logger.info(
            "Adstock ({}) aplicado em {} canais (max_lag={})",
            self.method,
            transformed_count,
            self.max_lag,
        )
        return result

    def find_optimal_decay(
        self,
        df: pd.DataFrame,
        target_column: str,
        channel: str,
        decay_range: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Encontra taxa de decaimento otima via grid search baseado em correlacao."""
        if decay_range is None:
            decay_range = np.arange(0.1, 1.0, 0.05)

        best_decay: float = 0.5
        best_corr: float = 0.0

        for decay in decay_range:
            adstocked: pd.Series = geometric_adstock(df[channel], decay, self.max_lag)
            corr: float = abs(adstocked.corr(df[target_column]))

            if corr > best_corr:
                best_corr = corr
                best_decay = float(decay)

        logger.info(
            "Decay otimo para {}: {:.2f} (correlacao: {:.4f})",
            channel,
            best_decay,
            best_corr,
        )
        return {"channel": channel, "optimal_decay": best_decay, "correlation": best_corr}

    def optimize_all_channels(
        self, df: pd.DataFrame, target_column: str
    ) -> Dict[str, float]:
        """Otimiza decay para todos os canais."""
        optimal_decays: Dict[str, float] = {}

        for channel in self.media_channels:
            if channel not in df.columns:
                continue
            result: Dict[str, float] = self.find_optimal_decay(df, target_column, channel)
            optimal_decays[channel] = result["optimal_decay"]

        self.decay_rates = optimal_decays
        logger.info("Decays otimizados para {} canais", len(optimal_decays))
        return optimal_decays


# "O efeito residual da publicidade e como o eco numa montanha." - David Ogilvy

