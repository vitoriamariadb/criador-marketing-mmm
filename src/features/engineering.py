"""Engenharia de features para dados de marketing."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import CONTROL_VARIABLES, DEFAULT_DATE_COLUMN, MEDIA_CHANNELS


class FeatureEngineer:
    """Responsavel pela criacao e transformacao de features de marketing."""

    def __init__(
        self,
        media_channels: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
        date_column: str = DEFAULT_DATE_COLUMN,
    ) -> None:
        self.media_channels: List[str] = media_channels or MEDIA_CHANNELS
        self.control_variables: List[str] = control_variables or CONTROL_VARIABLES
        self.date_column: str = date_column

    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais a partir da coluna de data."""
        result: pd.DataFrame = df.copy()
        dates: pd.Series = pd.to_datetime(result[self.date_column])

        result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        result["month"] = dates.dt.month
        result["quarter"] = dates.dt.quarter
        result["year"] = dates.dt.year

        result["sin_week"] = np.sin(2 * np.pi * result["week_of_year"] / 52)
        result["cos_week"] = np.cos(2 * np.pi * result["week_of_year"] / 52)
        result["sin_month"] = np.sin(2 * np.pi * result["month"] / 12)
        result["cos_month"] = np.cos(2 * np.pi * result["month"] / 12)

        logger.info("Features temporais criadas: 8 novas colunas")
        return result

    def create_lag_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None, lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Cria features de lag para colunas especificadas."""
        result: pd.DataFrame = df.copy()
        cols: List[str] = columns or self.media_channels
        lag_values: List[int] = lags or [1, 2, 4]

        for col in cols:
            if col not in result.columns:
                continue
            for lag in lag_values:
                result[f"{col}_lag{lag}"] = result[col].shift(lag)

        n_new: int = len([c for c in cols if c in result.columns]) * len(lag_values)
        logger.info("Features de lag criadas: {} novas colunas", n_new)
        return result

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Cria medias moveis e desvios padrao."""
        result: pd.DataFrame = df.copy()
        cols: List[str] = columns or self.media_channels
        win_sizes: List[int] = windows or [4, 8, 13]

        for col in cols:
            if col not in result.columns:
                continue
            for w in win_sizes:
                result[f"{col}_ma{w}"] = result[col].rolling(window=w, min_periods=1).mean()
                result[f"{col}_std{w}"] = result[col].rolling(window=w, min_periods=1).std()

        n_new = len([c for c in cols if c in result.columns]) * len(win_sizes) * 2
        logger.info("Features rolling criadas: {} novas colunas", n_new)
        return result

    def create_interaction_features(
        self, df: pd.DataFrame, pairs: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """Cria features de interacao entre canais de midia."""
        result: pd.DataFrame = df.copy()

        if pairs is None:
            available: List[str] = [c for c in self.media_channels if c in result.columns]
            pairs = []
            for i in range(len(available)):
                for j in range(i + 1, min(i + 3, len(available))):
                    pairs.append((available[i], available[j]))

        for col_a, col_b in pairs:
            if col_a in result.columns and col_b in result.columns:
                name_a: str = col_a.replace("_spend", "")
                name_b: str = col_b.replace("_spend", "")
                result[f"interact_{name_a}_{name_b}"] = result[col_a] * result[col_b]

        logger.info("Features de interacao criadas: {} pares", len(pairs))
        return result

    def create_share_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de share of spend por canal."""
        result: pd.DataFrame = df.copy()
        available: List[str] = [c for c in self.media_channels if c in result.columns]

        if not available:
            logger.warning("Nenhum canal de midia encontrado para calcular share")
            return result

        total_spend: pd.Series = result[available].sum(axis=1)
        total_spend = total_spend.replace(0, np.nan)

        for col in available:
            name: str = col.replace("_spend", "")
            result[f"share_{name}"] = result[col] / total_spend

        logger.info("Features de share criadas: {} colunas", len(available))
        return result

    def apply_log_transform(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Aplica transformacao log(1+x) nas colunas especificadas."""
        result: pd.DataFrame = df.copy()
        cols: List[str] = columns or self.media_channels

        for col in cols:
            if col in result.columns:
                result[f"{col}_log"] = np.log1p(result[col].clip(lower=0))

        logger.info("Transformacao log aplicada em {} colunas", len(cols))
        return result

    def prepare_features(
        self,
        df: pd.DataFrame,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_interactions: bool = False,
        include_shares: bool = True,
        include_log: bool = False,
    ) -> pd.DataFrame:
        """Pipeline completo de engenharia de features."""
        result: pd.DataFrame = self.create_date_features(df)

        if include_lags:
            result = self.create_lag_features(result)
        if include_rolling:
            result = self.create_rolling_features(result)
        if include_interactions:
            result = self.create_interaction_features(result)
        if include_shares:
            result = self.create_share_features(result)
        if include_log:
            result = self.apply_log_transform(result)

        result = result.dropna().reset_index(drop=True)
        logger.info(
            "Pipeline de features concluido: {} linhas, {} colunas",
            len(result),
            len(result.columns),
        )
        return result

    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Retorna dicionario com nomes de features por categoria."""
        all_cols: List[str] = df.columns.tolist()

        return {
            "media": [c for c in all_cols if any(m in c for m in self.media_channels)],
            "control": [c for c in all_cols if c in self.control_variables],
            "temporal": [
                c for c in all_cols
                if c in ("week_of_year", "month", "quarter", "year",
                         "sin_week", "cos_week", "sin_month", "cos_month")
            ],
            "share": [c for c in all_cols if c.startswith("share_")],
            "interaction": [c for c in all_cols if c.startswith("interact_")],
        }


# "Tortura os dados por tempo suficiente e eles confessarao qualquer coisa." - Ronald Coase

