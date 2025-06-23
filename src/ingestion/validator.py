"""Validacao de qualidade dos dados de marketing."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DEFAULT_DATE_COLUMN, DEFAULT_TARGET_COLUMN, MEDIA_CHANNELS


class DataValidator:
    """Valida integridade e qualidade dos dados de entrada."""

    def __init__(
        self,
        date_column: str = DEFAULT_DATE_COLUMN,
        target_column: str = DEFAULT_TARGET_COLUMN,
        media_channels: Optional[List[str]] = None,
    ) -> None:
        self.date_column: str = date_column
        self.target_column: str = target_column
        self.media_channels: List[str] = media_channels or MEDIA_CHANNELS

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Executa todas as validacoes e retorna status e lista de problemas."""
        issues: List[str] = []

        issues.extend(self._check_required_columns(df))
        issues.extend(self._check_missing_values(df))
        issues.extend(self._check_negative_values(df))
        issues.extend(self._check_date_continuity(df))
        issues.extend(self._check_outliers(df))

        is_valid: bool = len(issues) == 0

        if is_valid:
            logger.info("Validacao concluida sem problemas")
        else:
            logger.warning("Validacao encontrou {} problemas", len(issues))
            for issue in issues:
                logger.warning("  - {}", issue)

        return is_valid, issues

    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Verifica se todas as colunas obrigatorias estao presentes."""
        issues: List[str] = []
        required: List[str] = [self.date_column, self.target_column] + self.media_channels

        for col in required:
            if col not in df.columns:
                issues.append(f"Coluna obrigatoria ausente: {col}")

        return issues

    def _check_missing_values(self, df: pd.DataFrame) -> List[str]:
        """Verifica valores ausentes nas colunas numericas."""
        issues: List[str] = []
        missing: pd.Series = df.isnull().sum()
        cols_with_missing: pd.Series = missing[missing > 0]

        for col, count in cols_with_missing.items():
            pct: float = count / len(df) * 100
            issues.append(f"Coluna '{col}' tem {count} valores ausentes ({pct:.1f}%)")

        return issues

    def _check_negative_values(self, df: pd.DataFrame) -> List[str]:
        """Verifica valores negativos em colunas de spend e target."""
        issues: List[str] = []
        check_cols: List[str] = [
            c for c in self.media_channels + [self.target_column] if c in df.columns
        ]

        for col in check_cols:
            n_negative: int = int((df[col] < 0).sum())
            if n_negative > 0:
                issues.append(f"Coluna '{col}' tem {n_negative} valores negativos")

        return issues

    def _check_date_continuity(self, df: pd.DataFrame) -> List[str]:
        """Verifica se ha lacunas na serie temporal."""
        issues: List[str] = []

        if self.date_column not in df.columns:
            return issues

        dates: pd.Series = pd.to_datetime(df[self.date_column]).sort_values()
        diffs: pd.Series = dates.diff().dropna()

        if len(diffs) == 0:
            return issues

        median_diff: pd.Timedelta = diffs.median()
        gaps: pd.Series = diffs[diffs > median_diff * 1.5]

        if len(gaps) > 0:
            issues.append(
                f"Encontradas {len(gaps)} lacunas temporais "
                f"(frequencia mediana: {median_diff.days} dias)"
            )

        return issues

    def _check_outliers(self, df: pd.DataFrame) -> List[str]:
        """Detecta outliers usando metodo IQR nas colunas numericas."""
        issues: List[str] = []
        numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            q1: float = df[col].quantile(0.25)
            q3: float = df[col].quantile(0.75)
            iqr: float = q3 - q1
            lower: float = q1 - 3.0 * iqr
            upper: float = q3 + 3.0 * iqr
            n_outliers: int = int(((df[col] < lower) | (df[col] > upper)).sum())

            if n_outliers > 0:
                issues.append(f"Coluna '{col}' tem {n_outliers} outliers extremos (3x IQR)")

        return issues

    def get_summary(self, df: pd.DataFrame) -> Dict[str, object]:
        """Retorna resumo estatistico dos dados."""
        summary: Dict[str, object] = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "date_range": None,
            "missing_pct": df.isnull().mean().to_dict(),
            "numeric_stats": df.describe().to_dict(),
        }

        if self.date_column in df.columns:
            dates = pd.to_datetime(df[self.date_column])
            summary["date_range"] = {
                "start": str(dates.min()),
                "end": str(dates.max()),
                "n_periods": len(dates.unique()),
            }

        logger.info("Resumo dos dados gerado: {} linhas, {} colunas", len(df), len(df.columns))
        return summary


# "Sem dados, voce e apenas mais uma pessoa com uma opiniao." - W. Edwards Deming
