"""Carregamento de dados de marketing a partir de CSV e Excel."""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.config import DATA_RAW_DIR, DEFAULT_DATE_COLUMN


class DataLoader:
    """Responsavel por carregar e preparar dados brutos de marketing."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir: Path = data_dir or DATA_RAW_DIR

    def load_csv(self, filename: str, date_column: str = DEFAULT_DATE_COLUMN) -> pd.DataFrame:
        """Carrega dados de um arquivo CSV."""
        filepath: Path = self.data_dir / filename
        if not filepath.exists():
            logger.error("Arquivo nao encontrado: {}", filepath)
            raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

        logger.info("Carregando CSV: {}", filepath)
        df: pd.DataFrame = pd.read_csv(filepath, parse_dates=[date_column])
        df = df.sort_values(by=date_column).reset_index(drop=True)
        logger.info("CSV carregado com {} linhas e {} colunas", len(df), len(df.columns))
        return df

    def load_excel(
        self,
        filename: str,
        sheet_name: str = "Sheet1",
        date_column: str = DEFAULT_DATE_COLUMN,
    ) -> pd.DataFrame:
        """Carrega dados de um arquivo Excel."""
        filepath: Path = self.data_dir / filename
        if not filepath.exists():
            logger.error("Arquivo nao encontrado: {}", filepath)
            raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

        logger.info("Carregando Excel: {} (aba: {})", filepath, sheet_name)
        df: pd.DataFrame = pd.read_excel(
            filepath, sheet_name=sheet_name, parse_dates=[date_column]
        )
        df = df.sort_values(by=date_column).reset_index(drop=True)
        logger.info("Excel carregado com {} linhas e {} colunas", len(df), len(df.columns))
        return df

    def load_multiple_sources(
        self, filenames: list[str], date_column: str = DEFAULT_DATE_COLUMN
    ) -> pd.DataFrame:
        """Carrega e concatena multiplas fontes de dados."""
        frames: list[pd.DataFrame] = []

        for filename in filenames:
            suffix: str = Path(filename).suffix.lower()
            if suffix == ".csv":
                frames.append(self.load_csv(filename, date_column))
            elif suffix in (".xlsx", ".xls"):
                frames.append(self.load_excel(filename, date_column=date_column))
            else:
                logger.warning("Formato nao suportado: {}", suffix)

        if not frames:
            logger.error("Nenhum dado carregado de {} fontes", len(filenames))
            raise ValueError("Nenhum dado foi carregado")

        combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(by=date_column).reset_index(drop=True)
        logger.info("Total combinado: {} linhas de {} fontes", len(combined), len(frames))
        return combined

    def generate_sample_data(self, n_weeks: int = 104) -> pd.DataFrame:
        """Gera dados sinteticos para demonstracao e testes."""
        import numpy as np

        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(start="2021-01-04", periods=n_weeks, freq="W-MON")

        data: dict = {
            "date": dates,
            "tv_spend": rng.uniform(10000, 80000, n_weeks),
            "radio_spend": rng.uniform(5000, 30000, n_weeks),
            "digital_spend": rng.uniform(15000, 60000, n_weeks),
            "social_spend": rng.uniform(8000, 40000, n_weeks),
            "search_spend": rng.uniform(10000, 50000, n_weeks),
            "print_spend": rng.uniform(3000, 15000, n_weeks),
            "price_index": rng.uniform(0.9, 1.1, n_weeks),
            "competitor_spend": rng.uniform(50000, 150000, n_weeks),
            "holiday_flag": rng.choice([0, 1], n_weeks, p=[0.85, 0.15]),
        }

        df = pd.DataFrame(data)
        df["seasonality"] = np.sin(2 * np.pi * df.index / 52)

        base_revenue = 500000
        df["revenue"] = (
            base_revenue
            + 1.2 * df["tv_spend"]
            + 0.8 * df["radio_spend"]
            + 1.5 * df["digital_spend"]
            + 1.0 * df["social_spend"]
            + 1.3 * df["search_spend"]
            + 0.5 * df["print_spend"]
            + 50000 * df["seasonality"]
            + 30000 * df["holiday_flag"]
            - 0.3 * df["competitor_spend"]
            + rng.normal(0, 15000, n_weeks)
        )

        logger.info("Dados sinteticos gerados: {} semanas", n_weeks)
        return df


# "O que pode ser medido pode ser melhorado." - Peter Drucker

