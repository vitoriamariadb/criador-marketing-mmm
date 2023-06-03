"""Fixtures compartilhadas para testes do MMM."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_marketing_data() -> pd.DataFrame:
    """Gera dados sinteticos de marketing para testes."""
    rng = np.random.default_rng(seed=42)
    n_weeks: int = 52

    dates = pd.date_range(start="2022-01-03", periods=n_weeks, freq="W-MON")

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
    df["revenue"] = (
        500000
        + 1.2 * df["tv_spend"]
        + 0.8 * df["radio_spend"]
        + 1.5 * df["digital_spend"]
        + 1.0 * df["social_spend"]
        + 1.3 * df["search_spend"]
        + 0.5 * df["print_spend"]
        + 50000 * df["seasonality"]
        + 30000 * df["holiday_flag"]
        - 0.3 * df["competitor_spend"]
        + rng.normal(0, 10000, n_weeks)
    )

    return df


@pytest.fixture
def feature_columns() -> list:
    """Lista de colunas de features para testes."""
    return [
        "tv_spend", "radio_spend", "digital_spend",
        "social_spend", "search_spend", "print_spend",
        "price_index", "competitor_spend", "holiday_flag", "seasonality",
    ]


@pytest.fixture
def media_channels() -> list:
    """Lista de canais de midia."""
    return [
        "tv_spend", "radio_spend", "digital_spend",
        "social_spend", "search_spend", "print_spend",
    ]
