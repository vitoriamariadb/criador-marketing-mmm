"""Configuracoes centrais do projeto MMM."""

from pathlib import Path
from typing import Dict, List

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
LOGS_DIR: Path = BASE_DIR / "logs"

DEFAULT_DATE_COLUMN: str = "date"
DEFAULT_TARGET_COLUMN: str = "revenue"

MEDIA_CHANNELS: List[str] = [
    "tv_spend",
    "radio_spend",
    "digital_spend",
    "social_spend",
    "search_spend",
    "print_spend",
]

CONTROL_VARIABLES: List[str] = [
    "seasonality",
    "price_index",
    "competitor_spend",
    "holiday_flag",
]

MODEL_PARAMS: Dict[str, dict] = {
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 0.1},
    "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5},
}

ADSTOCK_DEFAULTS: Dict[str, float] = {
    "tv_spend": 0.7,
    "radio_spend": 0.5,
    "digital_spend": 0.3,
    "social_spend": 0.4,
    "search_spend": 0.2,
    "print_spend": 0.6,
}

SATURATION_DEFAULTS: Dict[str, float] = {
    "tv_spend": 0.0001,
    "radio_spend": 0.0003,
    "digital_spend": 0.0002,
    "social_spend": 0.0004,
    "search_spend": 0.0005,
    "print_spend": 0.0002,
}

LOG_ROTATION: str = "10 MB"
LOG_RETENTION: str = "30 days"
LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"

# "A simplicidade e a sofisticacao suprema." - Leonardo da Vinci
