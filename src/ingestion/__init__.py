"""Modulo de ingestao de dados."""

from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator

__all__ = ["DataLoader", "DataValidator"]
