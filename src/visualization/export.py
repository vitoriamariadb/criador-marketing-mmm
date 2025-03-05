"""Exportacao de resultados do MMM em diversos formatos."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BASE_DIR, DATA_PROCESSED_DIR


class ResultExporter:
    """Exporta resultados do modelo e analises em formatos consumiveis."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir: Path = output_dir or DATA_PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exported_files: List[Path] = []

    def _generate_filename(self, prefix: str, extension: str) -> Path:
        """Gera nome de arquivo com timestamp."""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename: str = f"{prefix}_{timestamp}.{extension}"
        return self.output_dir / filename

    def export_model_summary(
        self,
        metrics: Dict[str, float],
        coefficients: pd.DataFrame,
        model_type: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Exporta resumo do modelo para JSON."""
        filepath: Path = self.output_dir / filename if filename else self._generate_filename("model_summary", "json")

        summary: Dict[str, Any] = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "coefficients": coefficients.to_dict(orient="records"),
            "n_features": len(coefficients),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        self.exported_files.append(filepath)
        logger.info("Resumo do modelo exportado: {}", filepath)
        return filepath

    def export_contributions(
        self,
        contributions: pd.DataFrame,
        dates: Optional[pd.Series] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Exporta contribuicoes por canal para CSV."""
        filepath: Path = self.output_dir / filename if filename else self._generate_filename("contributions", "csv")

        result: pd.DataFrame = contributions.copy()
        if dates is not None:
            result.insert(0, "date", dates.values)

        result.to_csv(filepath, index=False, encoding="utf-8")
        self.exported_files.append(filepath)
        logger.info("Contribuicoes exportadas: {} ({} linhas)", filepath, len(result))
        return filepath

    def export_roi_analysis(
        self,
        roi_df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> Path:
        """Exporta analise de ROI para CSV."""
        filepath: Path = self.output_dir / filename if filename else self._generate_filename("roi_analysis", "csv")

        roi_df.to_csv(filepath, index=False, encoding="utf-8")
        self.exported_files.append(filepath)
        logger.info("Analise de ROI exportada: {}", filepath)
        return filepath

    def export_optimization_results(
        self,
        optimization: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """Exporta resultados da otimizacao para JSON."""
        filepath: Path = self.output_dir / filename if filename else self._generate_filename("optimization", "json")

        exportable: Dict[str, Any] = {}
        for key, value in optimization.items():
            if isinstance(value, (np.integer, np.floating)):
                exportable[key] = float(value)
            elif isinstance(value, np.ndarray):
                exportable[key] = value.tolist()
            elif isinstance(value, dict):
                exportable[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in value.items()
                }
            else:
                exportable[key] = value

        exportable["timestamp"] = datetime.now().isoformat()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(exportable, f, indent=2, ensure_ascii=False, default=str)

        self.exported_files.append(filepath)
        logger.info("Resultado de otimizacao exportado: {}", filepath)
        return filepath

    def export_scenarios(
        self,
        scenarios_df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> Path:
        """Exporta comparacao de cenarios para Excel."""
        filepath: Path = self.output_dir / filename if filename else self._generate_filename("scenarios", "xlsx")

        scenarios_df.to_excel(filepath, index=False, engine="openpyxl")
        self.exported_files.append(filepath)
        logger.info("Cenarios exportados: {} ({} cenarios)", filepath, len(scenarios_df))
        return filepath

    def export_predictions(
        self,
        dates: pd.Series,
        actual: np.ndarray,
        predicted: np.ndarray,
        filename: Optional[str] = None,
    ) -> Path:
        """Exporta valores reais vs previstos."""
        filepath: Path = self.output_dir / filename if filename else self._generate_filename("predictions", "csv")

        result: pd.DataFrame = pd.DataFrame({
            "date": dates,
            "actual": actual,
            "predicted": predicted,
            "residual": actual - predicted,
            "error_pct": np.abs((actual - predicted) / np.maximum(actual, 1)) * 100,
        })

        result.to_csv(filepath, index=False, encoding="utf-8")
        self.exported_files.append(filepath)
        logger.info("Predicoes exportadas: {} ({} periodos)", filepath, len(result))
        return filepath

    def export_full_report(
        self,
        model_metrics: Dict[str, float],
        coefficients: pd.DataFrame,
        contributions: pd.DataFrame,
        roi_df: pd.DataFrame,
        model_type: str,
        dates: Optional[pd.Series] = None,
    ) -> Dict[str, Path]:
        """Exporta relatorio completo com todos os artefatos."""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir: Path = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        old_dir: Path = self.output_dir
        self.output_dir = report_dir

        files: Dict[str, Path] = {
            "model_summary": self.export_model_summary(model_metrics, coefficients, model_type),
            "contributions": self.export_contributions(contributions, dates),
            "roi_analysis": self.export_roi_analysis(roi_df),
        }

        self.output_dir = old_dir
        logger.info("Relatorio completo exportado em: {} ({} arquivos)", report_dir, len(files))
        return files

    def list_exported_files(self) -> List[Dict[str, Any]]:
        """Lista todos os arquivos exportados."""
        return [
            {
                "path": str(f),
                "name": f.name,
                "size_kb": f.stat().st_size / 1024 if f.exists() else 0,
            }
            for f in self.exported_files
        ]


# "O que e medido melhora. O que e medido e reportado melhora exponencialmente." - Karl Pearson

