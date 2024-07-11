"""Graficos Plotly para analise de Marketing Mix Model."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


class MMMCharts:
    """Gera visualizacoes interativas para resultados do MMM."""

    COLORS: List[str] = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    def __init__(self, template: str = "plotly_white") -> None:
        self.template: str = template

    def plot_actual_vs_predicted(
        self,
        dates: pd.Series,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Real vs Previsto",
    ) -> go.Figure:
        """Grafico de serie temporal comparando valores reais e previstos."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=actual, name="Real",
            line=dict(color=self.COLORS[0], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=predicted, name="Previsto",
            line=dict(color=self.COLORS[1], width=2, dash="dash"),
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Data",
            yaxis_title="Receita",
            template=self.template,
            hovermode="x unified",
        )

        logger.info("Grafico real vs previsto gerado")
        return fig

    def plot_channel_contribution(
        self,
        contributions: pd.DataFrame,
        dates: Optional[pd.Series] = None,
        title: str = "Contribuicao por Canal",
    ) -> go.Figure:
        """Grafico de area empilhada com contribuicao de cada canal."""
        fig = go.Figure()

        x_values = dates if dates is not None else contributions.index
        cols: List[str] = [c for c in contributions.columns if "_pct" not in c]

        for i, col in enumerate(cols):
            fig.add_trace(go.Scatter(
                x=x_values,
                y=contributions[col],
                name=col.replace("_spend", "").replace("_adstock", "").replace("_saturated", ""),
                stackgroup="one",
                line=dict(color=self.COLORS[i % len(self.COLORS)]),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Periodo",
            yaxis_title="Contribuicao",
            template=self.template,
            hovermode="x unified",
        )

        logger.info("Grafico de contribuicao por canal gerado ({} canais)", len(cols))
        return fig

    def plot_waterfall_contribution(
        self,
        channel_totals: Dict[str, float],
        title: str = "Decomposicao Waterfall de Receita",
    ) -> go.Figure:
        """Grafico waterfall mostrando contribuicao total de cada canal."""
        sorted_items = sorted(channel_totals.items(), key=lambda x: abs(x[1]), reverse=True)
        labels: List[str] = [k.replace("_spend", "").replace("base_intercept", "Base") for k, _ in sorted_items]
        values: List[float] = [v for _, v in sorted_items]

        measures: List[str] = ["relative"] * len(values)
        labels.append("Total")
        values.append(sum(values))
        measures.append("total")

        fig = go.Figure(go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            connector=dict(line=dict(color="rgb(63, 63, 63)")),
            increasing=dict(marker=dict(color=self.COLORS[2])),
            decreasing=dict(marker=dict(color=self.COLORS[3])),
            totals=dict(marker=dict(color=self.COLORS[0])),
        ))

        fig.update_layout(
            title=title,
            yaxis_title="Contribuicao (R$)",
            template=self.template,
        )

        logger.info("Grafico waterfall gerado com {} canais", len(channel_totals))
        return fig

    def plot_roi_comparison(
        self,
        roi_df: pd.DataFrame,
        title: str = "Comparacao de ROAS por Canal",
    ) -> go.Figure:
        """Grafico de barras comparando ROAS entre canais."""
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=roi_df["channel"],
            y=roi_df["roas"],
            marker_color=[self.COLORS[i % len(self.COLORS)] for i in range(len(roi_df))],
            text=[f"{v:.2f}x" for v in roi_df["roas"]],
            textposition="outside",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Canal",
            yaxis_title="ROAS",
            template=self.template,
        )

        logger.info("Grafico de ROAS gerado")
        return fig

    def plot_saturation_curves(
        self,
        spend_range: np.ndarray,
        saturated_values: Dict[str, np.ndarray],
        title: str = "Curvas de Saturacao por Canal",
    ) -> go.Figure:
        """Plota curvas de saturacao para multiplos canais."""
        fig = go.Figure()

        for i, (channel, values) in enumerate(saturated_values.items()):
            fig.add_trace(go.Scatter(
                x=spend_range,
                y=values,
                name=channel.replace("_spend", ""),
                line=dict(color=self.COLORS[i % len(self.COLORS)], width=2),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Investimento",
            yaxis_title="Efeito Saturado",
            template=self.template,
            hovermode="x unified",
        )

        logger.info("Curvas de saturacao plotadas para {} canais", len(saturated_values))
        return fig

    def plot_budget_allocation(
        self,
        allocation: Dict[str, float],
        title: str = "Alocacao Otima de Budget",
    ) -> go.Figure:
        """Grafico de pizza com alocacao de budget."""
        labels: List[str] = [k.replace("_spend", "") for k in allocation.keys()]
        values: List[float] = list(allocation.values())

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=self.COLORS[:len(labels)]),
            textinfo="label+percent",
        ))

        fig.update_layout(
            title=title,
            template=self.template,
        )

        logger.info("Grafico de alocacao de budget gerado")
        return fig

    def plot_scenario_comparison(
        self,
        scenarios_df: pd.DataFrame,
        title: str = "Comparacao de Cenarios",
    ) -> go.Figure:
        """Grafico de barras agrupadas comparando cenarios."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Receita Prevista", "ROAS"),
        )

        fig.add_trace(go.Bar(
            x=scenarios_df["cenario"],
            y=scenarios_df["receita_prevista"],
            name="Receita",
            marker_color=self.COLORS[0],
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=scenarios_df["cenario"],
            y=scenarios_df["roas"],
            name="ROAS",
            marker_color=self.COLORS[1],
        ), row=1, col=2)

        fig.update_layout(
            title_text=title,
            template=self.template,
            showlegend=False,
        )

        logger.info("Grafico de comparacao de cenarios gerado ({} cenarios)", len(scenarios_df))
        return fig

    def plot_model_metrics(
        self,
        cv_results: pd.DataFrame,
        title: str = "Metricas por Fold de Validacao Cruzada",
    ) -> go.Figure:
        """Plota metricas de validacao cruzada por fold."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("R2", "RMSE", "MAE", "MAPE (%)"),
        )

        metrics_config = [
            ("r2", 1, 1), ("rmse", 1, 2),
            ("mae", 2, 1), ("mape", 2, 2),
        ]

        for metric, row, col in metrics_config:
            if metric in cv_results.columns:
                fig.add_trace(go.Bar(
                    x=cv_results["fold"],
                    y=cv_results[metric],
                    name=metric.upper(),
                    marker_color=self.COLORS[row * 2 + col - 3],
                ), row=row, col=col)

        fig.update_layout(
            title_text=title,
            template=self.template,
            showlegend=False,
            height=600,
        )

        logger.info("Grafico de metricas de CV gerado")
        return fig

    def plot_sensitivity(
        self,
        sensitivity_df: pd.DataFrame,
        channel: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Plota curva de sensibilidade para um canal."""
        channel_name: str = channel.replace("_spend", "")
        plot_title: str = title or f"Analise de Sensibilidade: {channel_name}"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sensitivity_df["spend"],
            y=sensitivity_df["predicted_revenue"],
            mode="lines+markers",
            line=dict(color=self.COLORS[0], width=2),
            marker=dict(size=6),
        ))

        fig.update_layout(
            title=plot_title,
            xaxis_title=f"Investimento em {channel_name}",
            yaxis_title="Receita Prevista",
            template=self.template,
        )

        logger.info("Grafico de sensibilidade gerado para {}", channel)
        return fig


# "Um bom grafico vale mais que mil tabelas." - Edward Tufte (adaptado)

