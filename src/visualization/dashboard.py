"""Dashboard Streamlit para Marketing Mix Modeling."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from src.config import DATA_RAW_DIR, MEDIA_CHANNELS
from src.features.adstock import AdstockTransformer
from src.features.engineering import FeatureEngineer
from src.features.saturation import SaturationTransformer
from src.ingestion.loader import DataLoader
from src.models.importance import FeatureImportanceAnalyzer
from src.models.regularized import RegularizedMMM, compare_models
from src.optimization.budget import BudgetOptimizer
from src.optimization.scenarios import ScenarioPlanner
from src.visualization.charts import MMMCharts


def init_session_state() -> None:
    """Inicializa estado da sessao Streamlit."""
    defaults: Dict[str, Any] = {
        "data_loaded": False,
        "model_trained": False,
        "df_raw": None,
        "df_features": None,
        "model": None,
        "charts": MMMCharts(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> Dict[str, Any]:
    """Renderiza sidebar com configuracoes."""
    st.sidebar.title("Configuracoes MMM")

    config: Dict[str, Any] = {}

    st.sidebar.subheader("Dados")
    data_source: str = st.sidebar.radio(
        "Fonte de dados",
        ["Dados sinteticos", "Upload CSV", "Upload Excel"],
    )
    config["data_source"] = data_source

    st.sidebar.subheader("Modelo")
    config["model_type"] = st.sidebar.selectbox(
        "Tipo de modelo",
        ["ridge", "lasso", "elasticnet"],
    )
    config["alpha"] = st.sidebar.slider("Alpha (regularizacao)", 0.001, 10.0, 1.0, 0.001)
    config["test_size"] = st.sidebar.slider("Tamanho do teste (%)", 10, 40, 20) / 100

    st.sidebar.subheader("Adstock")
    config["adstock_method"] = st.sidebar.selectbox("Metodo adstock", ["geometric", "weibull"])
    config["max_lag"] = st.sidebar.slider("Max lag (semanas)", 4, 24, 12)

    st.sidebar.subheader("Saturacao")
    config["saturation_method"] = st.sidebar.selectbox(
        "Metodo saturacao",
        ["exponential", "hill", "logistic", "power"],
    )

    return config


def render_data_page(config: Dict[str, Any]) -> None:
    """Pagina de carregamento e exploracao dos dados."""
    st.header("Dados de Marketing")

    loader = DataLoader()

    if config["data_source"] == "Dados sinteticos":
        n_weeks: int = st.slider("Numero de semanas", 52, 208, 104)
        if st.button("Gerar dados sinteticos"):
            df: pd.DataFrame = loader.generate_sample_data(n_weeks=n_weeks)
            st.session_state["df_raw"] = df
            st.session_state["data_loaded"] = True
            st.success(f"Dados gerados: {len(df)} semanas")

    elif config["data_source"] == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded, parse_dates=["date"])
            st.session_state["df_raw"] = df
            st.session_state["data_loaded"] = True
            st.success(f"CSV carregado: {len(df)} linhas")

    elif config["data_source"] == "Upload Excel":
        uploaded = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
        if uploaded is not None:
            df = pd.read_excel(uploaded, parse_dates=["date"])
            st.session_state["df_raw"] = df
            st.session_state["data_loaded"] = True
            st.success(f"Excel carregado: {len(df)} linhas")

    if st.session_state["data_loaded"]:
        df = st.session_state["df_raw"]
        st.subheader("Visao geral dos dados")
        st.dataframe(df.head(20))

        col1, col2, col3 = st.columns(3)
        col1.metric("Linhas", len(df))
        col2.metric("Colunas", len(df.columns))
        col3.metric("Periodo", f"{df['date'].min().date()} a {df['date'].max().date()}")

        st.subheader("Estatisticas descritivas")
        st.dataframe(df.describe())


def render_model_page(config: Dict[str, Any]) -> None:
    """Pagina de treinamento e avaliacao do modelo."""
    st.header("Modelagem MMM")

    if not st.session_state["data_loaded"]:
        st.warning("Carregue os dados na aba 'Dados' primeiro.")
        return

    df: pd.DataFrame = st.session_state["df_raw"]

    available_channels: List[str] = [c for c in MEDIA_CHANNELS if c in df.columns]

    if st.button("Treinar modelo"):
        with st.spinner("Processando features..."):
            engineer = FeatureEngineer(media_channels=available_channels)
            adstock = AdstockTransformer(media_channels=available_channels, method=config["adstock_method"])
            saturation = SaturationTransformer(media_channels=available_channels, method=config["saturation_method"])

            df_feat: pd.DataFrame = adstock.transform(df)
            df_feat = saturation.transform(df_feat)
            df_feat = engineer.create_date_features(df_feat)
            df_feat = df_feat.dropna().reset_index(drop=True)

        feature_cols: List[str] = [
            c for c in df_feat.select_dtypes(include=[np.number]).columns
            if c not in ("revenue", "date")
        ]

        with st.spinner("Treinando modelo..."):
            model_kwargs: dict = {"alpha": config["alpha"]}
            if config["model_type"] == "elasticnet":
                model_kwargs["l1_ratio"] = 0.5

            mmm = RegularizedMMM(
                model_type=config["model_type"],
                test_size=config["test_size"],
                **model_kwargs,
            )
            X_train, X_test, y_train, y_test = mmm.prepare_data(df_feat, feature_cols)
            mmm.fit(X_train, y_train)
            metrics: Dict[str, float] = mmm.evaluate(X_test, y_test)

        st.session_state["model"] = mmm
        st.session_state["df_features"] = df_feat
        st.session_state["model_trained"] = True

        st.success("Modelo treinado com sucesso")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R2", f"{metrics['r2']:.4f}")
        col2.metric("RMSE", f"{metrics['rmse']:.0f}")
        col3.metric("MAE", f"{metrics['mae']:.0f}")
        col4.metric("MAPE", f"{metrics['mape']:.2f}%")

        st.subheader("Coeficientes do modelo")
        coefs: pd.DataFrame = mmm.get_coefficients()
        st.dataframe(coefs.head(20))

        charts: MMMCharts = st.session_state["charts"]
        y_pred_all: np.ndarray = mmm.predict(df_feat[feature_cols].values)
        fig = charts.plot_actual_vs_predicted(
            df_feat["date"] if "date" in df_feat.columns else pd.Series(range(len(df_feat))),
            df_feat["revenue"].values,
            y_pred_all,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_optimization_page(config: Dict[str, Any]) -> None:
    """Pagina de otimizacao de budget."""
    st.header("Otimizacao de Budget")

    if not st.session_state["model_trained"]:
        st.warning("Treine o modelo na aba 'Modelo' primeiro.")
        return

    mmm: RegularizedMMM = st.session_state["model"]
    df_feat: pd.DataFrame = st.session_state["df_features"]

    total_budget: float = st.number_input(
        "Orcamento total (R$)",
        min_value=10000.0,
        max_value=10000000.0,
        value=500000.0,
        step=10000.0,
    )

    if st.button("Otimizar alocacao"):
        base_features: np.ndarray = df_feat[mmm.feature_names].mean().values

        optimizer = BudgetOptimizer(
            model=mmm.model,
            scaler=mmm.scaler,
            feature_names=mmm.feature_names,
            media_channels=[c for c in MEDIA_CHANNELS if c in mmm.feature_names],
        )

        result: Dict[str, Any] = optimizer.optimize(
            total_budget=total_budget,
            base_features=base_features,
        )

        st.subheader("Alocacao otima")
        alloc_df: pd.DataFrame = optimizer.get_allocation_summary(result)
        st.dataframe(alloc_df)

        col1, col2 = st.columns(2)
        col1.metric("Receita prevista", f"R$ {result['predicted_revenue']:,.0f}")
        col2.metric("Melhoria", f"{result['improvement_pct']:.2f}%")

        charts: MMMCharts = st.session_state["charts"]
        fig = charts.plot_budget_allocation(result["optimal_allocation"])
        st.plotly_chart(fig, use_container_width=True)


def render_scenarios_page(config: Dict[str, Any]) -> None:
    """Pagina de planejamento de cenarios."""
    st.header("Planejamento de Cenarios")

    if not st.session_state["model_trained"]:
        st.warning("Treine o modelo na aba 'Modelo' primeiro.")
        return

    mmm: RegularizedMMM = st.session_state["model"]
    df_feat: pd.DataFrame = st.session_state["df_features"]
    base_features: np.ndarray = df_feat[mmm.feature_names].mean().values

    available_channels: List[str] = [c for c in MEDIA_CHANNELS if c in mmm.feature_names]

    planner = ScenarioPlanner(
        model=mmm.model,
        scaler=mmm.scaler,
        feature_names=mmm.feature_names,
        media_channels=available_channels,
    )

    planner.create_baseline(base_features)

    st.subheader("Cenarios proporcionais")
    multipliers: List[float] = [0.5, 0.75, 1.25, 1.5, 2.0]
    for mult in multipliers:
        planner.create_proportional_scenario(base_features, mult)

    comparison: pd.DataFrame = planner.compare_scenarios()
    st.dataframe(comparison)

    charts: MMMCharts = st.session_state["charts"]
    fig = charts.plot_scenario_comparison(comparison)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Ponto de entrada do dashboard."""
    st.set_page_config(
        page_title="Marketing Mix Model",
        page_icon="",
        layout="wide",
    )

    st.title("Marketing Mix Modeling")
    st.markdown("---")

    init_session_state()
    config: Dict[str, Any] = render_sidebar()

    tab_data, tab_model, tab_optim, tab_scenarios = st.tabs([
        "Dados",
        "Modelo",
        "Otimizacao",
        "Cenarios",
    ])

    with tab_data:
        render_data_page(config)

    with tab_model:
        render_model_page(config)

    with tab_optim:
        render_optimization_page(config)

    with tab_scenarios:
        render_scenarios_page(config)


if __name__ == "__main__":
    main()


# "Quem nao mede, nao gerencia." - W. Edwards Deming

