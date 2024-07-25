"""API Flask para servir modelos de Marketing Mix."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from loguru import logger

from src.config import MEDIA_CHANNELS
from src.features.adstock import AdstockTransformer
from src.features.engineering import FeatureEngineer
from src.features.saturation import SaturationTransformer
from src.ingestion.loader import DataLoader
from src.models.regularized import RegularizedMMM
from src.optimization.budget import BudgetOptimizer
from src.optimization.scenarios import ScenarioPlanner


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

model_state: Dict[str, Any] = {
    "model": None,
    "df_features": None,
    "is_trained": False,
}


@app.route("/health", methods=["GET"])
def health_check() -> tuple:
    """Verifica status da API."""
    return jsonify({
        "status": "healthy",
        "model_trained": model_state["is_trained"],
    }), 200


@app.route("/train", methods=["POST"])
def train_model() -> tuple:
    """Treina o modelo MMM com dados fornecidos ou sinteticos."""
    params: Dict[str, Any] = request.get_json(silent=True) or {}

    model_type: str = params.get("model_type", "ridge")
    alpha: float = params.get("alpha", 1.0)
    test_size: float = params.get("test_size", 0.2)
    n_weeks: int = params.get("n_weeks", 104)

    try:
        loader = DataLoader()
        df: pd.DataFrame = loader.generate_sample_data(n_weeks=n_weeks)

        available_channels = [c for c in MEDIA_CHANNELS if c in df.columns]

        adstock = AdstockTransformer(media_channels=available_channels)
        saturation = SaturationTransformer(media_channels=available_channels)
        engineer = FeatureEngineer(media_channels=available_channels)

        df_feat: pd.DataFrame = adstock.transform(df)
        df_feat = saturation.transform(df_feat)
        df_feat = engineer.create_date_features(df_feat)
        df_feat = df_feat.dropna().reset_index(drop=True)

        feature_cols = [
            c for c in df_feat.select_dtypes(include=[np.number]).columns
            if c not in ("revenue", "date")
        ]

        model_kwargs: dict = {"alpha": alpha}
        if model_type == "elasticnet":
            model_kwargs["l1_ratio"] = params.get("l1_ratio", 0.5)

        mmm = RegularizedMMM(model_type=model_type, test_size=test_size, **model_kwargs)
        X_train, X_test, y_train, y_test = mmm.prepare_data(df_feat, feature_cols)
        mmm.fit(X_train, y_train)
        metrics: Dict[str, float] = mmm.evaluate(X_test, y_test)

        model_state["model"] = mmm
        model_state["df_features"] = df_feat
        model_state["is_trained"] = True

        logger.info("Modelo treinado via API: {}", model_type)
        return jsonify({
            "status": "success",
            "model_type": model_type,
            "metrics": metrics,
            "n_features": len(feature_cols),
            "n_samples": len(df_feat),
        }), 200

    except Exception as e:
        logger.error("Erro ao treinar modelo: {}", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    """Gera predicoes com o modelo treinado."""
    if not model_state["is_trained"]:
        return jsonify({"status": "error", "message": "Modelo nao treinado"}), 400

    data: Dict[str, Any] = request.get_json(silent=True) or {}

    try:
        mmm: RegularizedMMM = model_state["model"]
        features: Dict[str, float] = data.get("features", {})

        if not features:
            return jsonify({"status": "error", "message": "Features nao fornecidas"}), 400

        feature_values = np.array([[features.get(f, 0) for f in mmm.feature_names]])
        prediction: float = float(mmm.predict(feature_values)[0])

        return jsonify({
            "status": "success",
            "predicted_revenue": prediction,
        }), 200

    except Exception as e:
        logger.error("Erro na predicao: {}", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/coefficients", methods=["GET"])
def get_coefficients() -> tuple:
    """Retorna coeficientes do modelo."""
    if not model_state["is_trained"]:
        return jsonify({"status": "error", "message": "Modelo nao treinado"}), 400

    mmm: RegularizedMMM = model_state["model"]
    coefs: pd.DataFrame = mmm.get_coefficients()

    return jsonify({
        "status": "success",
        "coefficients": coefs.to_dict(orient="records"),
        "intercept": float(mmm.model.intercept_),
    }), 200


@app.route("/optimize", methods=["POST"])
def optimize_budget() -> tuple:
    """Otimiza alocacao de budget."""
    if not model_state["is_trained"]:
        return jsonify({"status": "error", "message": "Modelo nao treinado"}), 400

    data: Dict[str, Any] = request.get_json(silent=True) or {}
    total_budget: float = data.get("total_budget", 500000)

    try:
        mmm: RegularizedMMM = model_state["model"]
        df_feat: pd.DataFrame = model_state["df_features"]
        base_features: np.ndarray = df_feat[mmm.feature_names].mean().values

        media_in_features = [c for c in MEDIA_CHANNELS if c in mmm.feature_names]
        optimizer = BudgetOptimizer(
            model=mmm.model,
            scaler=mmm.scaler,
            feature_names=mmm.feature_names,
            media_channels=media_in_features,
        )

        result: Dict[str, Any] = optimizer.optimize(
            total_budget=total_budget,
            base_features=base_features,
        )

        serializable: Dict[str, Any] = {
            "status": "success",
            "total_budget": total_budget,
            "optimal_allocation": {k: float(v) for k, v in result["optimal_allocation"].items()},
            "predicted_revenue": float(result["predicted_revenue"]),
            "improvement_pct": float(result["improvement_pct"]),
        }

        return jsonify(serializable), 200

    except Exception as e:
        logger.error("Erro na otimizacao: {}", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/scenarios", methods=["POST"])
def compare_scenarios() -> tuple:
    """Cria e compara cenarios de alocacao."""
    if not model_state["is_trained"]:
        return jsonify({"status": "error", "message": "Modelo nao treinado"}), 400

    data: Dict[str, Any] = request.get_json(silent=True) or {}
    multipliers: list = data.get("multipliers", [0.5, 0.75, 1.0, 1.25, 1.5])

    try:
        mmm: RegularizedMMM = model_state["model"]
        df_feat: pd.DataFrame = model_state["df_features"]
        base_features: np.ndarray = df_feat[mmm.feature_names].mean().values

        media_in_features = [c for c in MEDIA_CHANNELS if c in mmm.feature_names]
        planner = ScenarioPlanner(
            model=mmm.model,
            scaler=mmm.scaler,
            feature_names=mmm.feature_names,
            media_channels=media_in_features,
        )

        planner.create_baseline(base_features)
        for mult in multipliers:
            if mult != 1.0:
                planner.create_proportional_scenario(base_features, mult)

        comparison: pd.DataFrame = planner.compare_scenarios()

        return jsonify({
            "status": "success",
            "scenarios": json.loads(comparison.to_json(orient="records")),
        }), 200

    except Exception as e:
        logger.error("Erro nos cenarios: {}", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def get_metrics() -> tuple:
    """Retorna metricas do modelo treinado."""
    if not model_state["is_trained"]:
        return jsonify({"status": "error", "message": "Modelo nao treinado"}), 400

    mmm: RegularizedMMM = model_state["model"]
    return jsonify({
        "status": "success",
        "metrics": mmm.metrics,
        "summary": mmm.summary(),
    }), 200


def create_app() -> Flask:
    """Factory function para criar a aplicacao Flask."""
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# "Uma interface e um contrato entre quem serve e quem consome." - Robert C. Martin

