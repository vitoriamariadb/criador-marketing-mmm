"""Testes para a API Flask do MMM."""

import json
from typing import Any, Dict

import pytest
from flask.testing import FlaskClient

from src.api.app import app, model_state


@pytest.fixture
def client() -> FlaskClient:
    """Cria cliente de teste Flask."""
    app.config["TESTING"] = True
    model_state["model"] = None
    model_state["df_features"] = None
    model_state["is_trained"] = False

    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    """Testes para o endpoint de health check."""

    def test_health_returns_200(self, client: FlaskClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_structure(self, client: FlaskClient) -> None:
        response = client.get("/health")
        data: Dict[str, Any] = json.loads(response.data)
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_trained" in data

    def test_health_model_not_trained(self, client: FlaskClient) -> None:
        response = client.get("/health")
        data = json.loads(response.data)
        assert data["model_trained"] is False


class TestTrainEndpoint:
    """Testes para o endpoint de treinamento."""

    def test_train_default_params(self, client: FlaskClient) -> None:
        response = client.post(
            "/train",
            data=json.dumps({"n_weeks": 52}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "metrics" in data

    def test_train_ridge(self, client: FlaskClient) -> None:
        response = client.post(
            "/train",
            data=json.dumps({"model_type": "ridge", "alpha": 1.0, "n_weeks": 52}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["model_type"] == "ridge"
        assert data["metrics"]["r2"] > 0

    def test_train_lasso(self, client: FlaskClient) -> None:
        response = client.post(
            "/train",
            data=json.dumps({"model_type": "lasso", "alpha": 0.1, "n_weeks": 52}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["status"] == "success"

    def test_train_elasticnet(self, client: FlaskClient) -> None:
        response = client.post(
            "/train",
            data=json.dumps({
                "model_type": "elasticnet",
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "n_weeks": 52,
            }),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["status"] == "success"


class TestPredictEndpoint:
    """Testes para o endpoint de predicao."""

    def test_predict_without_model(self, client: FlaskClient) -> None:
        response = client.post(
            "/predict",
            data=json.dumps({"features": {}}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_predict_empty_features(self, client: FlaskClient) -> None:
        client.post(
            "/train",
            data=json.dumps({"n_weeks": 52}),
            content_type="application/json",
        )

        response = client.post(
            "/predict",
            data=json.dumps({"features": {}}),
            content_type="application/json",
        )
        assert response.status_code == 400


class TestCoefficientsEndpoint:
    """Testes para o endpoint de coeficientes."""

    def test_coefficients_without_model(self, client: FlaskClient) -> None:
        response = client.get("/coefficients")
        assert response.status_code == 400

    def test_coefficients_with_model(self, client: FlaskClient) -> None:
        client.post(
            "/train",
            data=json.dumps({"n_weeks": 52}),
            content_type="application/json",
        )

        response = client.get("/coefficients")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "coefficients" in data
        assert "intercept" in data


class TestOptimizeEndpoint:
    """Testes para o endpoint de otimizacao."""

    def test_optimize_without_model(self, client: FlaskClient) -> None:
        response = client.post(
            "/optimize",
            data=json.dumps({"total_budget": 500000}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_optimize_with_model(self, client: FlaskClient) -> None:
        client.post(
            "/train",
            data=json.dumps({"n_weeks": 52}),
            content_type="application/json",
        )

        response = client.post(
            "/optimize",
            data=json.dumps({"total_budget": 500000}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "optimal_allocation" in data
        assert "predicted_revenue" in data


class TestScenariosEndpoint:
    """Testes para o endpoint de cenarios."""

    def test_scenarios_without_model(self, client: FlaskClient) -> None:
        response = client.post(
            "/scenarios",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_scenarios_with_model(self, client: FlaskClient) -> None:
        client.post(
            "/train",
            data=json.dumps({"n_weeks": 52}),
            content_type="application/json",
        )

        response = client.post(
            "/scenarios",
            data=json.dumps({"multipliers": [0.5, 1.0, 1.5]}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0


class TestMetricsEndpoint:
    """Testes para o endpoint de metricas."""

    def test_metrics_without_model(self, client: FlaskClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 400

    def test_metrics_with_model(self, client: FlaskClient) -> None:
        client.post(
            "/train",
            data=json.dumps({"n_weeks": 52}),
            content_type="application/json",
        )

        response = client.get("/metrics")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "metrics" in data
        assert "summary" in data


# "Confie, mas verifique." - Proverbio russo

