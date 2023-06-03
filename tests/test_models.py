"""Testes para modelos de regressao do MMM."""

import numpy as np
import pandas as pd
import pytest

from src.features.adstock import AdstockTransformer, geometric_adstock
from src.features.saturation import SaturationTransformer, exponential_saturation, hill_saturation
from src.models.cross_validation import HyperparameterSearch, TimeSeriesCV
from src.models.importance import FeatureImportanceAnalyzer
from src.models.regularized import RegularizedMMM, compare_models
from src.models.regression import MMMRegressor


class TestMMMRegressor:
    """Testes para o modelo de regressao linear base."""

    def test_fit_predict(self, sample_marketing_data: pd.DataFrame, feature_columns: list) -> None:
        model = MMMRegressor()
        X_train, X_test, y_train, y_test = model.prepare_data(
            sample_marketing_data, feature_columns
        )
        model.fit(X_train, y_train)

        assert model.is_fitted is True
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_evaluate_metrics(self, sample_marketing_data: pd.DataFrame, feature_columns: list) -> None:
        model = MMMRegressor()
        X_train, X_test, y_train, y_test = model.prepare_data(
            sample_marketing_data, feature_columns
        )
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert metrics["r2"] > 0.5

    def test_coefficients(self, sample_marketing_data: pd.DataFrame, feature_columns: list) -> None:
        model = MMMRegressor()
        X_train, X_test, y_train, y_test = model.prepare_data(
            sample_marketing_data, feature_columns
        )
        model.fit(X_train, y_train)
        coefs = model.get_coefficients()

        assert len(coefs) == len(feature_columns)
        assert "feature" in coefs.columns
        assert "coefficient" in coefs.columns

    def test_not_fitted_raises(self) -> None:
        model = MMMRegressor()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1, 2, 3]]))


class TestRegularizedMMM:
    """Testes para modelos regularizados."""

    @pytest.mark.parametrize("model_type", ["ridge", "lasso", "elasticnet"])
    def test_fit_all_types(
        self, sample_marketing_data: pd.DataFrame, feature_columns: list, model_type: str
    ) -> None:
        mmm = RegularizedMMM(model_type=model_type)
        X_train, X_test, y_train, y_test = mmm.prepare_data(
            sample_marketing_data, feature_columns
        )
        mmm.fit(X_train, y_train)

        assert mmm.is_fitted is True
        metrics = mmm.evaluate(X_test, y_test)
        assert metrics["r2"] > 0

    def test_ridge_regression(self, sample_marketing_data: pd.DataFrame, feature_columns: list) -> None:
        mmm = RegularizedMMM(model_type="ridge", alpha=1.0)
        X_train, X_test, y_train, y_test = mmm.prepare_data(
            sample_marketing_data, feature_columns
        )
        mmm.fit(X_train, y_train)
        metrics = mmm.evaluate(X_test, y_test)

        assert metrics["r2"] > 0.7
        coefs = mmm.get_coefficients()
        assert all(coefs["abs_coefficient"] > 0)

    def test_lasso_sparsity(self, sample_marketing_data: pd.DataFrame, feature_columns: list) -> None:
        mmm = RegularizedMMM(model_type="lasso", alpha=1000.0)
        X_train, X_test, y_train, y_test = mmm.prepare_data(
            sample_marketing_data, feature_columns
        )
        mmm.fit(X_train, y_train)
        coefs = mmm.get_coefficients()

        n_zero = int((~coefs["is_active"]).sum())
        assert n_zero >= 0

    def test_channel_contribution(
        self, sample_marketing_data: pd.DataFrame, feature_columns: list
    ) -> None:
        mmm = RegularizedMMM(model_type="ridge")
        X_train, X_test, y_train, y_test = mmm.prepare_data(
            sample_marketing_data, feature_columns
        )
        mmm.fit(X_train, y_train)
        contributions = mmm.get_channel_contribution(sample_marketing_data, feature_columns)

        assert len(contributions) == len(sample_marketing_data)
        assert "base" in contributions.columns

    def test_summary(self, sample_marketing_data: pd.DataFrame, feature_columns: list) -> None:
        mmm = RegularizedMMM(model_type="ridge")
        X_train, X_test, y_train, y_test = mmm.prepare_data(
            sample_marketing_data, feature_columns
        )
        mmm.fit(X_train, y_train)
        mmm.evaluate(X_test, y_test)
        summary = mmm.summary()

        assert summary["model_type"] == "ridge"
        assert summary["n_features"] == len(feature_columns)

    def test_invalid_model_type(self) -> None:
        with pytest.raises(ValueError):
            RegularizedMMM(model_type="invalid")


class TestCompareModels:
    """Testes para comparacao de modelos."""

    def test_compare_returns_dataframe(
        self, sample_marketing_data: pd.DataFrame, feature_columns: list
    ) -> None:
        result = compare_models(sample_marketing_data, feature_columns)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "model" in result.columns
        assert "r2" in result.columns


class TestAdstock:
    """Testes para transformacao adstock."""

    def test_geometric_adstock_shape(self) -> None:
        series = pd.Series([100, 200, 0, 0, 0], name="test_spend")
        result = geometric_adstock(series, decay_rate=0.5)
        assert len(result) == len(series)

    def test_geometric_adstock_decay(self) -> None:
        series = pd.Series([1000, 0, 0, 0, 0], name="test_spend")
        result = geometric_adstock(series, decay_rate=0.5)
        assert result.iloc[0] == 1000
        assert result.iloc[1] == pytest.approx(500, rel=0.01)
        assert result.iloc[2] == pytest.approx(250, rel=0.01)

    def test_decay_rate_bounds(self) -> None:
        series = pd.Series([100, 200], name="test_spend")
        with pytest.raises(ValueError):
            geometric_adstock(series, decay_rate=1.5)

    def test_adstock_transformer(
        self, sample_marketing_data: pd.DataFrame, media_channels: list
    ) -> None:
        transformer = AdstockTransformer(media_channels=media_channels)
        result = transformer.transform(sample_marketing_data)

        for ch in media_channels:
            assert f"{ch}_adstock" in result.columns


class TestSaturation:
    """Testes para curvas de saturacao."""

    def test_exponential_bounds(self) -> None:
        x = np.array([0, 0.5, 1.0, 10.0, 100.0])
        result = exponential_saturation(x, lambd=0.1)
        assert all(result >= 0)
        assert all(result <= 1)

    def test_hill_saturation_shape(self) -> None:
        x = np.linspace(0, 1, 100)
        result = hill_saturation(x, alpha=2.0, gamma=0.5)
        assert result[0] == pytest.approx(0.0, abs=0.01)

    def test_saturation_transformer(
        self, sample_marketing_data: pd.DataFrame, media_channels: list
    ) -> None:
        transformer = SaturationTransformer(media_channels=media_channels)
        result = transformer.transform(sample_marketing_data)

        for ch in media_channels:
            assert f"{ch}_saturated" in result.columns


class TestTimeSeriesCV:
    """Testes para validacao cruzada temporal."""

    def test_cv_execution(self, sample_marketing_data: pd.DataFrame, feature_columns: list) -> None:
        from sklearn.linear_model import Ridge

        X = sample_marketing_data[feature_columns].values
        y = sample_marketing_data["revenue"].values

        cv = TimeSeriesCV(n_splits=3)
        result = cv.validate(X, y, Ridge, {"alpha": 1.0})

        assert "mean_r2" in result
        assert "std_r2" in result
        assert result["n_folds"] == 3

    def test_cv_results_dataframe(
        self, sample_marketing_data: pd.DataFrame, feature_columns: list
    ) -> None:
        from sklearn.linear_model import Ridge

        X = sample_marketing_data[feature_columns].values
        y = sample_marketing_data["revenue"].values

        cv = TimeSeriesCV(n_splits=3)
        cv.validate(X, y, Ridge)
        results_df = cv.get_results_dataframe()

        assert len(results_df) == 3
        assert "r2" in results_df.columns


class TestFeatureImportance:
    """Testes para analise de importancia de features."""

    def test_coefficient_importance(
        self, sample_marketing_data: pd.DataFrame, feature_columns: list
    ) -> None:
        mmm = RegularizedMMM(model_type="ridge")
        X_train, X_test, y_train, y_test = mmm.prepare_data(
            sample_marketing_data, feature_columns
        )
        mmm.fit(X_train, y_train)

        analyzer = FeatureImportanceAnalyzer(feature_names=feature_columns)
        importance = analyzer.coefficient_importance(mmm.model)

        assert len(importance) == len(feature_columns)
        assert "relative_pct" in importance.columns
        assert importance["relative_pct"].sum() == pytest.approx(100, rel=0.01)


# "Testar e duvidar; nao testar e acreditar cegamente." - Bertrand Russell
