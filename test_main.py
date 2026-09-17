"""Unit tests for market-data adapters and return statistics in main.py."""
import numpy as np
import pandas as pd
import pytest

import main


def test_download_stock_data_normalizes_single_ticker_series(monkeypatch):
    index = pd.date_range("2025-01-02", periods=2, freq="B")
    raw = pd.DataFrame({"Close": [100.0, 101.5]}, index=index)
    captured = {}

    def fake_download(tickers, start, end, auto_adjust):
        captured.update(
            tickers=tickers, start=start, end=end, auto_adjust=auto_adjust
        )
        return raw

    monkeypatch.setattr(main.yf, "download", fake_download)

    result = main.download_stock_data("AAPL", "2025-01-01", "2025-02-01")

    assert captured == {
        "tickers": ["AAPL"],
        "start": "2025-01-01",
        "end": "2025-02-01",
        "auto_adjust": True,
    }
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["AAPL"]
    np.testing.assert_allclose(result["AAPL"], [100.0, 101.5])


def test_calculate_annualized_return_uses_requested_period_frequency():
    returns = pd.Series([0.10, -0.05])

    result = main.calculate_annualized_return(returns, periods_per_year=2)

    assert result == pytest.approx(0.045)


def test_calculate_mu_sigma_supports_periodic_and_annualized_results():
    returns = pd.DataFrame({
        "AAA": [0.01, 0.03, -0.01],
        "BBB": [0.02, 0.00, 0.04],
    })
    expected_mu = returns.mean().values
    expected_sigma = returns.cov().values

    mu, sigma = main.calculate_mu_sigma(returns, annualized=False)
    annual_mu, annual_sigma = main.calculate_mu_sigma(
        returns, annualized=True, periods_per_year=12
    )

    np.testing.assert_allclose(mu, expected_mu)
    np.testing.assert_allclose(sigma, expected_sigma)
    np.testing.assert_allclose(annual_mu, expected_mu * 12)
    np.testing.assert_allclose(annual_sigma, expected_sigma * 12)


def test_fetch_fundamentals_maps_fields_and_isolates_ticker_errors(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            if self.symbol == "BAD":
                raise RuntimeError("provider unavailable")
            return {
                "shortName": "Alpha Corp",
                "regularMarketPrice": 123.45,
                "currency": "USD",
                "marketCap": 1_000_000,
                "trailingPE": 20.0,
                "totalRevenue": 500_000,
                "beta": 1.1,
            }

    monkeypatch.setattr(main.yf, "Ticker", FakeTicker)

    result = main.fetch_fundamentals(["AAA", "BAD"])

    assert result["AAA"] == {
        "name": "Alpha Corp",
        "price": 123.45,
        "currency": "USD",
        "market_cap": 1_000_000,
        "pe": 20.0,
        "revenue": 500_000,
        "beta": 1.1,
    }
    assert result["BAD"] == {"error": "provider unavailable"}
