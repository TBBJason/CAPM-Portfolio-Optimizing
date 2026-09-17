"""API tests for backend.py using a mocked price download (no network)."""
import numpy as np
import pandas as pd
import pytest

import backend


def _synthetic_prices(columns, periods=4 * 252, seed=0):
    """Deterministic geometric random walk ending today."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=periods, freq="B")
    data = {}
    for i, col in enumerate(columns):
        daily = rng.normal(0.0004 + i * 0.0001, 0.01, size=periods)
        data[col] = 100 * np.cumprod(1 + daily)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def client():
    backend.app.config["TESTING"] = True
    return backend.app.test_client()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_optimize_requires_two_tickers(client):
    resp = client.post("/api/optimize", json={"tickers": ["AAPL"]})
    assert resp.status_code == 400


def test_optimize_returns_weights_labelled_by_column_order(client, monkeypatch):
    # yfinance returns columns alphabetically; the user submits a different
    # order. The response must label weights by the actual data columns.
    cols = ["AAPL", "GOOG", "MSFT"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols),
    )

    resp = client.post("/api/optimize", json={
        "tickers": ["MSFT", "AAPL", "GOOG"],  # deliberately not alphabetical
        "rf": 0.04,
        "lookback_years": 3,
    })
    assert resp.status_code == 200
    body = resp.get_json()

    assert set(body["weights"].keys()) == set(cols)
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    # All long-only weights are non-negative.
    assert all(w >= -1e-9 for w in body["weights"].values())
    # Backtest series starts at 1.0 and has more than one point (held-out year).
    assert body["backtest_values"][0] == pytest.approx(1.0)
    assert len(body["backtest_values"]) > 1
    for key in ("expected_return", "volatility", "sharpe"):
        assert key in body


def test_optimize_reports_dropped_tickers(client, monkeypatch):
    # Mock returns data for only two of the three requested tickers.
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(["AAPL", "MSFT"]),
    )
    resp = client.post("/api/optimize", json={
        "tickers": ["AAPL", "MSFT", "FAKETICKER"],
        "lookback_years": 2,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert "warning" in body
    assert "FAKETICKER" in body["warning"]


def test_optimize_allow_shorting_can_go_negative(client, monkeypatch):
    cols = ["AAA", "BBB"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols, seed=7),
    )
    resp = client.post("/api/optimize", json={
        "tickers": cols,
        "allow_shorting": True,
        "lookback_years": 2,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    # Weights still sum to 1 under shorting.
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def _assert_valid_frontier(fr, expected_tickers):
    assert set(fr.keys()) >= {"rf", "frontier", "tangency", "assets"}
    assert len(fr["frontier"]) > 1
    for pt in fr["frontier"]:
        assert pt["volatility"] >= 0
    # Tangency point is well-formed.
    assert fr["tangency"]["volatility"] > 0
    assert "sharpe" in fr["tangency"]
    # One asset point per ticker, labelled correctly.
    assert {a["ticker"] for a in fr["assets"]} == set(expected_tickers)


def test_optimize_includes_frontier(client, monkeypatch):
    cols = ["AAPL", "GOOG", "MSFT"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols),
    )
    resp = client.post("/api/optimize", json={
        "tickers": ["MSFT", "AAPL", "GOOG"],
        "lookback_years": 3,
    })
    assert resp.status_code == 200
    _assert_valid_frontier(resp.get_json()["frontier"], cols)


def test_frontier_endpoint(client, monkeypatch):
    cols = ["AAPL", "GOOG", "MSFT"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols),
    )
    resp = client.post("/api/frontier", json={
        "tickers": ["MSFT", "AAPL", "GOOG"],
        "rf": 0.04,
        "lookback_years": 3,
    })
    assert resp.status_code == 200
    _assert_valid_frontier(resp.get_json(), cols)


def test_frontier_endpoint_requires_two_tickers(client):
    resp = client.post("/api/frontier", json={"tickers": ["AAPL"]})
    assert resp.status_code == 400


def test_frontier_endpoint_get_is_ok(client):
    resp = client.get("/api/frontier")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_shrinkage_on_by_default_reports_intensity(client, monkeypatch):
    cols = ["AAPL", "GOOG", "MSFT"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols),
    )
    resp = client.post("/api/optimize", json={
        "tickers": cols, "lookback_years": 3,
    })
    assert resp.status_code == 200
    delta = resp.get_json()["shrinkage_intensity"]
    assert delta is not None
    assert 0.0 <= delta <= 1.0


def test_shrinkage_can_be_disabled(client, monkeypatch):
    cols = ["AAPL", "GOOG", "MSFT"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols),
    )
    resp = client.post("/api/optimize", json={
        "tickers": cols, "lookback_years": 3, "use_shrinkage": False,
    })
    assert resp.status_code == 200
    assert resp.get_json()["shrinkage_intensity"] is None


def _fake_fundamentals(tickers):
    return {
        t: {
            "name": f"{t} Inc.", "price": 100.0, "currency": "USD",
            "market_cap": 2_500_000_000_000, "pe": 28.5,
            "revenue": 390_000_000_000, "beta": 1.2,
        }
        for t in tickers
    }


def test_fundamentals_endpoint_returns_metrics(client, monkeypatch):
    monkeypatch.setattr(backend, "fetch_fundamentals", _fake_fundamentals)
    resp = client.post("/api/fundamentals", json={"tickers": ["AAPL", "MSFT"]})
    assert resp.status_code == 200
    funds = resp.get_json()["fundamentals"]
    assert set(funds.keys()) == {"AAPL", "MSFT"}
    for sym in ("AAPL", "MSFT"):
        assert {"market_cap", "pe", "revenue", "beta"} <= set(funds[sym].keys())


def test_fundamentals_allows_single_ticker(client, monkeypatch):
    monkeypatch.setattr(backend, "fetch_fundamentals", _fake_fundamentals)
    resp = client.post("/api/fundamentals", json={"tickers": ["AAPL"]})
    assert resp.status_code == 200
    assert "AAPL" in resp.get_json()["fundamentals"]


def test_fundamentals_requires_at_least_one_ticker(client):
    resp = client.post("/api/fundamentals", json={"tickers": []})
    assert resp.status_code == 400


def test_fundamentals_get_is_ok(client):
    resp = client.get("/api/fundamentals")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_optimize_get_is_ok(client):
    resp = client.get("/api/optimize")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_request_parser_normalizes_and_deduplicates_tickers():
    params = backend._parse_request({
        "tickers": [" aapl ", "AAPL", " msft", ""],
        "rf": 0.03,
        "lookback_years": 5,
        "allow_shorting": True,
        "use_shrinkage": False,
    })

    assert params == {
        "tickers": ["AAPL", "MSFT"],
        "rf": 0.03,
        "lookback_years": 5,
        "allow_shorting": True,
        "use_shrinkage": False,
    }


def test_optimize_rejects_missing_json_and_too_many_tickers(client):
    missing = client.post(
        "/api/optimize", data="null", content_type="application/json"
    )
    too_many = client.post(
        "/api/optimize",
        json={"tickers": [f"T{i}" for i in range(backend.MAX_TICKERS + 1)]},
    )

    assert missing.status_code == 400
    assert missing.get_json()["error"] == "No JSON data provided"
    assert too_many.status_code == 400
    assert too_many.get_json()["error"] == "Maximum 10 tickers allowed"


def test_clean_prices_drops_empty_symbols_and_fills_internal_gaps():
    prices = pd.DataFrame({
        "AAPL": [100.0, np.nan, 102.0],
        "INVALID": [np.nan, np.nan, np.nan],
    })

    cleaned = backend._clean_prices(prices)

    assert list(cleaned.columns) == ["AAPL"]
    np.testing.assert_allclose(cleaned["AAPL"], [100.0, 100.0, 102.0])


def test_options_expirations_endpoint_returns_provider_payload(client, monkeypatch):
    def fake_fetch(ticker):
        assert ticker == "AAPL"
        return {
            "ticker": ticker,
            "underlying_price": 200.0,
            "expirations": ["2026-06-19"],
        }

    monkeypatch.setattr(backend, "fetch_expirations", fake_fetch)

    resp = client.post("/api/options/expirations", json={"ticker": " aapl "})

    assert resp.status_code == 200
    assert resp.get_json()["expirations"] == ["2026-06-19"]


def test_options_expirations_rejects_empty_provider_result(client, monkeypatch):
    monkeypatch.setattr(
        backend,
        "fetch_expirations",
        lambda ticker: {"ticker": ticker, "expirations": []},
    )

    resp = client.post("/api/options/expirations", json={"ticker": "AAPL"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No listed options found for AAPL"


def test_options_chain_endpoint_forwards_model_parameters(client, monkeypatch):
    captured = {}

    def fake_chain(ticker, expiry, rf, q):
        captured.update(ticker=ticker, expiry=expiry, rf=rf, q=q)
        return {
            "ticker": ticker,
            "expiry": expiry,
            "underlying_price": 200.0,
            "risk_free_rate": rf,
            "years_to_expiry": 0.5,
            "calls": [],
            "puts": [],
        }

    monkeypatch.setattr(backend, "fetch_option_chain", fake_chain)

    resp = client.post("/api/options/chain", json={
        "ticker": " aapl ",
        "expiry": "2026-06-19",
        "rf": 0.035,
        "dividend_yield": 0.01,
    })

    assert resp.status_code == 200
    assert captured == {
        "ticker": "AAPL",
        "expiry": "2026-06-19",
        "rf": 0.035,
        "q": 0.01,
    }
    assert resp.get_json()["ticker"] == "AAPL"


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ({"ticker": ""}, "Need a ticker symbol"),
        ({"ticker": "AAPL"}, "Need an expiry date (YYYY-MM-DD)"),
    ],
)
def test_options_chain_validates_required_fields(client, payload, expected_error):
    resp = client.post("/api/options/chain", json=payload)

    assert resp.status_code == 400
    assert resp.get_json()["error"] == expected_error


def test_options_chain_translates_provider_value_error(client, monkeypatch):
    def reject_chain(*args, **kwargs):
        raise ValueError("invalid expiry")

    monkeypatch.setattr(backend, "fetch_option_chain", reject_chain)

    resp = client.post("/api/options/chain", json={
        "ticker": "AAPL", "expiry": "1900-01-01"
    })

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid expiry"


def test_api_preflight_returns_cors_headers(client):
    resp = client.options("/api/optimize")

    assert resp.status_code == 200
    assert resp.headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in resp.headers["Access-Control-Allow-Methods"]


def test_home_serves_the_frontend(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert "text/html" in resp.content_type
    assert b"<html" in resp.data.lower()
