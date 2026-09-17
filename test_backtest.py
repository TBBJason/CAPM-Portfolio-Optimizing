"""Unit tests for deterministic portfolio backtesting."""
import numpy as np
import pandas as pd

import backtest as backtest_module


def test_backtest_uses_column_aligned_weights_and_compounds_returns():
    prices = pd.DataFrame({
        "AAA": [100.0, 110.0, 121.0],
        "BBB": [200.0, 180.0, 198.0],
    })

    values = backtest_module.backtest(
        ["AAA", "BBB"], np.array([0.6, 0.4]), prices=prices
    )

    # Daily portfolio returns are 2% and 10%, compounded from a value of 1.
    np.testing.assert_allclose(values, [1.0, 1.02, 1.122])


def test_backtest_downloads_default_one_year_window_and_accepts_series(monkeypatch):
    end = pd.Timestamp("2025-06-30")
    calls = []

    def fake_download(stocks, start, end):
        calls.append((stocks, start, end))
        return pd.Series([100.0, 105.0], name="AAA")

    monkeypatch.setattr(backtest_module, "download_stock_data", fake_download)

    values = backtest_module.backtest(["AAA"], [1.0], end=end)

    assert calls == [(["AAA"], end - pd.DateOffset(years=1), end)]
    np.testing.assert_allclose(values, [1.0, 1.05])
