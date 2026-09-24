import yfinance as yf
import pandas as pd

def download_daily_prices(symbols, start, end):
    """Download daily adjusted close prices for given symbols."""
    if isinstance(symbols, str):
        symbols = [symbols]
    data = yf.download(symbols, start=start, end=end, auto_adjust=True)
    close = data['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    return close

if __name__ == "__main__":
    # Example usage
    df = download_daily_prices(['AAPL', 'MSFT'], '2020-01-01', '2020-12-31')
    print(type(df), list(df.columns))
    print(df.head())
