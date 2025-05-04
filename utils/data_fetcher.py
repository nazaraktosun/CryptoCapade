import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import time
from yfinance.shared import YFRateLimitError
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def _download_crypto(
    ticker: str,
    start: str,
    end: str
) -> pd.DataFrame:
    """Download OHLCV data for a ticker with retries on rate limit and cache for 1 hour."""
    for attempt in range(3):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                progress=False
            )
            return df
        except YFRateLimitError:
            # exponential backoff
            time.sleep(2 ** attempt)
    # Final attempt (allow any error to bubble)
    return yf.download(
        tickers=ticker,
        start=start,
        end=end,
        progress=False
    )

class DataFetcher:
    """
    DataFetcher fetches cryptocurrency data using yfinance
    and returns a clean DataFrame ready for feature building.
    """

    def __init__(self):
        pass

    def get_crypto_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        compute_log_returns: bool = True,
        n_lags: int = 5
    ) -> pd.DataFrame:
        """
        Fetch historical crypto data, coerce types, set the Date index, and optionally add log returns.

        :param symbol:      Cryptocurrency symbol (e.g. "BTC", "ETH")
        :param start_date:  Start date for analysis
        :param end_date:    End date for analysis
        :param compute_log_returns: Whether to add a "Log Returns" column
        :param n_lags:      If compute_log_returns, how many lags your FeatureBuilder will need
        :return:            DataFrame, indexed by Date, with numeric Open, High, Low, Close, Volume
                            (and Log Returns if requested)
        """
        ticker = f"{symbol}-USD"
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Download data (cached and retried)
        df = _download_crypto(ticker, start_str, end_str)

        if df.empty:
            raise ValueError(f"No data fetched for ticker {ticker}. Please verify the symbol and try again.")

        # Handle MultiIndex returned by yf.download
        if isinstance(df.columns, pd.MultiIndex) and "Ticker" in df.columns.names:
            df.columns = df.columns.droplevel("Ticker")

        # Select and coerce columns
        df = df[ ["Open", "High", "Low", "Close", "Volume"] ]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop missing rows
        df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

        # Compute log returns if requested
        if compute_log_returns:
            df["Log Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["Log Returns"], inplace=True)

        # Finalize index
        df.index.name = "Date"
        df.sort_index(inplace=True)

        return df
