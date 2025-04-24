#!/usr/bin/env python3
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def download_crypto_data(symbol, start_date, end_date, save_path):
    """
    Download cryptocurrency data from Yahoo Finance and save as CSV.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'ETH-USD')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_path: Complete path to save the CSV file
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    # Download data using yfinance
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Check if data was downloaded successfully
    if data.empty:
        print(f"No data available for {symbol} in the specified date range.")
        return
    
    # Rename the index column to 'Price' (matches your existing files format)
    data.index.name = 'Price'
    
    # Save to CSV
    data.to_csv(save_path)
    print(f"Data successfully saved to {save_path}")
    print(f"Downloaded {len(data)} rows of data.")

if __name__ == "__main__":
    # Parameters
    symbol = "ETH-USD"
    start_date = "2023-01-01"
    end_date = "2025-04-01"
    save_path = "/Users/nazaraktosun/CryptoCapade/trainers/sample_data/ETH-USD_data.csv"
    
    # Download and save data
    download_crypto_data(symbol, start_date, end_date, save_path)