"""
Module pour télécharger les données financières directement depuis Yahoo Finance API
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

def download_yahoo_data(symbol, start_date, end_date):
    """Télécharge les données depuis Yahoo Finance API directement"""
    
    # Convertir dates en timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history"
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            print(f"No data for {symbol}")
            return None
            
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        
        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit='s'),
            "Open": quotes["open"],
            "High": quotes["high"],
            "Low": quotes["low"],
            "Close": quotes["close"],
            "Volume": quotes["volume"]
        })
        df.set_index("Date", inplace=True)
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

def download_multiple_symbols(symbols, start_date, end_date):
    """Télécharge plusieurs symboles et retourne un DataFrame avec les prix de clôture"""
    
    all_data = {}
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        df = download_yahoo_data(symbol, start_date, end_date)
        if df is not None and not df.empty:
            all_data[symbol] = df["Close"]
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.DataFrame(all_data)
    result = result.dropna()
    return result

# Test
if __name__ == "__main__":
    symbols = ["SPY", "TLT", "GLD"]
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    df = download_multiple_symbols(symbols, start, end)
    print(f"\nDownloaded {len(df)} days of data")
    print(df.tail())
