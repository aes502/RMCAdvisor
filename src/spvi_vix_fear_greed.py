import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load stock data (e.g., from Yahoo Finance)
def load_stock_data(symbol, start_date, end_date):
    data = pd.read_csv(f'data/{symbol}.csv', parse_dates=['Date'], index_col='Date')
    return data[(data.index >= start_date) & (data.index <= end_date)]

# Calculate VIX (Volatility Index)
def calculate_vix(stock_data):
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['VIX'] = stock_data['Returns'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
    return stock_data['VIX']

# Calculate Stock-Level Fear-Greed Index
def calculate_fear_greed(stock_data):
    # Sample calculation for Fear-Greed Index based on relative strength and moving averages
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['FearGreed'] = np.where(stock_data['50_MA'] > stock_data['200_MA'], 1, -1)
    return stock_data['FearGreed']

# Calculate Stock-Level SPVI (Sentiment Price Volatility Index)
def calculate_spvi(stock_data, window=20):
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['Volatility'] = stock_data['Close'].pct_change().rolling(window=window).std()
    stock_data['SPVI'] = scaler.fit_transform(stock_data[['Volatility']])
    return stock_data['SPVI']

# Function to combine all indicators
def calculate_indicators(symbol, start_date, end_date):
    stock_data = load_stock_data(symbol, start_date, end_date)

    stock_data['VIX'] = calculate_vix(stock_data)
    stock_data['FearGreed'] = calculate_fear_greed(stock_data)
    stock_data['SPVI'] = calculate_spvi(stock_data)

    return stock_data[['Close', 'VIX', 'FearGreed', 'SPVI']]

# Example Usage
if __name__ == "__main__":
    symbol = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2021-12-31'
    
    stock_indicators = calculate_indicators(symbol, start_date, end_date)
    print(stock_indicators.head())
