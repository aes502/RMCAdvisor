import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import build_lstm_model
import yfinance as yf

def load_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocess_stock_data(stock_data):
    data = stock_data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_lstm_sequences(scaled_data, look_back=60):
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    return np.array(X_train), np.array(y_train)

def train_lstm_model(X_train, y_train, input_shape):
    model = build_lstm_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

def predict_stock_price(model, test_data, scaler):
    predicted_prices = model.predict(test_data)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices
