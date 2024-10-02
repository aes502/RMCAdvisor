import numpy as np
from lstm_model import load_stock_data, preprocess_stock_data, create_lstm_sequences, train_lstm_model, predict_stock_price

def run_stock_prediction(symbol, start_date, end_date, days_forward):
    # Load stock data
    stock_data = load_stock_data(symbol, start_date, end_date)
    
    # Preprocess data
    scaled_data, scaler = preprocess_stock_data(stock_data)
    
    # Create LSTM sequences
    X_train, y_train = create_lstm_sequences(scaled_data)
    
    # Reshape data to 3D input for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Train LSTM model
    input_shape = (X_train.shape[1], 1)
    model = train_lstm_model(X_train, y_train, input_shape)
    
    # Prepare test data and predict
    test_data = scaled_data[-days_forward:]
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
    
    predictions = predict_stock_price(model, test_data, scaler)
    
    print(f"Predicted stock prices for {symbol}: {predictions}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock Price Prediction using LSTM")
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days_forward', type=int, default=30, help='Number of days to predict forward')
    
    args = parser.parse_args()
    run_stock_prediction(args.symbol, args.start_date, args.end_date, args.days_forward)
