import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data():
    vix_data = pd.read_csv("data/vix.csv")
    fear_greed_data = pd.read_csv("data/fear_greed.csv")
    spvi_data = pd.read_csv("data/spvi.csv")
    
    return vix_data, fear_greed_data, spvi_data

def preprocess_data(dataframes):
    scaler = MinMaxScaler()
    scaled_data = [scaler.fit_transform(df) for df in dataframes]
    return scaled_data
