def add_features(vix_data, spvi_data, fear_greed_data):
    features = {
        'vix': vix_data['Close'],
        'spvi': spvi_data['volatility'],
        'fear_greed': fear_greed_data['value']
    }
    return features
