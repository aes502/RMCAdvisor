# RMC AI Advisor - Financial Market Analysis Using SPVI, VIX, and Fear-Greed Indexes with Stock Price Prediction using LSTM

## Overview

RMC AI Advisor is a financial prediction platform that leverages key market indices like SPVI, VIX, and Fear-Greed Indexes. This system uses AI to analyze trends, market sentiment, and volatility to provide real-time insights and predict stock prices using LSTM.

## Key Features
- Market analysis using SPVI, VIX, and Fear-Greed indexes.
- Stock price prediction using LSTM (Long Short-Term Memory) neural networks.
- NLP chatbot for personalized financial insights.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aes502/RMCAdvisor.git
   cd RMCAdvisor
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```


## Usage

Run the model:
```bash
python src/stock_prediction.py --symbol AAPL --start_date 2012-01-01 --end_date 2021-12-31 --days_forward 30
```

## License

This project is licensed under the MIT License.
