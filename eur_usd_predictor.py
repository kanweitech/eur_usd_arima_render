import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import streamlit as st

def fetch_forex_data(symbol="EURUSD=X", interval="1m", period="1d"):
    data = yf.download(tickers=symbol, interval=interval, period=period)
    data = data[['Close']].dropna()
    return data

def fetch_news_headlines():
    return [
        "EUR/USD rallies amid Fed rate hike pause",
        "Euro under pressure as German inflation dips",
        "Positive outlook on the European economy boosts Euro",
    ]

def get_sentiment_score(headlines):
    if not headlines:
        return 0.0
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return sum(scores) / len(scores)

def add_sentiment_to_data(data):
    sentiment = get_sentiment_score(fetch_news_headlines())
    data['Sentiment'] = sentiment
    return data

def apply_arima_forecast(df):
    prices = df['Close'].values
    model = ARIMA(prices, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)[0]
    df['Forecast'] = np.nan
    df.at[df.index[-1], 'Forecast'] = forecast
    return df, forecast

def generate_signals(df):
    df['Signal'] = 0
    last_idx = df.index[-1]

    forecast = df.at[last_idx, 'Forecast']
    close = df.at[last_idx, 'Close']
    sentiment = df.at[last_idx, 'Sentiment']

    if forecast > close and sentiment > 0:
        df.at[last_idx, 'Signal'] = 1  # Buy
    elif forecast < close and sentiment < 0:
        df.at[last_idx, 'Signal'] = -1  # Sell
    return df


def backtest(df, initial_balance=10000):
    balance = initial_balance
    position = 0
    entry_price = 0
    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
        elif signal == -1 and position == 0:
            position = -1
            entry_price = price
        elif signal == 0 and position != 0:
            pnl = (price - entry_price) * position
            balance += pnl
            position = 0
    return round(balance, 2)

def plot_forecast(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label='Actual')
    plt.plot(df['Forecast'], label='Forecast', linestyle='--')
    plt.title('EUR/USD ARIMA Forecast')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

def main():
    st.title("EUR/USD Predictor with ARIMA & Sentiment")
    st.write("Forecasting EUR/USD using ARIMA and news sentiment.")

    if st.button("Run Forecast"):
        df = fetch_forex_data()
        df = add_sentiment_to_data(df)
        df, forecast = apply_arima_forecast(df)
        df = generate_signals(df)
        st.dataframe(df.tail())
        plot_forecast(df)

        if st.button("Run Backtest"):
            result = backtest(df)
            st.success(f"Final Balance: ${result}")

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    import sys
    if "streamlit" in sys.argv:
        st.set_page_config(layout="wide")
        main()
    else:
        df = fetch_forex_data()
        df = add_sentiment_to_data(df)
        df, forecast = apply_arima_forecast(df)
        df = generate_signals(df)
        print(df.tail())
