import datetime
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from copy import deepcopy
import pandas_datareader as data
import yfinance as yf 
import requests

def load_data(stock_symbol):
    url = f'https://www.google.com/finance/quote/{stock_symbol}/history'  
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.content.decode('utf-8')
        print(data)  
        df = pd.read_csv(StringIO(data))
        return df
    else:
        st.error(f"Failed to fetch data for stock symbol {stock_symbol}")
        return None

def str_to_datetime(s):
    return pd.to_datetime(s)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    while target_date <= last_date:
        df_subset = dataframe.loc[:target_date][-(n+1):]

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
        else:
            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

        target_date += datetime.timedelta(days=1)

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n-i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


# Streamlit app
st.title("Stock Price Prediction")

st.sidebar.header("User Inputs")
stock_symbol = st.sidebar.selectbox("Select Stock Symbol", ["AAPL", "ABNB", "ADBE", "AMC", "AMZN", "AMD", "BABA", "BAC", "BA",
    "BB", "CGX.TO", "CLSK", "CMG", "CSCO", "CRM", "CVX", "DIA", "DIS", "DNA",
    "DNUT", "DWAC", "EDBL", "ENVX", "ETSY", "F", "FB", "GME", "GM", "GOOG",
    "GOOGL", "HD", "ILMN", "INTC", "INVZ", "ITC", "IRCTC.NS", "JNJ", "JPM", "KO", "LQR",
    "LUMN", "MCD", "MELI", "MGNI", "MS", "MSFT", "MRNA", "NKE", "NOK", "NVDA",
    "ORGN", "PEP", "PLUG", "PYPL", "PFE", "QQQ", "RELIANCE", "SINGD", "SONO",
    "SPCE", "SPY", "SQ", "T", "TTD", "TPR", "TSLA", "TWTR", "UBER", "UNH", "V",
    "VXX", "WELL.TO", "WFC", "WMT", "XOM", "ZM"])

start_date = st.sidebar.date_input("Start Date", datetime.date.today()-datetime.timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
window_size = st.sidebar.slider("Window Size", min_value=1, max_value=10, value=3)
submit_button = st.sidebar.button("Submit")

if submit_button:
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    windowed_df = df_to_windowed_df(df, start_date, end_date, window_size)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)
    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    model = Sequential([layers.Input((window_size, 1)), layers.LSTM(64), layers.Dense(32, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(1)])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'], run_eagerly=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()

    recursive_predictions = []
    recursive_dates = np.concatenate([dates_val, dates_test])
    last_window = deepcopy(X_train[-1])  

    for _ in range(len(recursive_dates)):
        next_prediction = model.predict(np.array([last_window])).flatten()
        last_window[-1] = next_prediction 
        recursive_predictions.append(next_prediction)

    
    st.subheader("Training Predictions")
    st.line_chart(pd.DataFrame({'Date': dates_train, 'Predictions': train_predictions, 'Observations': y_train}).set_index('Date'))

    st.subheader("Validation Predictions")
    st.line_chart(pd.DataFrame({'Date': dates_val, 'Predictions': val_predictions, 'Observations': y_val}).set_index('Date'))

    st.subheader("Testing Predictions")
    st.line_chart(pd.DataFrame({'Date': dates_test, 'Predictions': test_predictions, 'Observations': y_test}).set_index('Date'))

    recursive_dates_list = recursive_dates.tolist()
    recursive_predictions_list = [float(pred) for pred in recursive_predictions]

    st.subheader("Recursive Predictions")
    st.line_chart(pd.DataFrame({'Date': recursive_dates_list, 'Predictions': recursive_predictions_list}).set_index('Date'))
