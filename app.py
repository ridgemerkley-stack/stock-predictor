import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to fetch and preprocess data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")
    return data['Close'].values.reshape(-1, 1)

# Function to create dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to build and train LSTM model
def train_model(data_scaled, time_step=60):
    X, y = create_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=64, epochs=10, verbose=1)  # Reduced epochs for speed; increase for accuracy
    return model

# Function to make predictions
def predict_future(model, data_scaled, scaler, time_step=60, future_days=30):
    last_sequence = data_scaled[-time_step:].reshape(1, time_step, 1)
    predictions = []
    
    for _ in range(future_days):
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions
if __name__ == "__main__":
    data = fetch_data("AAPL", "2020-01-01", "2023-01-01")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    model = train_model(data_scaled)
    predictions = predict_future(model, data_scaled, scaler)
    print("Sample predictions:", predictions[:5])  # Prints first 5 predicted values
    import streamlit as st

# Streamlit UI
st.title("Stock Market Predictor")

# User inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))
future_days = st.slider("Days to Predict", min_value=1, max_value=90, value=30)

if st.button("Train Model and Predict"):
    try:
        with st.spinner("Fetching data and training model..."):
            data = fetch_data(ticker, start_date, end_date)
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data)
            
            model = train_model(data_scaled)
            predictions = predict_future(model, data_scaled, scaler, future_days=future_days)
        
        # Display historical data chart
        historical_df = pd.DataFrame(data, columns=["Close"], index=yf.download(ticker, start=start_date, end=end_date).index)
        st.subheader("Historical Closing Prices")
        st.line_chart(historical_df)
        
        # Display predictions
        future_dates = pd.date_range(start=historical_df.index[-1] + pd.Timedelta(days=1), periods=future_days)
        pred_df = pd.DataFrame(predictions, columns=["Predicted Close"], index=future_dates)
        st.subheader(f"Predicted Prices for Next {future_days} Days")
        st.line_chart(pred_df)
        
        # Combined chart
        combined_df = pd.concat([historical_df.tail(60), pred_df])
        st.subheader("Combined Historical and Predicted Prices")
        st.line_chart(combined_df)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")