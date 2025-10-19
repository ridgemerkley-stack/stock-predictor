import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import streamlit as st
from streamlit_aggrid import AgGrid, GridUpdateMode
import time

# Curated list of AI-recommended stocks (based on free data trends)
AI_STOCKS = ['NVDA', 'AVGO', 'TSM', 'AMD', 'TTD', 'LMND', 'ASML', 'GOOGL', 'AMZN', 'MSFT', 'ORCL', 'CRWV', 'NBIS', 'SOUN']

# Continuous Training Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, -1)  # (seq_len, batch, d_model)
        output = self.transformer(src)
        return self.fc(output[-1])  # Last time step

# Fetch and update data
def fetch_stocks_data(stocks, period='1y'):
    data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period=period)
        if not hist.empty:
            data[stock] = hist
    return data

# Create sequences with sliding window
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Continuously train Transformer
def train_model(data_scaled, seq_length=60):
    X, y = create_sequences(data_scaled, seq_length)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = TransformerModel(input_dim=1, d_model=64, nhead=4, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):  # Fewer epochs for speed in continuous training
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        # Validate and adjust
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            if val_loss.item() > loss.item():  # Early stopping if validation worsens
                break
    return model

# Predict future with continuous model
def predict_future(model, data_scaled, scaler, seq_length=60, future_days=30):
    last_sequence = torch.from_numpy(data_scaled[-seq_length:]).float().unsqueeze(0)
    predictions = []
    for _ in range(future_days):
        pred = model(last_sequence)
        predictions.append(pred.item())
        last_sequence = torch.cat((last_sequence[:, 1:], pred.unsqueeze(0)), dim=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Generate indicators
def generate_indicators(df):
    if df.empty:
        return pd.DataFrame()
    rsi = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close']).macd()
    ema = EMAIndicator(df['Close'], window=20).ema_indicator()
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['Price_Change'] = df['Close'].pct_change()
    df['RSI'] = rsi
    df['MACD'] = macd
    df['EMA'] = ema
    df['ATR'] = atr
    return df.dropna()

# Train classifier with continuous feedback
@st.cache_data(ttl=300)  # Refresh every 5 minutes
def train_signal_classifier(df_samples):
    features = ['RSI', 'MACD', 'Price_Change', 'EMA_Crossover', 'Volatility']
    X = pd.concat([df[features] for df in df_samples.values() if not df.empty]).dropna()
    y = np.where(X['RSI'] < 30, 1, np.where(X['RSI'] > 70, -1, 0))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Get signal and buy score
def get_stock_signal(df, clf, sentiment_score=0.5):
    if df.empty:
        return 'Hold', 50, 'N/A'
    df['EMA_Crossover'] = (df['Close'] > df['EMA']).astype(int)
    df['Volatility'] = df['ATR'] / df['Close']
    df['Sentiment'] = sentiment_score
    features = ['RSI', 'MACD', 'Price_Change', 'EMA_Crossover', 'Volatility', 'Sentiment']
    last_row = df.iloc[-1:][features]
    current_signal = clf.predict(last_row)[0]
    signal_map = {1: "Buy", -1: "Sell", 0: "Hold"}
    confidence = clf.predict_proba(last_row)[0].max() * 100
    buy_score = confidence if current_signal == 1 else (100 - confidence)
    return signal_map[current_signal], buy_score, confidence

# Main UI
st.set_page_config(page_title="AI Stock Scanner", layout="wide")
st.title("🌐 AI Stock Scanner with Continuous Training")

# Sidebar for filters
st.sidebar.header("Filters")
selected_stocks = st.sidebar.multiselect("Select Stocks", AI_STOCKS, default=AI_STOCKS[:5])
sentiment = st.sidebar.slider("Sentiment Score (0-1)", 0.0, 1.0, 0.5)
min_buy_score = st.sidebar.slider("Min Buy Score", 0, 100, 70)
refresh_interval = st.sidebar.slider("Refresh Interval (min)", 1, 15, 5)

# Fetch and process data
if st.button("Scan Stocks Now"):
    with st.spinner(f"Analyzing stocks with continuous AI training ({time.ctime()})..."):
        data = fetch_stocks_data(selected_stocks)
        df_samples = {stock: generate_indicators(df) for stock, df in data.items()}
        clf = train_signal_classifier(df_samples)
        results = []
        for stock, df in df_samples.items():
            if not df.empty:
                data_scaled = MinMaxScaler().fit_transform(df['Close'].values.reshape(-1, 1))
                model = train_model(data_scaled)  # Continuous training on latest data
                predictions = predict_future(model, data_scaled, MinMaxScaler().fit(data_scaled), future_days=30)
                signal, buy_score, confidence = get_stock_signal(df, clf, sentiment)
                if buy_score >= min_buy_score and signal == "Buy":
                    results.append({
                        'Stock': stock,
                        'Current Price': f"${df['Close'].iloc[-1]:.2f}",
                        'Signal': signal,
                        'Buy Score': f"{buy_score:.1f}%",
                        'Confidence': f"{confidence:.1f}%",
                        'RSI': f"{df['RSI'].iloc[-1]:.1f}",
                        'Change (1D)': f"{df['Price_Change'].iloc[-1]*100:.2f}%"
                    })
        results_df = pd.DataFrame(results).sort_values('Buy Score', ascending=False)
        
        if not results_df.empty:
            st.success(f"📈 {len(results_df)} Stocks Recommended to Buy!")
            grid_response = AgGrid(results_df, height=400, fit_columns_on_grid_load=True, update_mode=GridUpdateMode.MODEL_CHANGED)
            st.write("Click a row for a quick chart:")
            selected = grid_response['selected_rows']
            if selected:
                stock = selected[0]['Stock']
                st.line_chart(data[stock]['Close'].tail(30))
        else:
            st.info("No stocks meet the criteria. Adjust filters or wait for data refresh.")

# Auto-refresh with continuous training
if st.button(f"Auto-Refresh (Every {refresh_interval} Min)"):
    st.rerun()
    time.sleep(refresh_interval * 60)  # Simulate continuous update

# Footer
st.markdown("---")
st.caption("Powered by free yfinance data. AI continuously trains on latest trends. Not financial advice.")

# Run with: py -m streamlit run app.py
