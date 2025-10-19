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
from flask import Flask, render_template, request
import time

app = Flask(__name__)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, -1)
        output = self.transformer(src)
        return self.fc(output[-1])

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_model(data_scaled, seq_length=60):
    X, y = create_sequences(data_scaled, seq_length)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = TransformerModel(input_dim=1, d_model=64, nhead=4, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    return model

def predict_future(model, data_scaled, scaler, seq_length=60, future_days=30):
    last_sequence = torch.from_numpy(data_scaled[-seq_length:]).float().unsqueeze(0)
    predictions = []
    for _ in range(future_days):
        pred = model(last_sequence)
        predictions.append(pred.item())
        last_sequence = torch.cat((last_sequence[:, 1:], pred.unsqueeze(0)), dim=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

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

def train_signal_classifier(df_samples):
    features = ['RSI', 'MACD', 'Price_Change', 'EMA_Crossover', 'Volatility']
    X = pd.concat([df[features] for df in df_samples.values() if not df.empty]).dropna()
    y = np.where(X['RSI'] < 30, 1, np.where(X['RSI'] > 70, -1, 0))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    return clf

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

def fetch_stocks_data(stocks):
    data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period='1y')
        if not hist.empty:
            data[stock] = hist
    return data

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Stock Scanner</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .filter { margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .chart { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>AI Stock Scanner with Continuous Training</h1>
    <div class="filter">
        <label for="stocks">Enter Stock Tickers (comma-separated, e.g., AAPL, MSFT):</label><br>
        <textarea id="stocks" name="stocks" rows="2" cols="50">{{ stocks_input }}</textarea><br>
        <label for="sentiment">Sentiment Score (0-1):</label>
        <input type="range" id="sentiment" name="sentiment" min="0.0" max="1.0" step="0.1" value="{{ sentiment }}" oninput="this.nextElementSibling.value=this.value">
        <output>{{ sentiment }}</output><br>
        <label for="min_buy">Min Buy Score:</label>
        <input type="range" id="min_buy" name="min_buy" min="0" max="100" step="1" value="{{ min_buy }}" oninput="this.nextElementSibling.value=this.value">
        <output>{{ min_buy }}</output><br>
        <button onclick="scanStocks()">Scan Stocks Now</button>
        <button onclick="autoRefresh()">Auto-Refresh (5 Min)</button>
    </div>
    <div id="results">{{ results|safe }}</div>
    <div class="chart" id="chart">{{ chart|safe }}</div>
    <script>
        function scanStocks() {
            var stocks = document.getElementById('stocks').value;
            var sentiment = document.getElementById('sentiment').value;
            var min_buy = document.getElementById('min_buy').value;
            window.location.href = '/scan?stocks=' + encodeURIComponent(stocks) + '&sentiment=' + sentiment + '&min_buy=' + min_buy;
        }
        function autoRefresh() {
            setInterval(scanStocks, 300000); // 5 minutes
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template, stocks_input="", sentiment=0.5, min_buy=70)

@app.route('/scan')
def scan_stocks():
    stocks_input = request.args.get('stocks', '')
    selected_stocks = [s.strip() for s in stocks_input.split(',') if s.strip()]
    sentiment = float(request.args.get('sentiment', 0.5))
    min_buy = float(request.args.get('min_buy', 70))

    if not selected_stocks:
        return render_template_string(html_template, stocks_input=stocks_input, sentiment=sentiment, min_buy=min_buy, results="No stocks entered.", chart="")

    with st.spinner("Analyzing stocks with continuous AI training..."):
        data = fetch_stocks_data(selected_stocks)
        df_samples = {stock: generate_indicators(df) for stock, df in data.items()}
        clf = train_signal_classifier(df_samples)
        results = []
        for stock, df in df_samples.items():
            if not df.empty:
                data_scaled = MinMaxScaler().fit_transform(df['Close'].values.reshape(-1, 1))
                model = train_model(data_scaled)
                predictions = predict_future(model, data_scaled, MinMaxScaler().fit(data_scaled), future_days=30)
                signal, buy_score, confidence = get_stock_signal(df, clf, sentiment)
                if buy_score >= min_buy and signal == "Buy":
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

        if results_df.empty:
            return render_template_string(html_template, stocks_input=stocks_input, sentiment=sentiment, min_buy=min_buy, results="No stocks meet the criteria.", chart="")

        table_html = results_df.to_html(index=False, classes='table table-striped')
        chart_html = "<div>No chart available</div>"  # Simplified; add Plotly if desired
        return render_template_string(html_template, stocks_input=stocks_input, sentiment=sentiment, min_buy=min_buy, results=table_html, chart=chart_html)

if __name__ == '__main__':
    app.run(debug=True)
    