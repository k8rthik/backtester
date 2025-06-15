import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Load Data ---
btc = yf.download("BTC-USD", start="2020-01-01", end="2023-01-01")[['Close']]
btc.dropna(inplace=True)

# --- Feature Engineering ---
btc['MA10'] = btc['Close'].rolling(window=10).mean()
btc['Return'] = btc['Close'].pct_change()
btc['Signal'] = (btc['Return'].shift(-1) > 0).astype(int)
btc.dropna(inplace=True)

# --- AI Model ---
features = ['MA10', 'Return']
X = btc[features]
y = btc['Signal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier()
model.fit(X_train, y_train)
btc['Prediction'] = model.predict(X)

# --- Simulated Trading ---
initial_balance = 1000.0
balance = initial_balance
position = 0.0
entry_price = 0.0
leverage = 10
btc['Balance'] = np.nan

for i in range(1, len(btc)):
    signal = btc['Prediction'].iloc[i-1]
    price = float(btc['Close'].iloc[i])  # make sure it's scalar

    # Buy signal
    if signal == 1 and position == 0.0:
        position = (balance * leverage) / price
        entry_price = price
        print(f"[BUY] {btc.index[i].date()} at ${price:.2f}, position: {position:.4f}")

    # Sell signal
    elif signal == 0 and position > 0.0:
        proceeds = position * price
        borrowed = balance * leverage
        balance += proceeds - borrowed
        print(f"[SELL] {btc.index[i].date()} at ${price:.2f}, new balance: ${balance:.2f}")
        position = 0.0
        entry_price = 0.0

    # Track balance
    btc.loc[btc.index[i], 'Balance'] = balance

btc['Balance'].fillna(method='ffill', inplace=True)

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(btc.index, btc['Balance'], label='Simulated Balance')
plt.plot(btc.index, btc['Close'], label='BTC Price')
plt.title("BTC Backtest (AI + Leverage Trading)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
