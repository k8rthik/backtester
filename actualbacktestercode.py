import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
initial_balance = 1000.0
leverage = 3
risk_fraction = 0.3
stop_loss_pct = 0.05
slippage_pct = 0.001  # 0.1% per trade
start_date = "2020-01-01"
end_date = "2023-01-01"
train_cutoff = "2022-01-01"

# --- RSI FUNCTION ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- DOWNLOAD BTC DATA ---
btc_raw = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
btc = pd.DataFrame()
btc['Close'] = btc_raw['Close'].copy()

# --- FEATURE ENGINEERING ---
btc['MA10'] = btc['Close'].rolling(10).mean()
btc['Return'] = btc['Close'].pct_change()
btc['Momentum'] = btc['Close'] - btc['MA10']
btc['RSI'] = compute_rsi(btc['Close'])
ema12 = btc['Close'].ewm(span=12, adjust=False).mean()
ema26 = btc['Close'].ewm(span=26, adjust=False).mean()
btc['MACD'] = ema12 - ema26
btc['BollWidth'] = btc['Close'].rolling(20).std() / btc['MA10']
btc['Signal'] = (btc['Return'].shift(-1) > 0).astype(int)
btc.dropna(inplace=True)

# --- SPLIT DATA ---
features = ['MA10', 'Return', 'Momentum', 'RSI', 'MACD', 'BollWidth']
X = btc[features]
y = btc['Signal']
X_train = X.loc[:train_cutoff]
y_train = y.loc[:train_cutoff]
X_test = X.loc[train_cutoff:]
btc_test = btc.loc[train_cutoff:]

# --- TRAIN MODEL ---
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
btc_test['Prediction'] = model.predict(X_test)

# --- BACKTEST ---
balance = initial_balance
position = 0.0
entry_price = 0.0
btc_test['Balance'] = np.nan
btc_test['BuyHold'] = initial_balance * (btc_test['Close'] / btc_test['Close'].iloc[0])
trade_log = []

for i in range(1, len(btc_test)):
    signal = btc_test['Prediction'].iloc[i - 1]
    price = float(btc_test['Close'].iloc[i])
    date = btc_test.index[i]

    # Stop-loss
    if position > 0 and price < entry_price * (1 - stop_loss_pct):
        proceeds = position * price * (1 - slippage_pct)
        borrowed = position * entry_price - (position * entry_price) / leverage
        balance += proceeds - borrowed
        trade_log.append((date, "STOP-LOSS", price, balance))
        position = 0.0
        entry_price = 0.0

    # Buy
    elif signal == 1 and position == 0:
        capital_to_use = balance * risk_fraction
        position = (capital_to_use * leverage) / (price * (1 + slippage_pct))
        entry_price = price
        trade_log.append((date, "BUY", price, balance))

    # Sell
    elif signal == 0 and position > 0:
        proceeds = position * price * (1 - slippage_pct)
        borrowed = (position * entry_price) - (position * entry_price) / leverage
        balance += proceeds - borrowed
        trade_log.append((date, "SELL", price, balance))
        position = 0.0
        entry_price = 0.0

    btc_test.loc[date, 'Balance'] = balance

btc_test['Balance'].fillna(method='ffill', inplace=True)

# --- PLOT ---
plt.figure(figsize=(14, 6))
plt.plot(btc_test.index, btc_test['Balance'], label='AI Strategy Balance ($)', linewidth=2)
plt.plot(btc_test.index, btc_test['BuyHold'], label='Buy & Hold BTC ($)', linestyle='--', alpha=0.8)
plt.plot(btc_test.index, btc_test['Close'], label='BTC Price ($)', alpha=0.4)

plt.title("BTC Backtest (AI vs Buy & Hold)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Dollars")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --- TRADE LOG ---
print("\n--- Trade Log ---")
for t in trade_log:
    print(f"{t[0].date()} | {t[1]:<10} | Price: ${t[2]:.2f} | Balance: ${t[3]:.2f}")
