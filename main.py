#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bot de Trading de Bitcoin com IA e Aprendizado de Máquina
Modo: Paper trading Binance Testnet
Saídas via print() no terminal (compatível com Railway)
"""

import os
import time
import requests
import numpy  as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

from binance.client    import Client
from binance.enums     import *
from binance.exceptions import BinanceAPIException

from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import train_test_split
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------
# Configurações de API Binance (Testnet)
# ---------------------------------------------------------------------
API_KEY    = os.getenv("BINANCE_API_KEY", "SUA_API_KEY_TESTNET")
API_SECRET = os.getenv("BINANCE_API_SECRET", "SEU_API_SECRET_TESTNET")

# Inicializa cliente já apontando para Testnet
client = Client(API_KEY, API_SECRET, testnet=True)

SYMBOL      = "BTCUSDT"
INITIAL_USD = 450.0
POSITION    = 0.0   # BTC
USD_BALANCE = INITIAL_USD

# Parâmetros de estratégia
TARGET_DAILY_RETURN = 0.008    # 0.8% diário
TARGET_ACCURACY      = 0.86
TIMEFRAMES           = ["1m", "5m"]
FEATURE_WINDOW       = 20       # janelas de tempo para modelo
RETRAIN_INTERVAL     = 24 * 60 * 60  # 1 dia em segundos

# Histórico de trades para log
trade_log = deque(maxlen=10)

# ---------------------------------------------------------------------
# Helpers: indicadores técnicos
# ---------------------------------------------------------------------
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up   = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs      = ma_up / ma_down
    return 100 - 100/(1+rs)

def MACD(df, fast=12, slow=26, signal=9):
    ema_fast    = EMA(df['close'], fast)
    ema_slow    = EMA(df['close'], slow)
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def BBANDS(df, period=20, stddev=2):
    ma    = df['close'].rolling(window=period).mean()
    sd    = df['close'].rolling(window=period).std()
    upper = ma + stddev * sd
    lower = ma - stddev * sd
    return upper, lower

# ---------------------------------------------------------------------
# Sentimento real via API gratuita (Alternative.me)
# ---------------------------------------------------------------------
def fetch_sentiment():
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        return int(data["data"][0]["value"])
    except Exception as e:
        print(f"[Sentiment API error] {e}")
        return np.nan

# ---------------------------------------------------------------------
# Coleta e Pré-processamento
# ---------------------------------------------------------------------
def fetch_klines(symbol, interval, lookback_days=365):
    end_time   = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    data = client.get_historical_klines(
        symbol, interval,
        start_str=str(start_time),
        end_str=str(end_time),
        limit=1000
    )
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df = df[['open_time','open','high','low','close','volume']]
    df[['open','high','low','close','volume']] = \
        df[['open','high','low','close','volume']].astype(float)
    df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('dt', inplace=True)
    return df

def build_features(df):
    df = df.copy()
    df['ema_20']     = EMA(df['close'], 20)
    df['rsi14']      = RSI(df['close'], 14)
    df['macd'], df['macd_sig'] = MACD(df)
    df['bb_upper'], df['bb_lower'] = BBANDS(df)
    df['ret_1']      = df['close'].pct_change(1)
    df['ret_5']      = df['close'].pct_change(5)
    df['vol']        = df['volume'] / df['volume'].rolling(20).mean()
    df['hour']       = df.index.hour
    df['dow']        = df.index.dayofweek
    df['sentiment']  = fetch_sentiment()
    df.dropna(inplace=True)
    return df

def label_data(df, threshold=0.001):
    future_ret = df['close'].shift(-1) / df['close'] - 1
    df['label'] = (future_ret > threshold).astype(int)
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------------------
# Treinamento do Modelo
# ---------------------------------------------------------------------
def train_model(df):
    features = [c for c in df.columns if c not in ['open_time','close','label','volume']]
    X = df[features].values
    y = df['label'].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_lstm, y_lstm = [], []
    for i in range(FEATURE_WINDOW, len(X_scaled)):
        X_lstm.append(X_scaled[i-FEATURE_WINDOW:i])
        y_lstm.append(y[i])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_lstm, test_size=0.2, shuffle=False
    )

    model = Sequential([
        LSTM(64, input_shape=(FEATURE_WINDOW, X_lstm.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=50, batch_size=64, callbacks=[early], verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Treino] Acurácia no teste: {acc*100:.2f}%")
    return model, scaler, features

# ---------------------------------------------------------------------
# Backtesting Simples
# ---------------------------------------------------------------------
def backtest(df, model, scaler, features):
    cash, btc = INITIAL_USD, 0.0
    X = df[features].values
    X_scaled = scaler.transform(X)
    for i in range(FEATURE_WINDOW, len(df)-1):
        window = X_scaled[i-FEATURE_WINDOW:i]
        p = model.predict(window[np.newaxis,:,:], verbose=0)[0,0]
        price = df['close'].iloc[i]
        if p > 0.8 and cash > 0:
            qty = (cash * 0.05) / price
            cash -= qty * price
            btc  += qty
        elif p < 0.2 and btc > 0:
            cash += btc * price
            btc = 0
    final = cash + btc * df['close'].iloc[-1]
    pnl   = (final - INITIAL_USD) / INITIAL_USD * 100
    print(f"[Backtest] Valor final: ${final:.2f} | PnL: {pnl:.2f}%")

# ---------------------------------------------------------------------
# Loop de Paper Trading com prints
# ---------------------------------------------------------------------
def run_paper_trading(model, scaler, features):
    global USD_BALANCE, POSITION

    last_retrain = time.time()
    print("[Bot] Iniciando paper trading...")
    while True:
        try:
            # Re-treina diariamente
            if time.time() - last_retrain > RETRAIN_INTERVAL:
                print(f"[Bot] Re-treinando modelo às {datetime.utcnow()} UTC...")
                df_hist = fetch_klines(SYMBOL, TIMEFRAMES[1])
                df_feat = label_data(build_features(df_hist))
                model, scaler, features = train_model(df_feat)
                last_retrain = time.time()

            # Coleta último candle
            klines = client.get_klines(symbol=SYMBOL, interval=TIMEFRAMES[0], limit=FEATURE_WINDOW+1)
            df_live= pd.DataFrame(klines, columns=[
                "open_time","open","high","low","close","volume","close_time",
                "qav","num_trades","tb","tq","ignore"
            ])
            df_live[['open','high','low','close','volume']] = \
                df_live[['open','high','low','close','volume']].astype(float)
            df_live['dt'] = pd.to_datetime(df_live['open_time'], unit='ms')
            df_live.set_index('dt', inplace=True)
            df_feat = build_features(df_live)

            # Previsão e indicadores
            window    = scaler.transform(df_feat[features].values)[-FEATURE_WINDOW:]
            p         = model.predict(window[np.newaxis,:,:], verbose=0)[0,0]
            price     = df_feat['close'].iloc[-1]
            sentiment = df_feat['sentiment'].iloc[-1]
            stress    = df_feat['vol'].iloc[-1] * 100
            ema_ok    = df_feat['close'].iloc[-1] > df_feat['ema_20'].iloc[-1]
            rsi_ok    = df_feat['rsi14'].iloc[-1] < 70

            # Print dos indicadores
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            print("\n[Bot]", now)
            print(f"Signal:     {p*100:5.1f}%")
            print(f"Sentiment:  {sentiment:5.1f}%")
            print(f"Stress:     {stress:5.1f}%")
            print(f"EMA OK:     {ema_ok}")
            print(f"RSI OK:     {rsi_ok}")

            # Decisão de compra/venda
            action = "HOLD"
            if p > 0.8 and ema_ok and rsi_ok and USD_BALANCE > 0:
                qty = (USD_BALANCE * min(p, 0.5)) / price
                USD_BALANCE -= qty * price
                POSITION     += qty
                action = f"BUY  {qty:.5f} BTC @ {price:.2f}"
                trade_log.append((now, "BUY", f"{qty:.5f}", f"{price:.2f}", ""))
            elif p < 0.2 and POSITION > 0:
                USD_BALANCE += POSITION * price
                entry_price = float(trade_log[-1][3]) if trade_log else price
                pl = f"{((price / entry_price - 1)*100):.2f}%"
                action = f"SELL {POSITION:.5f} BTC @ {price:.2f} (P/L {pl})"
                trade_log.append((now, "SELL", f"{POSITION:.5f}", f"{price:.2f}", pl))
                POSITION = 0
            print("Action:     ", action)

            # Saldo e PnL
            total = USD_BALANCE + POSITION * price
            pnl   = (total - INITIAL_USD) / INITIAL_USD * 100
            print(f"USD Bal:    {USD_BALANCE:8.2f}")
            print(f"BTC Pos:    {POSITION:8.5f}")
            print(f"Total Val:  {total:8.2f} (PnL {pnl:6.2f}%)")

            # Exibe últimos trades
            print("Recent Trades:")
            for t in trade_log:
                print(" ", *t)

            time.sleep(60)

        except KeyboardInterrupt:
            print("Encerrando...")
            break
        except BinanceAPIException as e:
            print("[Erro BinanceAPI]", e)
            time.sleep(5)
        except Exception as e:
            print("[Erro geral]", e)
            time.sleep(5)

# ---------------------------------------------------------------------
# Fluxo Principal
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("📈 Preparando dados e modelo...")
    df_hist = fetch_klines(SYMBOL, TIMEFRAMES[1], lookback_days=365)
    df_feat = label_data(build_features(df_hist))
    model, scaler, features = train_model(df_feat)
    backtest(df_feat, model, scaler, features)
    run_paper_trading(model, scaler, features)
