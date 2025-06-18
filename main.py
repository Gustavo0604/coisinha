#!/usr/bin/env python3
import os
import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
import ccxt.async_support as ccxt

# ------------------------
# CONFIGURAÇÕES GERAIS
# ------------------------
SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1m"
FETCH_LIMIT     = 200     # quantos candles buscar a cada iteração
INITIAL_BALANCE = 450.0   # USDT
PROFIT_TARGET   = 0.10    # USD de ganho por trade
STOP_LOSS       = 0.12    # USD de perda por trade
MAX_RETRIES     = 5       # tentativas em caso de timeout
RETRY_DELAY     = 2       # segundos entre tentativas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

async def safe_fetch_ohlcv(exchange, symbol, timeframe, limit):
    """
    Busca OHLCV com retry. Retorna DataFrame ou None em caso de falha.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logging.warning(f"fetch_ohlcv failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(RETRY_DELAY)
    logging.error("fetch_ohlcv failed after max retries")
    return None

async def wait_until_next_minute():
    """
    Aguarda até o fechamento do próximo candle de 1 minuto.
    """
    now = datetime.now(timezone.utc)
    seconds = 60 - now.second - now.microsecond/1e6
    await asyncio.sleep(seconds + 0.1)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula EMAs, RSI14 e Bollinger Bands."""
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100/(1+rs))
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    return df

def detect_trend(df: pd.DataFrame, idx: int) -> str:
    """Detecta tendência via cruzamento EMA20/EMA50."""
    ema20 = df["ema20"].iat[idx]
    ema50 = df["ema50"].iat[idx]
    if ema20 > ema50:
        return "up"
    if ema20 < ema50:
        return "down"
    return "side"

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera sinais de entrada:
      1 = buy ; -1 = sell ; 0 = sem ação
    """
    df["signal"] = 0
    support    = df["low"].rolling(50).min()
    resistance = df["high"].rolling(50).max()
    vol20      = df["volume"].rolling(20).mean()

    for i in range(200, len(df)-1):
        price = df["close"].iat[i]
        rsi   = df["rsi14"].iat[i]
        trend = detect_trend(df, i)

        # Pullback a favor da tendência
        if trend == "up" and df["low"].iat[i] <= df["ema20"].iat[i] and rsi < 30 and df["close"].iat[i] > df["open"].iat[i]:
            df.loc[df.index[i], "signal"] = 1
        if trend == "down" and df["high"].iat[i] >= df["ema20"].iat[i] and rsi > 70 and df["close"].iat[i] < df["open"].iat[i]:
            df.loc[df.index[i], "signal"] = -1

        # Range reversal
        if abs(price - support.iat[i]) < 0.5 and rsi < 20:
            df.loc[df.index[i], "signal"] = 1
        if abs(price - resistance.iat[i]) < 0.5 and rsi > 80:
            df.loc[df.index[i], "signal"] = -1

        # Breakout rápido
        high5 = df["high"].iloc[i-5:i].max()
        low5  = df["low"].iloc[i-5:i].min()
        if df["close"].iat[i] > high5 and df["volume"].iat[i] > vol20.iat[i]:
            df.loc[df.index[i], "signal"] = 1
        if df["close"].iat[i] < low5 and df["volume"].iat[i] > vol20.iat[i]:
            df.loc[df.index[i], "signal"] = -1

    return df

async def main():
    # Credenciais de Testnet Binance (setar BINANCE_APIKEY_TEST e BINANCE_SECRET_TEST)
    api_key = os.getenv("BINANCE_APIKEY_TEST")
    secret  = os.getenv("BINANCE_SECRET_TEST")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
    })
    # ativa sandbox/testnet
    exchange.set_sandbox_mode(True)
    await exchange.load_markets()

    balance  = INITIAL_BALANCE
    position = None  # {"side":"long"/"short", "entry":float}

    logging.info(f"Starting paper trading simulation — initial balance {balance:.2f} USDT")

    try:
        while True:
            await wait_until_next_minute()

            df = await safe_fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, FETCH_LIMIT)
            if df is None or df.empty:
                continue

            df = compute_indicators(df)
            df = generate_signals(df)

            sig       = int(df["signal"].iat[-2])  # sinal do candle fechado
            next_open = df["open"].iat[-1]         # preço de abertura do próximo

            if position is None and sig != 0:
                side      = "long" if sig == 1 else "short"
                position  = {"side": side, "entry": next_open}
                logging.info(f"[ENTRY] {side.upper()} @ {next_open:.2f}")

            if position:
                ticker     = await exchange.fetch_ticker(SYMBOL)
                last_price = float(ticker["last"])
                side       = position["side"]
                entry      = position["entry"]

                if side == "long":
                    if last_price >= entry + PROFIT_TARGET:
                        balance += PROFIT_TARGET
                        logging.info(f"[TP] LONG +{PROFIT_TARGET:.2f} @ {last_price:.2f} → Balance {balance:.2f}")
                        position = None
                    elif last_price <= entry - STOP_LOSS:
                        balance -= STOP_LOSS
                        logging.info(f"[SL] LONG -{STOP_LOSS:.2f} @ {last_price:.2f} → Balance {balance:.2f}")
                        position = None
                else:  # short
                    if last_price <= entry - PROFIT_TARGET:
                        balance += PROFIT_TARGET
                        logging.info(f"[TP] SHORT +{PROFIT_TARGET:.2f} @ {last_price:.2f} → Balance {balance:.2f}")
                        position = None
                    elif last_price >= entry + STOP_LOSS:
                        balance -= STOP_LOSS
                        logging.info(f"[SL] SHORT -{STOP_LOSS:.2f} @ {last_price:.2f} → Balance {balance:.2f}")
                        position = None
            else:
                logging.info(f"No position — balance {balance:.2f} USDT")

    except KeyboardInterrupt:
        logging.info("Simulation stopped by user.")
    finally:
        await exchange.close()
        logging.info(f"Final balance: {balance:.2f} USDT")

if __name__ == "__main__":
    asyncio.run(main())
