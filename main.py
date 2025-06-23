#!/usr/bin/env python3
"""
Async Live Scalping Bot — Paper Trading fiel ao Backtest Intrabar

Após abrir posição no open do candle, faz polling a cada segundo
para detectar TP/SL via high/low intrabar — igual ao backtest.

Requisitos:
    pip install ccxt pandas numpy

Uso:
    python live_scalping_intrabar.py
    Ctrl+C para encerrar.
"""

import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
import ccxt.async_support as ccxt
from ccxt.base.errors import ExchangeNotAvailable

# ------------------------
# CONFIGURAÇÕES GERAIS
# ------------------------
SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1m"
FETCH_LIMIT     = 300
LOOKBACK        = 50
INITIAL_BALANCE = 450.0
PROFIT_TARGET   = 0.10  # USD
STOP_LOSS       = 0.12  # USD
MAX_RETRIES     = 5
RETRY_DELAY     = 2     # segundos
POLL_INTERVAL   = 1     # segundos entre polls intrabar
# ------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

async def create_exchange():
    for cls_name in ('binance', 'binanceus'):
        exchange = getattr(ccxt, cls_name)({'enableRateLimit': True})
        try:
            await exchange.load_markets()
            logging.info(f"Usando exchange: {cls_name}")
            return exchange
        except ExchangeNotAvailable as e:
            logging.warning(f"{cls_name} indisponível ({e}), tentando próximo…")
            await exchange.close()
    raise RuntimeError("Nenhuma exchange disponível")

async def safe_fetch_ohlcv(exchange, symbol, timeframe, limit):
    for i in range(1, MAX_RETRIES+1):
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logging.warning(f"fetch_ohlcv falhou (tentativa {i}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(RETRY_DELAY)
    logging.error("fetch_ohlcv falhou após tentativas")
    return None

async def wait_until_next_minute():
    now = datetime.now(timezone.utc)
    secs = 60 - now.second - now.microsecond/1e6
    await asyncio.sleep(secs + 0.1)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1+rs))
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    return df

def detect_trend(df: pd.DataFrame, idx: int) -> str:
    return "up" if df["ema20"].iat[idx] > df["ema50"].iat[idx] else \
           "down" if df["ema20"].iat[idx] < df["ema50"].iat[idx] else "side"

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["signal"] = 0
    support    = df["low"].rolling(50).min()
    resistance = df["high"].rolling(50).max()
    vol20      = df["volume"].rolling(20).mean()

    for i in range(LOOKBACK, len(df)-1):
        price = df["close"].iat[i]
        rsi   = df["rsi14"].iat[i]
        trend = detect_trend(df, i)

        # Pullback
        if trend=="up" and df["low"].iat[i] <= df["ema20"].iat[i] and rsi<30 and df["close"].iat[i]>df["open"].iat[i]:
            df.at[df.index[i],"signal"] = 1
        if trend=="down" and df["high"].iat[i] >= df["ema20"].iat[i] and rsi>70 and df["close"].iat[i]<df["open"].iat[i]:
            df.at[df.index[i],"signal"] = -1

        # Range reversal
        if abs(price-support.iat[i])<0.5 and rsi<20:
            df.at[df.index[i],"signal"] = 1
        if abs(price-resistance.iat[i])<0.5 and rsi>80:
            df.at[df.index[i],"signal"] = -1

        # Breakout
        high5 = df["high"].iloc[i-5:i].max()
        low5  = df["low"].iloc[i-5:i].min()
        if df["close"].iat[i]>high5 and df["volume"].iat[i]>vol20.iat[i]:
            df.at[df.index[i],"signal"] = 1
        if df["close"].iat[i]<low5 and df["volume"].iat[i]>vol20.iat[i]:
            df.at[df.index[i],"signal"] = -1

    return df

async def intrabar_exit(exchange, entry_price, side):
    """
    Faz polling do candle atual (limit=1) para detectar TP/SL intrabar.
    Retorna o preço de saída.
    """
    while True:
        ohlcv = await exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1)
        ts, o, h, l, c, v = ohlcv[0]
        # intrabar TP/SL
        if side=="long":
            if h >= entry_price + PROFIT_TARGET:
                return entry_price + PROFIT_TARGET
            if l <= entry_price - STOP_LOSS:
                return entry_price - STOP_LOSS
        else:
            if l <= entry_price - PROFIT_TARGET:
                return entry_price - PROFIT_TARGET
            if h >= entry_price + STOP_LOSS:
                return entry_price + STOP_LOSS
        # fim do minuto?
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        if now_ms >= ts + 60_000:
            return c  # fecha ao close do candle
        await asyncio.sleep(POLL_INTERVAL)

async def main():
    exchange = await create_exchange()
    balance = INITIAL_BALANCE
    logging.info(f"Paper trading intrabar — saldo inicial {balance:.2f} USDT")

    try:
        while True:
            await wait_until_next_minute()

            df = await safe_fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, FETCH_LIMIT)
            if df is None or len(df)<LOOKBACK:
                cnt = len(df) if df is not None else 0
                logging.info(f"Acumulando candles ({cnt}/{LOOKBACK})")
                continue

            df = compute_indicators(df)
            df = generate_signals(df)

            sig = int(df["signal"].iat[-2])
            next_open = df["open"].iat[-1]

            # ENTRY
            if sig!=0:
                side = "long" if sig==1 else "short"
                entry = next_open
                logging.info(f"[ENTRY] {side.upper()} @ {entry:.2f}")
                exit_price = await intrabar_exit(exchange, entry, side)

                # calcula PnL
                pnl =  (exit_price - entry) if side=="long" else (entry - exit_price)
                balance += pnl
                tag = "TP" if pnl>0 else "SL" if pnl<0 else "EXIT"
                logging.info(f"[{tag}] {side.upper()} {pnl:+.2f} @ {exit_price:.2f} → Balance {balance:.2f}")

            else:
                logging.info(f"No position — balance {balance:.2f} USDT")

    except KeyboardInterrupt:
        logging.info("Bot parado pelo usuário.")
    finally:
        await exchange.close()
        logging.info(f"Saldo final: {balance:.2f} USDT")

if __name__ == "__main__":
    asyncio.run(main())
