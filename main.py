#!/usr/bin/env python3
"""
Async Live Scalping Bot — Paper Trading (Binance US)

Este script:
  • Usa BINANCE US diretamente (sem tentar binance.com)
  • Busca candles 1m a cada minuto
  • Aguarda mínimo de LOOKBACK candles
  • Calcula indicadores, gera sinais, faz paper trades
  • Fecha sempre o exchange no finally para não vazar sessões

Requisitos:
    pip install ccxt pandas numpy

Uso:
    python async_live_scalping_bot.py
    Ctrl+C para encerrar.
"""

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
FETCH_LIMIT     = 300      # >= LOOKBACK
LOOKBACK        = 50       # mínimo de candles antes de operar
INITIAL_BALANCE = 450.0    # USDT
PROFIT_TARGET   = 0.10     # USD de ganho por trade
STOP_LOSS       = 0.12     # USD de perda por trade
MAX_RETRIES     = 5
RETRY_DELAY     = 2        # segundos
# ------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

async def safe_fetch_ohlcv(exchange, symbol, timeframe, limit):
    """OHLCV com retry."""
    for i in range(1, MAX_RETRIES+1):
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe, limit)
            df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df.set_index("timestamp")
        except Exception as e:
            logging.warning(f"fetch_ohlcv falhou ({i}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(RETRY_DELAY)
    logging.error("fetch_ohlcv falhou após máximo de tentativas")
    return None

async def wait_until_next_minute():
    """Espera o próximo candle de 1m fechar."""
    now = datetime.now(timezone.utc)
    sec = 60 - now.second - now.microsecond/1e6
    await asyncio.sleep(sec + 0.05)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_g = gain.rolling(14).mean(); avg_l = loss.rolling(14).mean()
    rs = avg_g/avg_l; df["rsi14"] = 100 - (100/(1+rs))
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]
    return df

def detect_trend(df: pd.DataFrame, i: int) -> str:
    e20, e50 = df["ema20"].iat[i], df["ema50"].iat[i]
    if e20>e50: return "up"
    if e20<e50: return "down"
    return "side"

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["signal"] = 0
    sup = df["low"].rolling(50).min()
    res = df["high"].rolling(50).max()
    vol20 = df["volume"].rolling(20).mean()
    for i in range(LOOKBACK, len(df)-1):
        p = df["close"].iat[i]; rsi = df["rsi14"].iat[i]; tr = detect_trend(df, i)
        # pullback
        if tr=="up" and df["low"].iat[i]<=df["ema20"].iat[i] and rsi<30 and df["close"].iat[i]>df["open"].iat[i]:
            df.loc[df.index[i],"signal"]=1
        if tr=="down" and df["high"].iat[i]>=df["ema20"].iat[i] and rsi>70 and df["close"].iat[i]<df["open"].iat[i]:
            df.loc[df.index[i],"signal"]=-1
        # range reversal
        if abs(p-sup.iat[i])<0.5 and rsi<20:   df.loc[df.index[i],"signal"]=1
        if abs(p-res.iat[i])<0.5 and rsi>80:   df.loc[df.index[i],"signal"]=-1
        # breakout
        h5 = df["high"].iloc[i-5:i].max()
        l5 = df["low"].iloc[i-5:i].min()
        if df["close"].iat[i]>h5 and df["volume"].iat[i]>vol20.iat[i]:
            df.loc[df.index[i],"signal"]=1
        if df["close"].iat[i]<l5 and df["volume"].iat[i]>vol20.iat[i]:
            df.loc[df.index[i],"signal"]=-1
    return df

async def main():
    exchange = ccxt.binanceus({"enableRateLimit": True})
    # carrega mercados e sessions internas
    await exchange.load_markets()

    balance = INITIAL_BALANCE
    position = None

    logging.info(f"Simulação live iniciada — saldo inicial {balance:.2f} USDT")
    try:
        while True:
            await wait_until_next_minute()

            df = await safe_fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, FETCH_LIMIT)
            if df is None or len(df)<LOOKBACK:
                cnt = len(df) if df is not None else 0
                logging.info(f"Aguardando {cnt}/{LOOKBACK} candles...")
                continue

            df = compute_indicators(df)
            df = generate_signals(df)

            sig = int(df["signal"].iat[-2])
            nxt = df["open"].iat[-1]

            if position is None and sig!=0:
                side = "long" if sig==1 else "short"
                position = {"side":side, "entry":nxt}
                logging.info(f"[ENTRY] {side.upper()} @ {nxt:.2f}")

            if position:
                tk = await exchange.fetch_ticker(SYMBOL)
                lp = float(tk["last"]); ent = position["entry"]
                if position["side"]=="long":
                    if lp>=ent+PROFIT_TARGET:
                        balance += PROFIT_TARGET
                        logging.info(f"[TP] LONG +{PROFIT_TARGET:.2f} @ {lp:.2f} → Balanço {balance:.2f}")
                        position=None
                    elif lp<=ent-STOP_LOSS:
                        balance -= STOP_LOSS
                        logging.info(f"[SL] LONG -{STOP_LOSS:.2f} @ {lp:.2f} → Balanço {balance:.2f}")
                        position=None
                else:
                    if lp<=ent-PROFIT_TARGET:
                        balance += PROFIT_TARGET
                        logging.info(f"[TP] SHORT +{PROFIT_TARGET:.2f} @ {lp:.2f} → Balanço {balance:.2f}")
                        position=None
                    elif lp>=ent+STOP_LOSS:
                        balance -= STOP_LOSS
                        logging.info(f"[SL] SHORT -{STOP_LOSS:.2f} @ {lp:.2f} → Balanço {balance:.2f}")
                        position=None
            else:
                logging.info(f"Sem posição — balanço {balance:.2f} USDT")

    except KeyboardInterrupt:
        logging.info("Simulação interrompida pelo usuário.")
    finally:
        # fecha TODAS as conexões internamente e evita warnings
        await exchange.close()
        logging.info(f"Saldo final: {balance:.2f} USDT")

if __name__ == "__main__":
    asyncio.run(main())
