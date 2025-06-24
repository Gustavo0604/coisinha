import ccxt, time, csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===== CONFIGURAÇÃO =====
exchange = ccxt.binance()
symbol   = 'BTC/USDT'
timeframe= '4h'
limit    = 1000

# Parâmetros finalistas (exemplo: conjunto #1 do walk‐forward)
params = {
    'atr_w':       21,
    'grid_lev':    3,
    'ent_i':       2,
    'ex_i':        2,
    'ma_w':        50,
    'vol_thr_mul': 1.2,
    'vol_vol_mul': 0.8,
    'sl_mul':      1.0,
    'tp_mul':      1.0,
    'cost':        0.0004,
    'slip':        0.0005,
}

# Conta de paper
capital0 = 10_000.0
capital  = capital0
position = 0       # 1 = long, -1 = short, 0 = flat
entry_p   = None

# Arquivo de log
logfile = 'paper_trades.csv'
with open(logfile, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['timestamp','action','price','capital','pnl'])

# Função que decide ações ao final de cada candle
def on_new_candle(df):
    global capital, position, entry_p
    p = params
    df = df.copy()

    # calcula indicadores
    df['tr']  = np.maximum(df.high - df.low,
                  np.maximum((df.high - df.close.shift()).abs(),
                             (df.low  - df.close.shift()).abs()))
    df['atr'] = df.tr.rolling(p['atr_w']).mean()
    df['ma']  = df.close.rolling(p['ma_w']).mean()
    df['atr_m']   = df.atr.rolling(p['atr_w']).mean()
    df['vol_thr'] = df['atr_m'] * p['vol_thr_mul']
    df['vol_m']   = df.volume.rolling(p['atr_w']).mean()
    df['vol_thr_v'] = df['vol_m'] * p['vol_vol_mul']

    # último candle
    ts = df.index[-1]
    price = df.close.iloc[-1]
    atr   = df.atr.iloc[-1]
    ma    = df.ma.iloc[-1]
    vol   = df.volume.iloc[-1]

    # se under filters, nada
    if vol <= df.vol_thr_v.iloc[-1] or atr <= df.vol_thr.iloc[-1]:
        return

    center = df.close.iloc[-2]
    levels = np.linspace(center-atr, center+atr, p['grid_lev']+1)

    buy_p  = price*(1+p['slip'])*(1+p['cost'])
    sell_p = price*(1-p['slip'])*(1-p['cost'])

    action = None
    # ENTRADA LONG
    if position==0 and price > levels[p['ent_i']] and price > ma:
        position = 1
        entry_p  = buy_p
        action   = 'BUY'
    # ENTRADA SHORT
    elif position==0 and price < levels[-p['ent_i']] and price < ma:
        position = -1
        entry_p  = sell_p
        action   = 'SELL'
    # SAÍDA
    elif position != 0:
        slp = entry_p - position*p['sl_mul']*atr
        tpp = entry_p + position*p['tp_mul']*atr
        exit_p = None
        if (position==1 and price>=tpp) or (position==-1 and price<=tpp):
            exit_p = tpp*(1-position*p['slip'])*(1-p['cost'])
            action = 'TP'
        elif (position==1 and price<=slp) or (position==-1 and price>=slp):
            exit_p = slp*(1+position*p['slip'])*(1-p['cost'])
            action = 'SL'
        elif position==1 and price < levels[-p['ex_i']]:
            exit_p = sell_p; action='GRID_EXIT'
        elif position==-1 and price > levels[p['ex_i']]:
            exit_p = buy_p;  action='GRID_EXIT'

        if exit_p is not None:
            pnl = position*(exit_p-entry_p)/entry_p * capital
            capital += pnl
            position = 0

    # Loga no CSV
    if action:
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        pnl  = (capital-capital0) if action=='TP' or action=='SL' or action=='GRID_EXIT' else 0
        with open(logfile, 'a', newline='') as f:
            csv.writer(f).writerow([now, action, price, round(capital,2), round(pnl,2)])
        print(f"[{now}] {action} @ {price:.2f}  Capital: {capital:.2f}  PnL: {pnl:.2f}")

# Inicializa histórico
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
df.set_index('timestamp', inplace=True)

# Loop principal: espera fechamento de cada candle de 4h
print("⏳ Aguarde o próximo fechamento de candle para iniciar o teste real-time...")
while True:
    # calcula próximo timestamp de fechamento
    last = df.index[-1]
    next_close = last + pd.to_timedelta(timeframe)
    sleep_secs = (next_close - datetime.utcnow()).total_seconds()
    if sleep_secs > 0:
        time.sleep(sleep_secs + 1)

    # busca candle novo
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
    new = ohlcv[-1]
    ts  = pd.to_datetime(new[0], unit='ms')
    if ts > df.index[-1]:
        df = df.append(pd.DataFrame([new[1:]], index=[ts], columns=['open','high','low','close','volume']))
        on_new_candle(df)
    else:
        # se já atualizou, espera um intervalo para não spammar API
        time.sleep(30)
