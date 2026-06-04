"""
CHECK - Verificación de datos reales MT5
python check.py
"""
from mt5_connector import MT5Connector

c = MT5Connector()
if not c.connect():
    exit()

import MetaTrader5 as mt5

# Saldo
info = mt5.account_info()
print(f"\n  CUENTA")
print(f"  Login   : {info.login}")
print(f"  Servidor: {info.server}")
print(f"  Balance : {info.balance:.2f} {info.currency}")
print(f"  Equity  : {info.equity:.2f} {info.currency}")

# Precios actuales
for sym_key, sym_name in [('BOOM1000','Boom 1000 Index'),('CRASH1000','Crash 1000 Index')]:
    tick = mt5.symbol_info_tick(sym_name)
    if tick:
        import datetime
        t = datetime.datetime.fromtimestamp(tick.time).strftime('%H:%M:%S')
        print(f"\n  {sym_key}")
        print(f"  Bid : {tick.bid:.2f}")
        print(f"  Ask : {tick.ask:.2f}")
        print(f"  Hora: {t}")

c.disconnect()
