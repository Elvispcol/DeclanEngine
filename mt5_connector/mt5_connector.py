"""
MT5 CONNECTOR
DeclanEngine - Sprint 5

Conecta con MetaTrader 5 de Deriv para obtener datos OHLC reales.

Requisito: pip install MetaTrader5
Solo funciona en Windows con MT5 instalado y activo.

Uso:
    connector = MT5Connector()
    if connector.connect():
        df = connector.get_ohlc('BOOM1000', 'M5', n_bars=300)
        connector.disconnect()
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from .symbol_map import SymbolMap

# MT5 solo disponible en Windows — importación condicional
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class MT5Connector:
    """
    Conector con MetaTrader 5 de Deriv.

    Parameters
    ----------
    login    : número de cuenta MT5 (opcional, usa cuenta activa si omitido)
    password : contraseña (opcional)
    server   : servidor del broker (opcional)
    """

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None
    ):
        self.login    = login
        self.password = password
        self.server   = server
        self.connected = False

    def connect(self) -> bool:
        """Inicializa conexión con MT5."""
        if not MT5_AVAILABLE:
            print("[MT5] ❌ MetaTrader5 no instalado.")
            print("[MT5]    Ejecute: pip install MetaTrader5")
            return False

        if not mt5.initialize():
            print(f"[MT5] ❌ No se pudo inicializar MT5. ¿Está abierto?")
            print(f"[MT5]    Error: {mt5.last_error()}")
            return False

        # Login explícito si se proveen credenciales
        if self.login and self.password and self.server:
            ok = mt5.login(self.login, password=self.password, server=self.server)
            if not ok:
                print(f"[MT5] ❌ Login fallido: {mt5.last_error()}")
                mt5.shutdown()
                return False

        info = mt5.account_info()
        if info:
            print(f"[MT5] ✅ Conectado — Cuenta: {info.login} | "
                  f"Servidor: {info.server} | Balance: {info.balance}")
        else:
            print("[MT5] ✅ MT5 inicializado (sin info de cuenta)")

        self.connected = True
        return True

    def disconnect(self):
        """Cierra conexión con MT5."""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            print("[MT5] Desconectado.")

    def get_ohlc(
        self,
        instrument: str,
        timeframe: str = 'M5',
        n_bars: int = 300
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos OHLC reales desde MT5.

        Parameters
        ----------
        instrument : nombre interno ('BOOM1000', 'CRASH1000', etc.)
        timeframe  : 'M1','M5','M15','M30','H1','H4','D1'
        n_bars     : número de barras a obtener

        Returns
        -------
        DataFrame con columnas: ['time','open','high','low','close','tick_volume']
        """
        if not self.connected:
            print("[MT5] No conectado. Llame connect() primero.")
            return None

        symbol = SymbolMap.get_symbol(instrument)
        tf     = SymbolMap.get_timeframe(timeframe)

        # Verificar que el símbolo existe
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            print(f"[MT5] ❌ Símbolo '{symbol}' no encontrado.")
            print(f"[MT5]    Verifique SymbolMap — símbolos disponibles:")
            syms = mt5.symbols_get()
            if syms:
                matching = [s.name for s in syms if 'Boom' in s.name or 'Crash' in s.name]
                for s in matching[:10]:
                    print(f"         {s}")
            return None

        # Asegurar que el símbolo está visible en Market Watch
        if not sym_info.visible:
            mt5.symbol_select(symbol, True)

        # Obtener barras
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

        if rates is None or len(rates) == 0:
            print(f"[MT5] ❌ Sin datos para {symbol} {timeframe}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'tick_volume': 'volume'})
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('time').reset_index(drop=True)

        print(f"[MT5] ✅ {symbol} {timeframe} — {len(df)} barras "
              f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})")

        return df

    def get_multi_timeframe(
        self,
        instrument: str,
        timeframes: list = None,
        n_bars: int = 300
    ) -> dict:
        """
        Obtiene OHLC para múltiples timeframes simultáneamente.

        Returns
        -------
        dict: {'M1': df, 'M5': df, 'M15': df, 'H1': df}
        """
        if timeframes is None:
            timeframes = ['M1', 'M5', 'M15', 'H1']

        result = {}
        for tf in timeframes:
            df = self.get_ohlc(instrument, tf, n_bars)
            if df is not None:
                result[tf] = df

        return result

    def get_current_price(self, instrument: str) -> Optional[dict]:
        """Retorna bid/ask actual del instrumento."""
        if not self.connected:
            return None

        symbol = SymbolMap.get_symbol(instrument)
        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            return None

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'time': datetime.fromtimestamp(tick.time),
            'spread': round(tick.ask - tick.bid, 5)
        }

    def list_available_symbols(self) -> list:
        """Lista todos los símbolos Boom/Crash disponibles en el terminal."""
        if not self.connected or not MT5_AVAILABLE:
            return []

        syms = mt5.symbols_get()
        if not syms:
            return []

        keywords = ['Boom', 'Crash', 'Volatility', 'Step']
        return [s.name for s in syms if any(k in s.name for k in keywords)]
