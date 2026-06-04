"""
SWING DETECTOR
DeclanEngine - Sprint 1 | Structure Engine

Detecta swing highs y swing lows usando lógica fractal.
Un swing high válido: su high es mayor que los N candles de cada lado.
Un swing low válido: su low es menor que los N candles de cada lado.

Principio institucional:
Los swings no se detectan individualmente — se interpretan en contexto
de impulso y corrección. Un swing solo importa si tiene desplazamiento.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class SwingDetector:
    """
    Detecta swings significativos en datos OHLC.
    
    Parameters
    ----------
    left_bars  : candles requeridos a la izquierda del swing
    right_bars : candles requeridos a la derecha del swing (confirmación)
    """

    def __init__(self, left_bars: int = 3, right_bars: int = 3):
        self.left_bars = left_bars
        self.right_bars = right_bars

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta swings en el DataFrame OHLC.

        Parameters
        ----------
        df : DataFrame con columnas ['open', 'high', 'low', 'close']

        Returns
        -------
        DataFrame original con columnas adicionales:
            swing_high : True si el candle es un swing high confirmado
            swing_low  : True si el candle es un swing low confirmado
            swing_high_price : precio del swing high (NaN si no aplica)
            swing_low_price  : precio del swing low (NaN si no aplica)
        """
        df = df.copy()
        df['swing_high'] = False
        df['swing_low'] = False
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan

        highs = df['high'].values
        lows = df['low'].values
        n = len(df)

        for i in range(self.left_bars, n - self.right_bars):
            # --- Swing High ---
            left_highs = highs[i - self.left_bars:i]
            right_highs = highs[i + 1:i + self.right_bars + 1]

            if highs[i] > left_highs.max() and highs[i] > right_highs.max():
                df.iloc[i, df.columns.get_loc('swing_high')] = True
                df.iloc[i, df.columns.get_loc('swing_high_price')] = highs[i]

            # --- Swing Low ---
            left_lows = lows[i - self.left_bars:i]
            right_lows = lows[i + 1:i + self.right_bars + 1]

            if lows[i] < left_lows.min() and lows[i] < right_lows.min():
                df.iloc[i, df.columns.get_loc('swing_low')] = True
                df.iloc[i, df.columns.get_loc('swing_low_price')] = lows[i]

        return df

    def get_swing_list(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retorna listas limpias de swing highs y swing lows detectados.

        Returns
        -------
        (swing_highs_df, swing_lows_df)
        Cada uno con columnas: ['index', 'price']
        """
        result = self.detect(df)

        swing_highs = result[result['swing_high'] == True][['swing_high_price']].copy()
        swing_highs.columns = ['price']
        swing_highs.index.name = 'bar_index'

        swing_lows = result[result['swing_low'] == True][['swing_low_price']].copy()
        swing_lows.columns = ['price']
        swing_lows.index.name = 'bar_index'

        return swing_highs, swing_lows
