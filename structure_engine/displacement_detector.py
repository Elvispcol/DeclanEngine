"""
DISPLACEMENT DETECTOR
DeclanEngine - Sprint 1 | Structure Engine

Detecta candles de desplazamiento institucional:
movimientos impulsivos con cuerpo dominante, rango expandido
y mínima mecha opuesta.

Principio institucional:
Un desplazamiento real es la huella del dinero institucional.
No es simplemente un candle grande — es un candle con:
  1. Expansión de rango significativa (vs ATR)
  2. Cuerpo dominante (> 60% del rango)
  3. Mecha opuesta mínima (< 20% del rango)
  4. Aparece en contexto estructural relevante
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class DisplacementCandle:
    """Candle de desplazamiento detectado."""
    bar_index: int
    direction: str           # 'bullish' o 'bearish'
    body_ratio: float        # Ratio cuerpo/rango (0.0 a 1.0)
    range_vs_atr: float      # Rango del candle vs ATR (>1.0 = expansión)
    opposite_wick_ratio: float  # Mecha opuesta / rango (menor = mejor)
    quality: str             # 'premium' | 'standard' | 'weak'
    open: float
    close: float
    high: float
    low: float


class DisplacementDetector:
    """
    Detecta candles de desplazamiento institucional.
    
    Parameters
    ----------
    min_body_ratio         : Ratio mínimo cuerpo/rango (default 0.6)
    min_range_vs_atr       : Expansión mínima de rango vs ATR (default 1.2)
    max_opposite_wick_ratio: Mecha opuesta máxima permitida (default 0.25)
    atr_period             : Período ATR para comparación (default 14)
    """

    def __init__(
        self,
        min_body_ratio: float = 0.6,
        min_range_vs_atr: float = 1.2,
        max_opposite_wick_ratio: float = 0.25,
        atr_period: int = 14
    ):
        self.min_body_ratio = min_body_ratio
        self.min_range_vs_atr = min_range_vs_atr
        self.max_opposite_wick_ratio = max_opposite_wick_ratio
        self.atr_period = atr_period

    def detect(self, df: pd.DataFrame) -> List[DisplacementCandle]:
        """
        Detecta candles de desplazamiento en el DataFrame.

        Parameters
        ----------
        df : DataFrame OHLC con columnas ['open', 'high', 'low', 'close']

        Returns
        -------
        Lista de DisplacementCandle.
        """
        displacements = []
        atr_values = self._compute_atr_series(df)

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        for i in range(self.atr_period, len(df)):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            candle_range = h - l

            if candle_range == 0:
                continue

            body = abs(c - o)
            body_ratio = body / candle_range

            atr = atr_values[i]
            range_vs_atr = candle_range / atr if atr > 0 else 0

            direction = 'bullish' if c > o else 'bearish'

            # Mecha opuesta según dirección
            if direction == 'bullish':
                opposite_wick = o - l  # mecha inferior en candle alcista
            else:
                opposite_wick = h - o  # mecha superior en candle bajista

            opposite_wick_ratio = opposite_wick / candle_range

            # Filtros de desplazamiento
            if (body_ratio >= self.min_body_ratio
                    and range_vs_atr >= self.min_range_vs_atr
                    and opposite_wick_ratio <= self.max_opposite_wick_ratio):

                quality = self._score_quality(body_ratio, range_vs_atr, opposite_wick_ratio)

                displacements.append(DisplacementCandle(
                    bar_index=i,
                    direction=direction,
                    body_ratio=round(body_ratio, 3),
                    range_vs_atr=round(range_vs_atr, 3),
                    opposite_wick_ratio=round(opposite_wick_ratio, 3),
                    quality=quality,
                    open=o,
                    close=c,
                    high=h,
                    low=l
                ))

        return displacements

    def get_recent_displacement(
        self,
        displacements: List[DisplacementCandle],
        lookback: int = 10
    ) -> List[DisplacementCandle]:
        """Retorna desplazamientos dentro de los últimos N candles."""
        if not displacements:
            return []
        max_index = max(d.bar_index for d in displacements)
        return [d for d in displacements if d.bar_index >= max_index - lookback]

    def _score_quality(
        self,
        body_ratio: float,
        range_vs_atr: float,
        opposite_wick_ratio: float
    ) -> str:
        score = 0
        if body_ratio > 0.75:
            score += 2
        elif body_ratio > 0.65:
            score += 1

        if range_vs_atr > 1.8:
            score += 2
        elif range_vs_atr > 1.4:
            score += 1

        if opposite_wick_ratio < 0.10:
            score += 2
        elif opposite_wick_ratio < 0.18:
            score += 1

        if score >= 5:
            return 'premium'
        elif score >= 3:
            return 'standard'
        else:
            return 'weak'

    def _compute_atr_series(self, df: pd.DataFrame) -> np.ndarray:
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)
        atr = np.zeros(n)

        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            if i < self.atr_period:
                atr[i] = tr
            else:
                atr[i] = (atr[i - 1] * (self.atr_period - 1) + tr) / self.atr_period

        return atr
