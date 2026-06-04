"""
PRESSURE ANALYZER
DeclanEngine - Sprint 3 | Candle Pressure Engine

Analiza los últimos N candles para calcular presión compradora/vendedora.

Principio institucional:
Las velas NO se analizan individualmente.
Se interpretan secuencialmente en ventana de 3-5 candles.

Score de presión:
  +1.0 = presión compradora máxima
  -1.0 = presión vendedora máxima
   0.0 = neutral / indecisión
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class PressureReading:
    """Lectura de presión en un bar específico."""
    bar_index: int
    pressure_score: float      # -1.0 a +1.0
    buying_pressure: float     # 0.0 a 1.0
    selling_pressure: float    # 0.0 a 1.0
    body_ratio: float          # cuerpo / rango
    upper_wick_ratio: float    # mecha superior / rango
    lower_wick_ratio: float    # mecha inferior / rango
    candle_direction: str      # 'bullish' | 'bearish' | 'doji'
    absorption_score: float    # 0.0 a 1.0 (absorción institucional)
    rejection_score: float     # 0.0 a 1.0 (rechazo de nivel)
    label: str                 # 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell'


@dataclass
class WindowPressure:
    """Presión acumulada de una ventana de candles."""
    end_bar: int
    window_size: int
    net_pressure: float         # Promedio ponderado presión neta
    dominant_side: str          # 'buyers' | 'sellers' | 'neutral'
    consistency: float          # Qué tan consistente es la presión (0-1)
    absorption_detected: bool
    rejection_detected: bool
    pressure_trend: str         # 'increasing' | 'decreasing' | 'stable'
    quality: str                # 'strong' | 'moderate' | 'weak'


class PressureAnalyzer:
    """
    Calcula presión de velas en ventanas deslizantes.

    Parameters
    ----------
    window_size  : candles a analizar en cada lectura (default 5)
    atr_period   : período ATR para normalizar rangos
    """

    def __init__(self, window_size: int = 5, atr_period: int = 14):
        self.window_size = window_size
        self.atr_period = atr_period

    def analyze_all(self, df: pd.DataFrame) -> List[PressureReading]:
        """Genera una PressureReading por cada candle."""
        readings = []
        atr = self._compute_atr(df)

        opens  = df['open'].values
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values

        for i in range(len(df)):
            r = self._read_candle(i, opens[i], highs[i], lows[i], closes[i], atr[i])
            readings.append(r)

        return readings

    def analyze_window(
        self,
        df: pd.DataFrame,
        readings: List[PressureReading],
        end_bar: int = -1
    ) -> WindowPressure:
        """
        Analiza la presión de la ventana de N candles que termina en end_bar.
        end_bar=-1 usa el último candle disponible.
        """
        if end_bar == -1:
            end_bar = len(readings) - 1

        start = max(0, end_bar - self.window_size + 1)
        window = readings[start:end_bar + 1]

        if not window:
            return WindowPressure(
                end_bar=end_bar, window_size=0,
                net_pressure=0.0, dominant_side='neutral',
                consistency=0.0, absorption_detected=False,
                rejection_detected=False, pressure_trend='stable',
                quality='weak'
            )

        # Presión neta ponderada (candles recientes pesan más)
        weights = np.linspace(0.5, 1.0, len(window))
        scores = np.array([r.pressure_score for r in window])
        net = float(np.average(scores, weights=weights))

        # Consistencia: todos apuntan al mismo lado
        same_side = sum(1 for s in scores if np.sign(s) == np.sign(net))
        consistency = same_side / len(scores)

        dominant = (
            'buyers' if net > 0.15 else
            'sellers' if net < -0.15 else
            'neutral'
        )

        absorption = any(r.absorption_score > 0.6 for r in window)
        rejection  = any(r.rejection_score > 0.6 for r in window)

        # Tendencia de presión (¿aumentando o disminuyendo?)
        if len(window) >= 3:
            first_half = np.mean([r.pressure_score for r in window[:len(window)//2]])
            second_half = np.mean([r.pressure_score for r in window[len(window)//2:]])
            diff = second_half - first_half
            if abs(diff) > 0.15:
                trend = 'increasing' if diff > 0 else 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        quality = self._score_window_quality(abs(net), consistency, absorption or rejection)

        return WindowPressure(
            end_bar=end_bar,
            window_size=len(window),
            net_pressure=round(net, 3),
            dominant_side=dominant,
            consistency=round(consistency, 2),
            absorption_detected=absorption,
            rejection_detected=rejection,
            pressure_trend=trend,
            quality=quality
        )

    def get_current_pressure(
        self,
        df: pd.DataFrame,
        readings: List[PressureReading]
    ) -> WindowPressure:
        """Retorna la presión de la ventana más reciente."""
        return self.analyze_window(df, readings, end_bar=len(readings) - 1)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _read_candle(
        self,
        bar_index: int,
        o: float, h: float, l: float, c: float,
        atr: float
    ) -> PressureReading:
        candle_range = h - l
        if candle_range == 0:
            return PressureReading(
                bar_index=bar_index, pressure_score=0.0,
                buying_pressure=0.5, selling_pressure=0.5,
                body_ratio=0.0, upper_wick_ratio=0.0, lower_wick_ratio=0.0,
                candle_direction='doji', absorption_score=0.0,
                rejection_score=0.0, label='neutral'
            )

        body         = abs(c - o)
        upper_wick   = h - max(o, c)
        lower_wick   = min(o, c) - l
        body_ratio   = body / candle_range
        upper_ratio  = upper_wick / candle_range
        lower_ratio  = lower_wick / candle_range

        direction = 'bullish' if c > o else ('bearish' if c < o else 'doji')

        # --- Presión compradora ---
        # Cuerpo alcista + mecha inferior pequeña = presión compradora
        buy_pressure = 0.0
        if direction == 'bullish':
            buy_pressure += body_ratio * 0.6
            buy_pressure += (1 - upper_ratio) * 0.2
            buy_pressure += (1 - lower_ratio) * 0.2
        else:
            buy_pressure += lower_ratio * 0.4   # Lower wick = demanda presente
            buy_pressure += (1 - body_ratio) * 0.2

        buy_pressure = min(buy_pressure, 1.0)

        # --- Presión vendedora ---
        sell_pressure = 0.0
        if direction == 'bearish':
            sell_pressure += body_ratio * 0.6
            sell_pressure += (1 - lower_ratio) * 0.2
            sell_pressure += (1 - upper_ratio) * 0.2
        else:
            sell_pressure += upper_ratio * 0.4
            sell_pressure += (1 - body_ratio) * 0.2

        sell_pressure = min(sell_pressure, 1.0)

        # Score neto
        pressure_score = buy_pressure - sell_pressure

        # Absorción: cuerpo pequeño + rango grande (indecisión en zona de liquidez)
        absorption = 0.0
        if body_ratio < 0.35 and candle_range > atr * 0.8:
            absorption = 1.0 - body_ratio
        elif body_ratio < 0.5:
            absorption = (0.5 - body_ratio) * 0.6

        # Rechazo: mecha larga dominante
        rejection = max(upper_ratio, lower_ratio)
        if rejection > 0.55:
            rejection = min(rejection * 1.3, 1.0)

        label = self._score_label(pressure_score)

        return PressureReading(
            bar_index=bar_index,
            pressure_score=round(pressure_score, 3),
            buying_pressure=round(buy_pressure, 3),
            selling_pressure=round(sell_pressure, 3),
            body_ratio=round(body_ratio, 3),
            upper_wick_ratio=round(upper_ratio, 3),
            lower_wick_ratio=round(lower_ratio, 3),
            candle_direction=direction,
            absorption_score=round(absorption, 3),
            rejection_score=round(rejection, 3),
            label=label
        )

    @staticmethod
    def _score_label(score: float) -> str:
        if score >  0.50: return 'strong_buy'
        if score >  0.20: return 'buy'
        if score < -0.50: return 'strong_sell'
        if score < -0.20: return 'sell'
        return 'neutral'

    @staticmethod
    def _score_window_quality(net: float, consistency: float, event: bool) -> str:
        score = net * 0.5 + consistency * 0.3 + (0.2 if event else 0.0)
        if score > 0.55: return 'strong'
        if score > 0.35: return 'moderate'
        return 'weak'

    def _compute_atr(self, df: pd.DataFrame) -> np.ndarray:
        h, l, c = df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        atr = np.zeros(n)
        for i in range(1, n):
            tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
            atr[i] = (atr[i-1]*13 + tr)/14 if i >= 14 else tr
        first = next((v for v in atr if v > 0), 1.0)
        atr[atr == 0] = first
        return atr
