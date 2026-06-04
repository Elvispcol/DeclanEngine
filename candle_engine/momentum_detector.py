"""
MOMENTUM DETECTOR
DeclanEngine - Sprint 3 | Candle Pressure Engine

Detecta:
  - Momentum decay: precio sigue moviéndose pero pierde fuerza
  - Momentum acceleration: expansión creciente y consistente
  - Exhaustion: señales de fin de movimiento

Principio institucional:
El momentum decay es una señal crítica de reversión inminente.
No requiere indicadores — se lee en el comportamiento de los candles:
  * impulsos cada vez más cortos
  * mechas de rechazo crecientes
  * cuerpos encogiendo con precio aún moviéndose
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class MomentumState:
    """Estado de momentum en un punto del mercado."""
    bar_index: int
    state: str              # 'accelerating' | 'stable' | 'decaying' | 'exhausted'
    decay_score: float      # 0.0 (ninguno) a 1.0 (exhaustión total)
    impulse_shrink: bool    # Los impulsos están encogiendo
    wick_growth: bool       # Las mechas están creciendo
    body_shrink: bool       # Los cuerpos están encogiendo
    exhaustion_signal: bool # Señal de agotamiento activa
    direction: str          # 'bullish_decay' | 'bearish_decay' | 'neutral'


class MomentumDetector:
    """
    Detecta momentum decay y aceleración en ventana deslizante.

    Parameters
    ----------
    window       : candles para análisis de tendencia de momentum
    decay_thresh : umbral decay_score para declarar decay activo
    """

    def __init__(self, window: int = 6, decay_thresh: float = 0.55):
        self.window = window
        self.decay_thresh = decay_thresh

    def detect(self, df: pd.DataFrame) -> List[MomentumState]:
        """Genera MomentumState para cada bar desde window en adelante."""
        states = []
        atr = self._compute_atr(df)

        opens  = df['open'].values
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values

        for i in range(self.window, len(df)):
            w_o = opens[i - self.window:i + 1]
            w_h = highs[i - self.window:i + 1]
            w_l = lows[i - self.window:i + 1]
            w_c = closes[i - self.window:i + 1]
            atr_i = atr[i]

            state = self._evaluate_window(i, w_o, w_h, w_l, w_c, atr_i)
            states.append(state)

        return states

    def get_current(self, states: List[MomentumState]) -> MomentumState:
        return states[-1] if states else None

    # -----------------------------------------------------------------------

    def _evaluate_window(self, bar_index, o, h, l, c, atr) -> MomentumState:
        n = len(c)
        ranges  = h - l
        bodies  = np.abs(c - o)
        upwicks = h - np.maximum(o, c)
        dnwicks = np.minimum(o, c) - l

        # Dirección dominante de la ventana
        net_move = c[-1] - c[0]
        direction_bias = 'bullish' if net_move > 0 else 'bearish'

        # ---- Impulso encogiendo ----
        # Comparar primera mitad vs segunda mitad del rango de candles
        mid = n // 2
        first_range  = np.mean(ranges[:mid])
        second_range = np.mean(ranges[mid:])
        impulse_shrink = second_range < first_range * 0.80

        # ---- Mechas creciendo ----
        if direction_bias == 'bullish':
            first_wick  = np.mean(upwicks[:mid])
            second_wick = np.mean(upwicks[mid:])
        else:
            first_wick  = np.mean(dnwicks[:mid])
            second_wick = np.mean(dnwicks[mid:])
        wick_growth = second_wick > first_wick * 1.30

        # ---- Cuerpos encogiendo ----
        first_body  = np.mean(bodies[:mid])
        second_body = np.mean(bodies[mid:])
        body_shrink = second_body < first_body * 0.75

        # ---- Decay score ----
        decay = 0.0
        if impulse_shrink: decay += 0.35
        if wick_growth:    decay += 0.30
        if body_shrink:    decay += 0.25

        # Penalizar si último candle es doji/pin bar
        last_body_ratio = bodies[-1] / ranges[-1] if ranges[-1] > 0 else 0
        if last_body_ratio < 0.30:
            decay += 0.10

        decay = min(decay, 1.0)

        # ---- Estado ----
        exhaustion = decay >= 0.75
        if decay >= self.decay_thresh:
            state = 'exhausted' if exhaustion else 'decaying'
            direction = f'{direction_bias}_decay'
        elif second_range > first_range * 1.15 and second_body > first_body * 1.10:
            state = 'accelerating'
            direction = 'neutral'
        else:
            state = 'stable'
            direction = 'neutral'

        return MomentumState(
            bar_index=bar_index,
            state=state,
            decay_score=round(decay, 3),
            impulse_shrink=impulse_shrink,
            wick_growth=wick_growth,
            body_shrink=body_shrink,
            exhaustion_signal=exhaustion,
            direction=direction
        )

    def _compute_atr(self, df: pd.DataFrame) -> np.ndarray:
        h, l, c = df['high'].values, df['low'].values, df['close'].values
        n, atr  = len(df), np.zeros(len(df))
        for i in range(1, n):
            tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
            atr[i] = (atr[i-1]*13+tr)/14 if i >= 14 else tr
        first = next((v for v in atr if v > 0), 1.0)
        atr[atr == 0] = first
        return atr
