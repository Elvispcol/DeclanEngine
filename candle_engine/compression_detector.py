"""
COMPRESSION DETECTOR
DeclanEngine - Sprint 3 | Candle Pressure Engine

Detecta compresión de rango: volatilidad reduciéndose progresivamente,
señal de acumulación de energía antes de expansión violenta.

Principio institucional:
La compresión precede frecuentemente a movimientos explosivos.
No es señal de dirección — es señal de PREPARACIÓN.
El motor debe elevar la alerta cuando la compresión es extrema.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class CompressionState:
    """Estado de compresión en un punto del mercado."""
    bar_index: int
    is_compressed: bool
    compression_ratio: float   # rango actual / rango histórico (< 1 = comprimido)
    compression_score: float   # 0.0 a 1.0 (1.0 = máxima compresión)
    bars_in_compression: int   # cuántos bars consecutivos en compresión
    breakout_pending: bool     # compresión extrema = breakout inminente
    quality: str               # 'extreme' | 'moderate' | 'mild'


class CompressionDetector:
    """
    Detecta fases de compresión de rango.

    Parameters
    ----------
    window          : bars para calcular rango histórico de referencia
    compress_thresh : compression_ratio mínimo para declarar compresión
    extreme_thresh  : umbral para compresión extrema (breakout inminente)
    """

    def __init__(
        self,
        window: int = 20,
        compress_thresh: float = 0.65,
        extreme_thresh: float = 0.35
    ):
        self.window = window
        self.compress_thresh = compress_thresh
        self.extreme_thresh = extreme_thresh

    def detect(self, df: pd.DataFrame) -> List[CompressionState]:
        states = []
        highs  = df['high'].values
        lows   = df['low'].values
        ranges = highs - lows

        consecutive = 0

        for i in range(self.window, len(df)):
            hist_range   = np.mean(ranges[i - self.window:i])
            current_mean = np.mean(ranges[max(0, i-3):i+1])  # media últimos 3

            ratio = current_mean / hist_range if hist_range > 0 else 1.0

            is_compressed = ratio <= self.compress_thresh

            if is_compressed:
                consecutive += 1
            else:
                consecutive = 0

            score = max(0.0, 1.0 - (ratio / self.compress_thresh))
            score = min(score, 1.0)

            breakout_pending = ratio <= self.extreme_thresh or consecutive >= 8

            if ratio <= self.extreme_thresh:
                quality = 'extreme'
            elif ratio <= self.compress_thresh:
                quality = 'moderate'
            else:
                quality = 'mild'

            states.append(CompressionState(
                bar_index=i,
                is_compressed=is_compressed,
                compression_ratio=round(ratio, 3),
                compression_score=round(score, 3),
                bars_in_compression=consecutive,
                breakout_pending=breakout_pending,
                quality=quality
            ))

        return states

    def get_current(self, states: List[CompressionState]) -> CompressionState:
        return states[-1] if states else None

    def get_summary(self, states: List[CompressionState]) -> dict:
        if not states:
            return {'compressed': False, 'quality': 'mild', 'bars': 0}
        current = states[-1]
        return {
            'compressed': current.is_compressed,
            'score': current.compression_score,
            'quality': current.quality,
            'bars': current.bars_in_compression,
            'breakout_pending': current.breakout_pending
        }
