"""
BOS / CHOCH DETECTOR
DeclanEngine - Sprint 1 | Structure Engine

BOS  = Break of Structure  → precio cierra más allá del último swing en dirección de tendencia
CHOCH = Change of Character → BOS opuesto a la tendencia actual (posible reversión)

Principio institucional:
No todo BOS importa. Un BOS solo es significativo si tiene:
  1. Desplazamiento real (candle con cuerpo dominante)
  2. Cierre de candle más allá del nivel (no solo wick)
  3. Momentum acompañando el movimiento

Un CHOCH solo indica que el control PUEDE estar cambiando.
No es reversión confirmada. Requiere más confirmación.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .structure_classifier import StructurePoint


@dataclass
class StructureEvent:
    """Evento de estructura detectado (BOS o CHOCH)."""
    bar_index: int
    event_type: str          # 'BOS' o 'CHOCH'
    direction: str           # 'bullish' o 'bearish'
    broken_level: float      # Nivel de swing roto
    close_price: float       # Precio de cierre que confirmó el evento
    significance: str        # 'strong' | 'moderate' | 'weak'
    body_ratio: float        # Ratio cuerpo/rango del candle de ruptura


class BOSCHOCHDetector:
    """
    Detecta eventos de Break of Structure y Change of Character.
    
    Parameters
    ----------
    require_close_beyond : Si True, requiere cierre (no solo wick) más allá del nivel
    min_body_ratio       : Ratio mínimo cuerpo/rango para validar BOS (0.0 a 1.0)
    """

    def __init__(
        self,
        require_close_beyond: bool = True,
        min_body_ratio: float = 0.4
    ):
        self.require_close_beyond = require_close_beyond
        self.min_body_ratio = min_body_ratio

    def detect(
        self,
        df: pd.DataFrame,
        structure_points: List[StructurePoint]
    ) -> List[StructureEvent]:
        """
        Detecta BOS y CHOCH en el DataFrame usando los puntos de estructura.

        Parameters
        ----------
        df               : DataFrame OHLC original
        structure_points : Lista de StructurePoint (output de StructureClassifier)

        Returns
        -------
        Lista de StructureEvent ordenada cronológicamente.
        """
        events = []

        if len(structure_points) < 2:
            return events

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values

        # Separar swing highs y lows de la estructura
        swing_highs = [p for p in structure_points if p.swing_type == 'high']
        swing_lows = [p for p in structure_points if p.swing_type == 'low']

        current_bias = self._compute_initial_bias(structure_points)

        # Iterar por cada candle buscando rupturas
        for i in range(1, len(df)):
            close = closes[i]
            high = highs[i]
            low = lows[i]
            candle_range = high - low
            body = abs(close - opens[i])
            body_ratio = body / candle_range if candle_range > 0 else 0

            # --- Buscar ruptura de swing high ---
            relevant_highs = [p for p in swing_highs if p.bar_index < i]
            if relevant_highs:
                last_swing_high = max(relevant_highs, key=lambda x: x.bar_index)
                level = last_swing_high.price

                broke_above = (
                    close > level if self.require_close_beyond else high > level
                )

                if broke_above and body_ratio >= self.min_body_ratio:
                    event_type = 'BOS' if current_bias == 'bullish' else 'CHOCH'
                    significance = self._score_significance(body_ratio, close - level, df, i)

                    events.append(StructureEvent(
                        bar_index=i,
                        event_type=event_type,
                        direction='bullish',
                        broken_level=level,
                        close_price=close,
                        significance=significance,
                        body_ratio=round(body_ratio, 3)
                    ))

                    if event_type == 'CHOCH':
                        current_bias = 'bullish'

            # --- Buscar ruptura de swing low ---
            relevant_lows = [p for p in swing_lows if p.bar_index < i]
            if relevant_lows:
                last_swing_low = max(relevant_lows, key=lambda x: x.bar_index)
                level = last_swing_low.price

                broke_below = (
                    close < level if self.require_close_beyond else low < level
                )

                if broke_below and body_ratio >= self.min_body_ratio:
                    event_type = 'BOS' if current_bias == 'bearish' else 'CHOCH'
                    significance = self._score_significance(body_ratio, level - close, df, i)

                    events.append(StructureEvent(
                        bar_index=i,
                        event_type=event_type,
                        direction='bearish',
                        broken_level=level,
                        close_price=close,
                        significance=significance,
                        body_ratio=round(body_ratio, 3)
                    ))

                    if event_type == 'CHOCH':
                        current_bias = 'bearish'

        return events

    def get_last_event(self, events: List[StructureEvent]) -> Optional[StructureEvent]:
        return events[-1] if events else None

    def get_events_summary(self, events: List[StructureEvent]) -> dict:
        if not events:
            return {'total': 0, 'bos': 0, 'choch': 0, 'last': None}

        bos_count = sum(1 for e in events if e.event_type == 'BOS')
        choch_count = sum(1 for e in events if e.event_type == 'CHOCH')

        return {
            'total': len(events),
            'bos': bos_count,
            'choch': choch_count,
            'last': self.get_last_event(events)
        }

    def _compute_initial_bias(self, structure_points: List[StructurePoint]) -> str:
        if not structure_points:
            return 'neutral'
        recent = structure_points[:4]
        bullish = sum(1 for p in recent if p.point_type in ('HH', 'HL'))
        bearish = sum(1 for p in recent if p.point_type in ('LH', 'LL'))
        if bullish > bearish:
            return 'bullish'
        elif bearish > bullish:
            return 'bearish'
        return 'neutral'

    def _score_significance(
        self,
        body_ratio: float,
        penetration: float,
        df: pd.DataFrame,
        bar_index: int
    ) -> str:
        """Califica la significancia del BOS/CHOCH."""
        atr = self._compute_atr(df, bar_index, period=14)
        pen_ratio = penetration / atr if atr > 0 else 0

        if body_ratio > 0.7 and pen_ratio > 0.3:
            return 'strong'
        elif body_ratio > 0.5 and pen_ratio > 0.15:
            return 'moderate'
        else:
            return 'weak'

    @staticmethod
    def _compute_atr(df: pd.DataFrame, bar_index: int, period: int = 14) -> float:
        start = max(0, bar_index - period)
        subset = df.iloc[start:bar_index + 1]
        if len(subset) < 2:
            return (df['high'].iloc[bar_index] - df['low'].iloc[bar_index])

        highs = subset['high'].values
        lows = subset['low'].values
        closes = subset['close'].values

        tr_list = []
        for i in range(1, len(subset)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            tr_list.append(tr)

        return float(np.mean(tr_list)) if tr_list else 0.0
