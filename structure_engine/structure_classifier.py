"""
STRUCTURE CLASSIFIER
DeclanEngine - Sprint 1 | Structure Engine

Clasifica la estructura de mercado en tiempo real:
  HH  = Higher High  (máximo más alto)
  HL  = Higher Low   (mínimo más alto)
  LH  = Lower High   (máximo más bajo)
  LL  = Lower Low    (mínimo más bajo)

Principio institucional:
La estructura no es perfecta. El mercado manipula. 
Por eso se califica la CALIDAD del swing antes de clasificarlo.
Una estructura solo es válida si tiene desplazamiento real detrás.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StructurePoint:
    """Punto de estructura de mercado."""
    bar_index: int
    price: float
    point_type: str        # 'HH', 'HL', 'LH', 'LL'
    swing_type: str        # 'high' o 'low'
    trend_bias: str        # 'bullish', 'bearish', 'neutral'


class StructureClassifier:
    """
    Clasifica swings detectados como HH/HL/LH/LL y determina
    el bias de tendencia actual.
    
    Parameters
    ----------
    min_swings_for_trend : mínimo de swings para determinar bias
    """

    def __init__(self, min_swings_for_trend: int = 2):
        self.min_swings_for_trend = min_swings_for_trend

    def classify(
        self,
        swing_highs: pd.DataFrame,
        swing_lows: pd.DataFrame
    ) -> List[StructurePoint]:
        """
        Clasifica swings highs y lows en HH/HL/LH/LL.

        Parameters
        ----------
        swing_highs : DataFrame con columna 'price', indexado por bar_index
        swing_lows  : DataFrame con columna 'price', indexado por bar_index

        Returns
        -------
        Lista de StructurePoint ordenada cronológicamente.
        """
        structure_points = []

        # --- Clasificar Swing Highs ---
        high_prices = swing_highs['price'].values
        high_indices = swing_highs.index.values

        for i in range(1, len(high_prices)):
            if high_prices[i] > high_prices[i - 1]:
                point_type = 'HH'
            else:
                point_type = 'LH'

            structure_points.append(StructurePoint(
                bar_index=int(high_indices[i]),
                price=float(high_prices[i]),
                point_type=point_type,
                swing_type='high',
                trend_bias=self._bias_from_high(point_type)
            ))

        # --- Clasificar Swing Lows ---
        low_prices = swing_lows['price'].values
        low_indices = swing_lows.index.values

        for i in range(1, len(low_prices)):
            if low_prices[i] > low_prices[i - 1]:
                point_type = 'HL'
            else:
                point_type = 'LL'

            structure_points.append(StructurePoint(
                bar_index=int(low_indices[i]),
                price=float(low_prices[i]),
                point_type=point_type,
                swing_type='low',
                trend_bias=self._bias_from_low(point_type)
            ))

        # Ordenar cronológicamente
        structure_points.sort(key=lambda x: x.bar_index)

        return structure_points

    def get_current_bias(self, structure_points: List[StructurePoint]) -> str:
        """
        Determina el bias de tendencia actual basado en los últimos swings.

        Returns
        -------
        'bullish' | 'bearish' | 'neutral'
        """
        if len(structure_points) < self.min_swings_for_trend * 2:
            return 'neutral'

        recent = structure_points[-6:]  # Últimos 6 puntos de estructura

        bullish_signals = sum(1 for p in recent if p.point_type in ('HH', 'HL'))
        bearish_signals = sum(1 for p in recent if p.point_type in ('LH', 'LL'))

        if bullish_signals > bearish_signals + 1:
            return 'bullish'
        elif bearish_signals > bullish_signals + 1:
            return 'bearish'
        else:
            return 'neutral'

    def get_structure_summary(self, structure_points: List[StructurePoint]) -> dict:
        """
        Retorna resumen de la estructura actual.
        """
        if not structure_points:
            return {
                'bias': 'neutral',
                'last_point': None,
                'hh_count': 0,
                'hl_count': 0,
                'lh_count': 0,
                'll_count': 0,
                'total_points': 0
            }

        recent = structure_points[-10:]
        counts = {'HH': 0, 'HL': 0, 'LH': 0, 'LL': 0}
        for p in recent:
            counts[p.point_type] += 1

        return {
            'bias': self.get_current_bias(structure_points),
            'last_point': structure_points[-1],
            'hh_count': counts['HH'],
            'hl_count': counts['HL'],
            'lh_count': counts['LH'],
            'll_count': counts['LL'],
            'total_points': len(structure_points)
        }

    @staticmethod
    def _bias_from_high(point_type: str) -> str:
        return 'bullish' if point_type == 'HH' else 'bearish'

    @staticmethod
    def _bias_from_low(point_type: str) -> str:
        return 'bullish' if point_type == 'HL' else 'bearish'
