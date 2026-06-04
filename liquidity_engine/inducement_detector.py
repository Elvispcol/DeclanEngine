"""
INDUCEMENT DETECTOR
DeclanEngine - Sprint 2 | Liquidity Engine

Detecta inducement: movimiento que parece una ruptura legítima
pero es una trampa diseñada para atraer entradas retail antes
del movimiento institucional real.

Principio institucional:
El inducement ocurre cuando:
  1. Precio hace un BOS/CHOCH aparentemente válido
  2. Atrae a traders en esa dirección
  3. Luego revierte con fuerza, atrapando esas posiciones

Señales de inducement:
  - BOS débil (weak) seguido de reversión
  - Ruptura sin desplazamiento real
  - Breakout en zona de liquidez conocida
  - Momentum deteriorado después del breakout
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from structure_engine.bos_choch_detector import StructureEvent
from structure_engine.displacement_detector import DisplacementCandle


@dataclass
class InducementZone:
    """Zona de inducement detectada."""
    bar_index: int                    # Bar donde se detecta el inducement
    inducement_type: str              # 'false_breakout_high' | 'false_breakout_low'
    trap_level: float                 # Nivel que fue la trampa
    breakout_bar: int                 # Bar del breakout falso
    reversal_bar: int                 # Bar donde revirtió
    trap_quality: str                 # 'premium' | 'standard' | 'weak'
    trapped_direction: str            # Dirección de traders atrapados
    probable_move: str                # Dirección del movimiento real esperado
    bos_significance: str             # Calidad del BOS falso que generó la trampa


class InducementDetector:
    """
    Detecta zonas de inducement y breakout traps.

    Parameters
    ----------
    reversal_window  : bars máximos para confirmar reversión post-breakout
    min_reversal_body: ratio mínimo de cuerpo en candle de reversión
    """

    def __init__(
        self,
        reversal_window: int = 5,
        min_reversal_body: float = 0.45
    ):
        self.reversal_window = reversal_window
        self.min_reversal_body = min_reversal_body

    def detect(
        self,
        df: pd.DataFrame,
        bos_events: List[StructureEvent],
        displacements: List[DisplacementCandle]
    ) -> List[InducementZone]:
        """
        Detecta zonas de inducement buscando BOS débiles que revierten.

        Parameters
        ----------
        df           : DataFrame OHLC
        bos_events   : Output de BOSCHOCHDetector
        displacements: Output de DisplacementDetector

        Returns
        -------
        Lista de InducementZone.
        """
        zones = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values

        disp_indices = {d.bar_index for d in displacements}

        for event in bos_events:
            # Solo buscar inducement en BOS débiles o moderados
            if event.significance == 'strong':
                continue

            bar = event.bar_index
            if bar + self.reversal_window >= len(df):
                continue

            # Buscar reversión en los próximos N candles
            reversal_found = False
            reversal_bar = -1

            for r in range(bar + 1, min(bar + self.reversal_window + 1, len(df))):
                candle_range = highs[r] - lows[r]
                body = abs(closes[r] - opens[r])
                body_ratio = body / candle_range if candle_range > 0 else 0

                if body_ratio < self.min_reversal_body:
                    continue

                # BOS bullish → buscar reversión bajista
                if event.direction == 'bullish':
                    if closes[r] < event.broken_level:
                        reversal_found = True
                        reversal_bar = r
                        break

                # BOS bearish → buscar reversión alcista
                elif event.direction == 'bearish':
                    if closes[r] > event.broken_level:
                        reversal_found = True
                        reversal_bar = r
                        break

            if not reversal_found:
                continue

            # Verificar que NO había desplazamiento real en el breakout
            had_displacement = bar in disp_indices

            # Score de la trampa
            trap_quality = self._score_trap(
                event.significance,
                had_displacement,
                reversal_bar - bar
            )

            inducement_type = (
                'false_breakout_high' if event.direction == 'bullish'
                else 'false_breakout_low'
            )

            zones.append(InducementZone(
                bar_index=reversal_bar,
                inducement_type=inducement_type,
                trap_level=event.broken_level,
                breakout_bar=bar,
                reversal_bar=reversal_bar,
                trap_quality=trap_quality,
                trapped_direction=event.direction,
                probable_move='bearish' if event.direction == 'bullish' else 'bullish',
                bos_significance=event.significance
            ))

        zones.sort(key=lambda x: x.bar_index)
        return zones

    def get_summary(self, zones: List[InducementZone]) -> dict:
        if not zones:
            return {'total': 0, 'premium': 0, 'last': None}

        premium = [z for z in zones if z.trap_quality == 'premium']
        return {
            'total': len(zones),
            'premium': len(premium),
            'false_highs': sum(1 for z in zones if z.inducement_type == 'false_breakout_high'),
            'false_lows': sum(1 for z in zones if z.inducement_type == 'false_breakout_low'),
            'last': zones[-1]
        }

    def _score_trap(
        self,
        bos_significance: str,
        had_displacement: bool,
        bars_to_reversal: int
    ) -> str:
        score = 0

        # BOS débil = mejor trampa
        if bos_significance == 'weak':
            score += 3
        elif bos_significance == 'moderate':
            score += 1

        # Sin desplazamiento = movimiento falso
        if not had_displacement:
            score += 2

        # Reversión rápida = trampa más limpia
        if bars_to_reversal <= 2:
            score += 2
        elif bars_to_reversal <= 4:
            score += 1

        if score >= 5:
            return 'premium'
        elif score >= 3:
            return 'standard'
        else:
            return 'weak'
