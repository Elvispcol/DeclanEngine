"""
SWEEP DETECTOR
DeclanEngine - Sprint 2 | Liquidity Engine

Detecta sweeps de liquidez: violación temporal de un nivel
seguida de rechazo y recuperación rápida.

Principio institucional:
Un sweep NO es simplemente que el precio rompió un nivel.
Un sweep REAL tiene:
  1. Violación del nivel (wick o cierre más allá)
  2. Rechazo inmediato (cierre de regreso al rango)
  3. Candle de absorción/rechazo fuerte

Los sweeps son el setup de entrada de más alta probabilidad
porque representan colección de liquidez antes del movimiento real.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .equal_levels_detector import EqualLevel


@dataclass
class LiquiditySweep:
    """Evento de sweep de liquidez confirmado."""
    bar_index: int
    sweep_type: str          # 'sweep_high' | 'sweep_low'
    level_swept: float       # Precio del nivel barrido
    wick_penetration: float  # Cuánto penetró el wick más allá del nivel
    close_reclaim: bool      # Si el cierre recuperó el nivel
    rejection_body: float    # Ratio cuerpo de rechazo
    sweep_quality: str       # 'premium' | 'standard' | 'weak'
    probable_direction: str  # Dirección probable post-sweep: 'bullish' | 'bearish'
    atr_penetration: float   # Penetración como % del ATR


class SweepDetector:
    """
    Detecta sweeps de liquidez sobre equal highs/lows y swing points.

    Parameters
    ----------
    require_close_reclaim : el cierre debe recuperar el nivel (default True)
    min_penetration_atr   : penetración mínima del wick en ATR (default 0.05)
    max_penetration_atr   : penetración máxima (evitar BOS reales) (default 0.8)
    lookback_candles      : candles para buscar rechazo post-penetración
    """

    def __init__(
        self,
        require_close_reclaim: bool = True,
        min_penetration_atr: float = 0.05,
        max_penetration_atr: float = 0.80,
        lookback_candles: int = 3
    ):
        self.require_close_reclaim = require_close_reclaim
        self.min_penetration_atr = min_penetration_atr
        self.max_penetration_atr = max_penetration_atr
        self.lookback_candles = lookback_candles

    def detect_from_equal_levels(
        self,
        df: pd.DataFrame,
        equal_levels: List[EqualLevel]
    ) -> List[LiquiditySweep]:
        """
        Detecta sweeps sobre zonas de equal highs/lows.
        Solo busca sweeps a partir del último toque de cada nivel.
        """
        sweeps = []
        atr_series = self._compute_atr_series(df)

        for level in equal_levels:
            if level.swept:
                new_sweeps = self._check_sweep(
                    df=df,
                    level_price=level.price,
                    level_type=level.level_type,
                    search_from=level.last_bar + 1,
                    search_to=min(level.sweep_bar + self.lookback_candles + 1, len(df)),
                    atr_series=atr_series
                )
                sweeps.extend(new_sweeps)

        sweeps.sort(key=lambda x: x.bar_index)
        return sweeps

    def detect_from_swings(
        self,
        df: pd.DataFrame,
        swing_highs: pd.DataFrame,
        swing_lows: pd.DataFrame
    ) -> List[LiquiditySweep]:
        """
        Detecta sweeps directamente sobre swing highs/lows individuales.
        """
        sweeps = []
        atr_series = self._compute_atr_series(df)

        # Sweeps sobre swing highs
        for idx, row in swing_highs.iterrows():
            level_price = row['price']
            search_from = int(idx) + 1
            search_to = min(search_from + 30, len(df))

            new_sweeps = self._check_sweep(
                df=df,
                level_price=level_price,
                level_type='equal_high',
                search_from=search_from,
                search_to=search_to,
                atr_series=atr_series
            )
            sweeps.extend(new_sweeps)

        # Sweeps sobre swing lows
        for idx, row in swing_lows.iterrows():
            level_price = row['price']
            search_from = int(idx) + 1
            search_to = min(search_from + 30, len(df))

            new_sweeps = self._check_sweep(
                df=df,
                level_price=level_price,
                level_type='equal_low',
                search_from=search_from,
                search_to=search_to,
                atr_series=atr_series
            )
            sweeps.extend(new_sweeps)

        sweeps.sort(key=lambda x: x.bar_index)
        # Deduplicar por bar_index
        seen = set()
        unique = []
        for s in sweeps:
            if s.bar_index not in seen:
                seen.add(s.bar_index)
                unique.append(s)

        return unique

    def get_recent_sweeps(
        self,
        sweeps: List[LiquiditySweep],
        lookback: int = 20
    ) -> List[LiquiditySweep]:
        if not sweeps:
            return []
        max_bar = max(s.bar_index for s in sweeps)
        return [s for s in sweeps if s.bar_index >= max_bar - lookback]

    def get_summary(self, sweeps: List[LiquiditySweep]) -> dict:
        if not sweeps:
            return {'total': 0, 'premium': 0, 'last': None}

        premium = [s for s in sweeps if s.sweep_quality == 'premium']
        return {
            'total': len(sweeps),
            'premium': len(premium),
            'sweep_highs': sum(1 for s in sweeps if s.sweep_type == 'sweep_high'),
            'sweep_lows': sum(1 for s in sweeps if s.sweep_type == 'sweep_low'),
            'last': sweeps[-1]
        }

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _check_sweep(
        self,
        df: pd.DataFrame,
        level_price: float,
        level_type: str,
        search_from: int,
        search_to: int,
        atr_series: np.ndarray
    ) -> List[LiquiditySweep]:
        """Busca un sweep válido en el rango de bars especificado."""
        sweeps = []
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        closes = df['close'].values

        for i in range(search_from, min(search_to, len(df))):
            atr = atr_series[i]
            if atr == 0:
                continue

            if level_type == 'equal_high':
                # Wick viola el nivel por arriba
                penetration = highs[i] - level_price
                if penetration <= 0:
                    continue

                pen_in_atr = penetration / atr
                if pen_in_atr < self.min_penetration_atr:
                    continue
                if pen_in_atr > self.max_penetration_atr:
                    continue

                # Cierre recupera el nivel (rechazo)
                close_reclaim = closes[i] < level_price
                if self.require_close_reclaim and not close_reclaim:
                    continue

                candle_range = highs[i] - lows[i]
                body = abs(closes[i] - opens[i])
                rejection_body = body / candle_range if candle_range > 0 else 0

                quality = self._score_sweep_quality(pen_in_atr, rejection_body, close_reclaim)

                sweeps.append(LiquiditySweep(
                    bar_index=i,
                    sweep_type='sweep_high',
                    level_swept=level_price,
                    wick_penetration=round(penetration, 5),
                    close_reclaim=close_reclaim,
                    rejection_body=round(rejection_body, 3),
                    sweep_quality=quality,
                    probable_direction='bearish',
                    atr_penetration=round(pen_in_atr, 3)
                ))
                break  # Un sweep por nivel

            else:  # equal_low
                penetration = level_price - lows[i]
                if penetration <= 0:
                    continue

                pen_in_atr = penetration / atr
                if pen_in_atr < self.min_penetration_atr:
                    continue
                if pen_in_atr > self.max_penetration_atr:
                    continue

                close_reclaim = closes[i] > level_price
                if self.require_close_reclaim and not close_reclaim:
                    continue

                candle_range = highs[i] - lows[i]
                body = abs(closes[i] - opens[i])
                rejection_body = body / candle_range if candle_range > 0 else 0

                quality = self._score_sweep_quality(pen_in_atr, rejection_body, close_reclaim)

                sweeps.append(LiquiditySweep(
                    bar_index=i,
                    sweep_type='sweep_low',
                    level_swept=level_price,
                    wick_penetration=round(penetration, 5),
                    close_reclaim=close_reclaim,
                    rejection_body=round(rejection_body, 3),
                    sweep_quality=quality,
                    probable_direction='bullish',
                    atr_penetration=round(pen_in_atr, 3)
                ))
                break

        return sweeps

    def _score_sweep_quality(
        self,
        pen_in_atr: float,
        rejection_body: float,
        close_reclaim: bool
    ) -> str:
        score = 0

        # Penetración significativa pero no excesiva (sweet spot: 0.1 - 0.4 ATR)
        if 0.10 <= pen_in_atr <= 0.40:
            score += 2
        elif pen_in_atr < 0.10:
            score += 1

        # Cuerpo de rechazo fuerte
        if rejection_body > 0.65:
            score += 2
        elif rejection_body > 0.45:
            score += 1

        # Cierre reclamado
        if close_reclaim:
            score += 2

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
            atr[i] = (atr[i - 1] * 13 + tr) / 14 if i >= 14 else tr

        first_valid = next((v for v in atr if v > 0), 1.0)
        atr[atr == 0] = first_valid
        return atr
