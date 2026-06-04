"""
EQUAL LEVELS DETECTOR
DeclanEngine - Sprint 2 | Liquidity Engine

Detecta equal highs y equal lows — zonas donde el precio ha tocado
el mismo nivel múltiples veces sin romperlo.

Principio institucional (del GPT especialista):
Tolerancia ATR-relativa: abs(high1 - high2) <= ATR(14) * 0.10
No fixed delta. No porcentaje fijo. ATR adapta dinámicamente
a la volatilidad del instrumento y estado actual del mercado.

Zonas de equal levels = pools de liquidez.
El mercado frecuentemente se mueve HACIA estas zonas antes
de iniciar el movimiento real.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class EqualLevel:
    """Zona de equal highs o equal lows detectada."""
    level_type: str          # 'equal_high' | 'equal_low'
    price: float             # Precio promedio del nivel
    touches: int             # Número de toques confirmados
    first_bar: int           # Bar del primer toque
    last_bar: int            # Bar del toque más reciente
    touch_indices: List[int] # Todos los bars donde tocó
    touch_quality: float     # Score 0.0 - 1.0
    quality_label: str       # 'premium' | 'standard' | 'weak'
    swept: bool = False      # Si ya fue barrido (sweep confirmado)
    sweep_bar: int = -1      # Bar donde fue barrido


class EqualLevelsDetector:
    """
    Detecta zonas de equal highs y equal lows usando tolerancia ATR-relativa.

    Parameters
    ----------
    atr_multiplier  : tolerancia = ATR(14) * atr_multiplier (default 0.10)
    min_touches     : mínimo de toques para confirmar nivel (default 2)
    atr_period      : período ATR (default 14)
    min_bars_between: mínimo de bars entre toques para evitar duplicados
    """

    def __init__(
        self,
        atr_multiplier: float = 0.10,
        min_touches: int = 2,
        atr_period: int = 14,
        min_bars_between: int = 3
    ):
        self.atr_multiplier = atr_multiplier
        self.min_touches = min_touches
        self.atr_period = atr_period
        self.min_bars_between = min_bars_between

    def detect(
        self,
        df: pd.DataFrame,
        swing_highs: pd.DataFrame,
        swing_lows: pd.DataFrame
    ) -> List[EqualLevel]:
        """
        Detecta equal highs y equal lows en el DataFrame.

        Parameters
        ----------
        df          : DataFrame OHLC
        swing_highs : Output de SwingDetector
        swing_lows  : Output de SwingDetector

        Returns
        -------
        Lista de EqualLevel ordenada por last_bar.
        """
        atr_series = self._compute_atr_series(df)
        results = []

        # Detectar equal highs
        results += self._find_equal_levels(
            prices=swing_highs['price'].values,
            indices=swing_highs.index.values,
            level_type='equal_high',
            atr_series=atr_series
        )

        # Detectar equal lows
        results += self._find_equal_levels(
            prices=swing_lows['price'].values,
            indices=swing_lows.index.values,
            level_type='equal_low',
            atr_series=atr_series
        )

        # Marcar niveles ya barridos
        results = self._mark_swept(results, df)

        # Ordenar por bar más reciente
        results.sort(key=lambda x: x.last_bar)

        return results

    def get_active_levels(self, levels: List[EqualLevel]) -> List[EqualLevel]:
        """Retorna niveles no barridos (liquidez pendiente de colectar)."""
        return [l for l in levels if not l.swept]

    def get_summary(self, levels: List[EqualLevel]) -> dict:
        active = self.get_active_levels(levels)
        swept = [l for l in levels if l.swept]
        premium = [l for l in active if l.quality_label == 'premium']

        eq_highs = [l for l in active if l.level_type == 'equal_high']
        eq_lows = [l for l in active if l.level_type == 'equal_low']

        return {
            'total': len(levels),
            'active': len(active),
            'swept': len(swept),
            'premium': len(premium),
            'equal_highs_active': len(eq_highs),
            'equal_lows_active': len(eq_lows),
            'nearest_high': max(eq_highs, key=lambda x: x.last_bar) if eq_highs else None,
            'nearest_low': max(eq_lows, key=lambda x: x.last_bar) if eq_lows else None
        }

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _find_equal_levels(
        self,
        prices: np.ndarray,
        indices: np.ndarray,
        level_type: str,
        atr_series: np.ndarray
    ) -> List[EqualLevel]:
        results = []
        used = [False] * len(prices)

        for i in range(len(prices)):
            if used[i]:
                continue

            group_prices = [prices[i]]
            group_indices = [int(indices[i])]
            used[i] = True

            for j in range(i + 1, len(prices)):
                if used[j]:
                    continue

                bar_j = int(indices[j])
                atr_at_j = atr_series[min(bar_j, len(atr_series) - 1)]
                tolerance = atr_at_j * self.atr_multiplier

                # Comparar contra precio promedio del grupo actual
                group_mean = np.mean(group_prices)
                if abs(prices[j] - group_mean) <= tolerance:
                    # Verificar separación mínima entre toques
                    last_idx = group_indices[-1]
                    if bar_j - last_idx >= self.min_bars_between:
                        group_prices.append(prices[j])
                        group_indices.append(bar_j)
                        used[j] = True

            if len(group_prices) >= self.min_touches:
                avg_price = float(np.mean(group_prices))
                quality_score, quality_label = self._score_quality(
                    touches=len(group_prices),
                    indices=group_indices,
                    prices=group_prices,
                    avg_price=avg_price
                )

                results.append(EqualLevel(
                    level_type=level_type,
                    price=round(avg_price, 5),
                    touches=len(group_prices),
                    first_bar=group_indices[0],
                    last_bar=group_indices[-1],
                    touch_indices=group_indices,
                    touch_quality=round(quality_score, 3),
                    quality_label=quality_label
                ))

        return results

    def _mark_swept(self, levels: List[EqualLevel], df: pd.DataFrame) -> List[EqualLevel]:
        """Marca niveles que ya fueron violados por precio (barridos)."""
        highs = df['high'].values
        lows = df['low'].values

        for lvl in levels:
            start_bar = lvl.last_bar + 1
            if start_bar >= len(df):
                continue

            if lvl.level_type == 'equal_high':
                for b in range(start_bar, len(df)):
                    if highs[b] > lvl.price:
                        lvl.swept = True
                        lvl.sweep_bar = b
                        break
            else:
                for b in range(start_bar, len(df)):
                    if lows[b] < lvl.price:
                        lvl.swept = True
                        lvl.sweep_bar = b
                        break

        return levels

    def _score_quality(
        self,
        touches: int,
        indices: List[int],
        prices: List[float],
        avg_price: float
    ) -> tuple:
        score = 0.0

        # Más toques = más liquidez acumulada
        if touches >= 4:
            score += 0.4
        elif touches == 3:
            score += 0.25
        else:
            score += 0.10

        # Spread de precios (cuanto más ajustado, mejor el nivel)
        price_spread = max(prices) - min(prices)
        spread_ratio = price_spread / avg_price if avg_price > 0 else 1
        if spread_ratio < 0.001:
            score += 0.3
        elif spread_ratio < 0.003:
            score += 0.2
        else:
            score += 0.05

        # Distribución temporal de los toques
        if len(indices) >= 2:
            span = indices[-1] - indices[0]
            if span > 20:
                score += 0.3   # Nivel respetado durante mucho tiempo
            elif span > 10:
                score += 0.2
            else:
                score += 0.05

        score = min(score, 1.0)

        if score >= 0.7:
            label = 'premium'
        elif score >= 0.4:
            label = 'standard'
        else:
            label = 'weak'

        return score, label

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

        # Rellenar zeros iniciales con primer ATR válido
        first_valid = next((v for v in atr if v > 0), 1.0)
        atr[atr == 0] = first_valid

        return atr
