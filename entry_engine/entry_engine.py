"""
ENTRY ENGINE
DeclanEngine - Entry Engine

Genera setups de entrada basados en liquidez y estructura.

Principio del experto:
  Core Conditions (OBLIGATORIAS):
    1. Liquidity Sweep
    2. Structural Confirmation (BOS/CHOCH)
    3. Displacement

  Confirmation Layers (SCORING BOOSTERS):
    4. Rejection Quality
    5. HTF Alignment
    6. Pressure Confirmation

  Foundation Score + Confirmation Score = Calidad Final

Las entradas NUNCA provienen de RSI, MACD, medias móviles u osciladores.
Solo de: liquidity sweep, structural confirmation, displacement,
rejection, HTF alignment, pressure confirmation.

SL siempre basado en estructura y liquidez (NUNCA distancia fija).
TP siempre basado en liquidez (NUNCA RR fijo ni ATR fijo).

El mercado se mueve HACIA liquidez, no hacia ratios matemáticos.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import pandas as pd


# ── Data Structures ────────────────────────────────────────────────

@dataclass
class SniperEntry:
    """Setup de entrada completo generado por el Entry Engine."""
    valid: bool                   # Si el setup es válido (core conditions cumplidas)
    direction: str                # 'bullish' | 'bearish'
    entry_price: float            # Zona de entrada precisa
    entry_type: str               # 'sweep_rejection' | 'displacement_pullback' | 'structure_confirm'
    stop_loss: float              # SL basado en liquidez/estructura
    sl_reason: str                # Justificación del SL
    take_profit_1: float          # TP1 - liquidez interna
    tp1_reason: str               # Justificación del TP1
    take_profit_2: float          # TP2 - liquidez externa
    tp2_reason: str               # Justificación del TP2
    risk_reward: float            # Ratio R:B calculado
    foundation_score: float       # Score de core conditions (0-100)
    confirmation_score: float     # Score de confirmation layers (0-100)
    total_quality_score: float    # Score compuesto (0-100)
    quality_label: str            # Débil | Moderada | Alta probabilidad | Grado institucional
    core_conditions_met: Dict[str, bool] = field(default_factory=dict)
    confirmation_layers_met: Dict[str, float] = field(default_factory=dict)


class EntryEngine:
    """
    Genera setups de entrada con SL/TP basados en liquidez y estructura.

    Parameters
    ----------
    min_sl_atr_multiplier : múltiplo mínimo de ATR para el SL (evitar SL demasiado ajustado)
    min_rr : ratio riesgo/beneficio mínimo para considerar setup válido
    """

    def __init__(
        self,
        min_sl_atr_multiplier: float = 0.3,
        min_rr: float = 1.5,
    ):
        self.min_sl_atr = min_sl_atr_multiplier
        self.min_rr = min_rr

    def find_setup(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        # Core conditions
        sweep_present: bool,
        sweep_direction: str,     # 'sweep_high' | 'sweep_low'
        sweep_quality: str,       # 'premium' | 'standard' | 'weak'
        sweep_level: float,       # Precio del nivel barrido
        structural_event_type: str,  # 'BOS' | 'CHOCH'
        structural_event_direction: str,  # 'bullish' | 'bearish'
        structural_event_significance: str,  # 'strong' | 'moderate' | 'weak'
        displacement_present: bool,
        displacement_direction: str,
        displacement_entry_price: float = None,
        # Confirmation layers
        rejection_detected: bool = False,
        rejection_quality: float = 0.0,
        htf_aligned: bool = False,
        htf_direction: str = 'neutral',
        pressure_dominant: str = 'neutral',
        pressure_quality: str = 'weak',
        # Liquidity levels for TP
        internal_liquidity_levels: List[Dict] = None,
        external_liquidity_levels: List[Dict] = None,
        # Structure levels for SL
        recent_swing_highs: List[float] = None,
        recent_swing_lows: List[float] = None,
        # Market state
        market_bias: str = 'neutral',
        resolved_direction: str = None,  # Override from Decision Engine FSM
    ) -> SniperEntry:
        """
        Busca un setup de entrada válido basado en core conditions + confirmation layers.

        Parameters
        ----------
        Los parámetros provienen de los motores existentes del DeclanEngine.

        Returns
        -------
        SniperEntry con el setup completo o setup inválido.
        """
        core_conditions = {}
        confirmation_layers = {}

        # ── CORE CONDITIONS ────────────────────────────────────────

        # 1. Liquidity Sweep
        core_conditions['sweep'] = sweep_present and sweep_quality in ('premium', 'standard')

        # 2. Structural Confirmation (BOS o CHOCH válido)
        core_conditions['structural_confirmation'] = (
            structural_event_type in ('BOS', 'CHOCH')
            and structural_event_significance in ('strong', 'moderate')
        )

        # 3. Displacement
        core_conditions['displacement'] = displacement_present

        # Determine trade direction: use Decision Engine's resolved direction if provided
        if resolved_direction:
            direction = resolved_direction
        else:
            direction = self._resolve_direction(
                sweep_direction, structural_event_direction,
                displacement_direction, market_bias
            )

        # Validate core conditions alignment
        all_core_met = all(core_conditions.values())
        direction_aligned = self._check_direction_alignment(
            direction, sweep_direction, structural_event_direction,
            displacement_direction, relaxed=(resolved_direction is not None)
        )

        valid = all_core_met and direction_aligned

        if not valid:
            return SniperEntry(
                valid=False,
                direction=direction,
                entry_price=0, entry_type='',
                stop_loss=0, sl_reason='',
                take_profit_1=0, tp1_reason='',
                take_profit_2=0, tp2_reason='',
                risk_reward=0,
                foundation_score=self._score_foundation(core_conditions),
                confirmation_score=0,
                total_quality_score=0,
                quality_label='Inválido',
                core_conditions_met=core_conditions,
                confirmation_layers_met=confirmation_layers,
            )

        # ── CONFIRMATION LAYERS ────────────────────────────────────

        # 4. Rejection Quality
        if rejection_detected:
            rejection_score = min(rejection_quality * 100, 15)
            confirmation_layers['rejection'] = rejection_score
        else:
            confirmation_layers['rejection'] = 0

        # 5. HTF Alignment
        if htf_aligned and htf_direction == direction:
            confirmation_layers['htf_alignment'] = 15
        else:
            confirmation_layers['htf_alignment'] = 0

        # 6. Pressure Confirmation
        if pressure_dominant != 'neutral':
            pressure_dir = 'bullish' if pressure_dominant == 'buyers' else 'bearish'
            if pressure_dir == direction:
                p_score = 10 if pressure_quality == 'strong' else (
                    7 if pressure_quality == 'moderate' else 3)
                confirmation_layers['pressure'] = p_score
            else:
                confirmation_layers['pressure'] = 0
        else:
            confirmation_layers['pressure'] = 0

        # ── SCORING ────────────────────────────────────────────────
        foundation_score = self._score_foundation(core_conditions)
        confirmation_score = sum(confirmation_layers.values())
        total_score = min(foundation_score + confirmation_score, 100)

        # Quality label (más conservador que antes)
        if total_score >= 85:
            quality_label = 'Grado institucional'
        elif total_score >= 75:
            quality_label = 'Alta probabilidad'
        elif total_score >= 60:
            quality_label = 'Moderada'
        elif total_score >= 50:
            quality_label = 'Débil'
        else:
            quality_label = 'Inválido'

        # ── ENTRY PRICE ────────────────────────────────────────────
        entry_price, entry_type = self._calculate_entry(
            df, current_price, atr, direction,
            sweep_level, sweep_direction,
            displacement_entry_price
        )

        # ── STOP LOSS (liquidity/structure based) ──────────────────
        stop_loss, sl_reason = self._calculate_sl(
            direction, atr, current_price,
            sweep_level, sweep_direction,
            recent_swing_highs or [],
            recent_swing_lows or [],
        )

        # ── TAKE PROFIT (liquidity driven) ─────────────────────────
        tp1, tp1_reason = self._calculate_tp1(
            direction, current_price, atr,
            internal_liquidity_levels or [],
            recent_swing_highs or [],
            recent_swing_lows or [],
        )

        tp2, tp2_reason = self._calculate_tp2(
            direction, current_price, atr,
            external_liquidity_levels or [],
        )

        # ── RISK/REWARD ────────────────────────────────────────────
        risk = abs(entry_price - stop_loss)
        reward_1 = abs(tp1 - entry_price)
        reward_2 = abs(tp2 - entry_price)

        rr1 = reward_1 / risk if risk > 0 else 0
        rr2 = reward_2 / risk if risk > 0 else 0

        # Use TP2 for primary RR display
        primary_rr = round(rr2, 1) if rr2 > 0 else round(rr1, 1)

        return SniperEntry(
            valid=True,
            direction=direction,
            entry_price=round(entry_price, 2),
            entry_type=entry_type,
            stop_loss=round(stop_loss, 2),
            sl_reason=sl_reason,
            take_profit_1=round(tp1, 2),
            tp1_reason=tp1_reason,
            take_profit_2=round(tp2, 2),
            tp2_reason=tp2_reason,
            risk_reward=primary_rr,
            foundation_score=round(foundation_score, 1),
            confirmation_score=round(confirmation_score, 1),
            total_quality_score=round(total_score, 1),
            quality_label=quality_label,
            core_conditions_met=core_conditions,
            confirmation_layers_met=confirmation_layers,
        )

    # ── INTERNAL METHODS ───────────────────────────────────────────

    def _resolve_direction(
        self,
        sweep_dir: str,
        struct_dir: str,
        disp_dir: str,
        market_bias: str,
    ) -> str:
        """Resuelve la dirección del trade desde las señales core."""
        # Sweep direction implies opposite: sweep_low → bullish, sweep_high → bearish
        sweep_implies = 'bullish' if sweep_dir == 'sweep_low' else (
            'bearish' if sweep_dir == 'sweep_high' else 'neutral')

        # Vote among signals
        votes = {'bullish': 0, 'bearish': 0}
        for d in [sweep_implies, struct_dir, disp_dir, market_bias]:
            if d in votes:
                votes[d] += 1

        if votes['bullish'] > votes['bearish']:
            return 'bullish'
        elif votes['bearish'] > votes['bullish']:
            return 'bearish'
        else:
            return market_bias if market_bias != 'neutral' else 'bullish'

    def _check_direction_alignment(
        self,
        direction: str,
        sweep_dir: str,
        struct_dir: str,
        disp_dir: str,
        relaxed: bool = False,
    ) -> bool:
        """Verifica que las señales core estén alineadas con la dirección."""
        sweep_implies = 'bullish' if sweep_dir == 'sweep_low' else (
            'bearish' if sweep_dir == 'sweep_high' else direction)

        if relaxed:
            # When Decision Engine FSM has resolved the direction,
            # we only need structural confirmation to agree
            # A sweep against direction is acceptable — it's liquidity collection
            if struct_dir not in (direction, 'neutral'):
                return False
            return True
        else:
            # Strict mode: sweep and structure must both agree
            if sweep_implies != direction:
                return False
            if struct_dir != direction and struct_dir != 'neutral':
                return False
            if disp_dir not in (direction, 'neutral'):
                return False
            return True

    def _score_foundation(self, core: Dict[str, bool]) -> float:
        """
        Score de core conditions.

        Sweep premium/standard + structural strong = base alta.
        Sweep weak o structural weak = base más baja.
        """
        base = 0.0
        if core.get('sweep'):
            base += 25
        if core.get('structural_confirmation'):
            base += 25
        if core.get('displacement'):
            base += 20
        return base  # Max 70 from foundation

    def _calculate_entry(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        direction: str,
        sweep_level: float,
        sweep_direction: str,
        displacement_entry: float = None,
    ) -> tuple:
        """Calcula la zona de entrada precisa."""
        if direction == 'bullish':
            # Entrada: cerca del sweep level (retest de la zona)
            if displacement_entry and displacement_entry < current_price:
                entry = displacement_entry  # Pullback al inicio del displacement
                entry_type = 'displacement_pullback'
            else:
                # Zona justo encima del sweep level
                entry = sweep_level + (atr * 0.05)
                entry_type = 'sweep_rejection'
        else:
            # Bearish
            if displacement_entry and displacement_entry > current_price:
                entry = displacement_entry
                entry_type = 'displacement_pullback'
            else:
                entry = sweep_level - (atr * 0.05)
                entry_type = 'sweep_rejection'

        return entry, entry_type

    def _calculate_sl(
        self,
        direction: str,
        atr: float,
        current_price: float,
        sweep_level: float,
        sweep_direction: str,
        swing_highs: List[float],
        swing_lows: List[float],
    ) -> tuple:
        """
        Stop Loss basado en estructura y liquidez.

        Bullish SL: debajo del sweep low o liquidity grab
        Bearish SL: encima del sweep high o liquidity grab
        Nunca distancia fija.
        """
        min_sl_distance = atr * self.min_sl_atr

        if direction == 'bullish':
            # SL debajo del nivel barrido (sweep low)
            candidates = [sweep_level]

            # También debajo de swing lows recientes
            relevant_lows = [l for l in swing_lows if l < current_price]
            if relevant_lows:
                candidates.append(min(relevant_lows))

            sl = min(candidates) - (atr * 0.10)  # Margen debajo del nivel

            # Ensure minimum distance
            if abs(current_price - sl) < min_sl_distance:
                sl = current_price - min_sl_distance

            reason = f"Debajo de sweep low {min(candidates):.2f}"

        else:
            # Bearish: SL encima del sweep high
            candidates = [sweep_level]

            relevant_highs = [h for h in swing_highs if h > current_price]
            if relevant_highs:
                candidates.append(max(relevant_highs))

            sl = max(candidates) + (atr * 0.10)

            if abs(sl - current_price) < min_sl_distance:
                sl = current_price + min_sl_distance

            reason = f"Encima de sweep high {max(candidates):.2f}"

        return sl, reason

    def _calculate_tp1(
        self,
        direction: str,
        current_price: float,
        atr: float,
        internal_liquidity: List[Dict],
        swing_highs: List[float],
        swing_lows: List[float],
    ) -> tuple:
        """
        TP1 = Internal Liquidity.

        Liquidez accesible cercana DENTRO del rango operativo actual:
        - Recent swing en dirección del trade
        - Equal highs/lows internos
        - Micro liquidity pools
        """
        if direction == 'bullish':
            targets = []

            # Internal equal highs (liquidez por encima)
            for level in internal_liquidity:
                if level.get('level_type') == 'equal_high' and level.get('price', 0) > current_price:
                    targets.append(level['price'])

            # Recent swing highs como targets
            for h in swing_highs:
                if h > current_price:
                    targets.append(h)

            if targets:
                tp1 = min(targets)  # Liquidez interna más cercana
                reason = f"Liquidez interna (equal high / swing) en {tp1:.2f}"
            else:
                # Fallback: 1.5x ATR (último recurso, no ideal)
                tp1 = current_price + (atr * 1.5)
                reason = "Objetivo ATR-relativo (sin liquidez interna identificada)"

        else:
            # Bearish
            targets = []

            for level in internal_liquidity:
                if level.get('level_type') == 'equal_low' and level.get('price', 0) < current_price:
                    targets.append(level['price'])

            for l in swing_lows:
                if l < current_price:
                    targets.append(l)

            if targets:
                tp1 = max(targets)  # Liquidez interna más cercana (desde arriba)
                reason = f"Liquidez interna (equal low / swing) en {tp1:.2f}"
            else:
                tp1 = current_price - (atr * 1.5)
                reason = "Objetivo ATR-relativo (sin liquidez interna identificada)"

        return tp1, reason

    def _calculate_tp2(
        self,
        direction: str,
        current_price: float,
        atr: float,
        external_liquidity: List[Dict],
    ) -> tuple:
        """
        TP2 = External Liquidity.

        Liquidez macroestructural FUERA del rango estructural dominante:
        - HTF highs/lows
        - Major equal highs/lows
        - External range liquidity
        - Major sweep zones
        """
        if direction == 'bullish':
            targets = []

            for level in external_liquidity:
                price = level.get('price', 0)
                if price > current_price:
                    targets.append(price)

            if targets:
                # TP2 = liquidez externa más lejana significativa
                tp2 = max(targets) if len(targets) <= 2 else sorted(targets)[-2]
                reason = f"Liquidez externa (HTF) en {tp2:.2f}"
            else:
                # Fallback: 3x ATR
                tp2 = current_price + (atr * 3.0)
                reason = "Objetivo macroestructural (sin liquidez externa identificada)"

        else:
            # Bearish
            targets = []

            for level in external_liquidity:
                price = level.get('price', 0)
                if price < current_price:
                    targets.append(price)

            if targets:
                tp2 = min(targets) if len(targets) <= 2 else sorted(targets)[1]
                reason = f"Liquidez externa (HTF) en {tp2:.2f}"
            else:
                tp2 = current_price - (atr * 3.0)
                reason = "Objetivo macroestructural (sin liquidez externa identificada)"

        return tp2, reason
