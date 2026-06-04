"""
DECISION ENGINE
DeclanEngine - Decision Engine

Capa principal de decisión que convierte complejidad institucional
en inteligencia operacional actionable para el trader.

ESTE es el módulo que transforma el sistema de una consola de
diagnóstico institucional a un sistema de decisiones para traders.

El Decision Engine:
  1. Sintetiza el Market State (FSM)
  2. Aplica Penalty Engine
  3. Genera Sniper Entries (Entry Engine)
  4. Produce la estructura de salida MANDATORIA:
     - CONTEXTO DEL MERCADO
     - DIRECCIÓN DOMINANTE
     - CONDICIÓN DEL MERCADO
     - ACCIÓN RECOMENDADA
     - SNIPER ENTRY SETUP
     - CALIDAD DEL SETUP
     - ADVERTENCIA

NO expone señales internas directamente.
Traduce todo a lenguaje trader.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .market_state_machine import MarketStateMachine, MarketState
from .output_formatter import OutputFormatter
from penalty_engine.penalty_engine import PenaltyEngine, PenaltyAssessment
from entry_engine.entry_engine import EntryEngine, SniperEntry


# ── Allowed Output Values ──────────────────────────────────────────

DOMINANT_DIRECTIONS = ['ALCISTA', 'BAJISTA', 'TRANSICIÓN', 'NEUTRAL']

MARKET_CONDITIONS = [
    'Tendencia saludable',
    'Expansión agresiva',
    'Compresión',
    'Agotamiento',
    'Rango manipulativo',
    'Posible reversión',
    'Acumulación',
    'Distribución',
]

RECOMMENDED_ACTIONS = [
    'Buscar compras únicamente',
    'Buscar ventas únicamente',
    'Esperar confirmación',
    'Evitar operar',
    'Esperar sweep de liquidez',
    'Esperar retroceso',
    'Mercado sin ventaja clara',
]


@dataclass
class DecisionOutput:
    """Salida completa del Decision Engine — lo que ve el trader."""
    # 1. Market Context
    contexto: str

    # 2. Dominant Direction
    direccion_dominante: str  # ALCISTA | BAJISTA | TRANSICIÓN | NEUTRAL

    # 3. Market Condition
    condicion_mercado: str

    # 4. Recommended Action
    accion_recomendada: str
    accion_detalle: str  # Detalle adicional de la acción

    # 5. Sniper Entry Setup
    setup: Optional[SniperEntry]  # None si no hay setup válido

    # 6. Quality Score
    calidad_setup: str  # "82% — Alta probabilidad"

    # 7. Warning
    advertencia: str

    # Internal (NO mostrado directamente al trader, disponible para debug)
    _market_state: Optional[MarketState] = None
    _penalty_assessment: Optional[PenaltyAssessment] = None
    _adjusted_score: float = 0.0


class DecisionEngine:
    """
    Capa de decisión principal del DeclanEngine.

    Convierte señales internas en decisiones operacionales para el trader.

    Parameters
    ----------
    instrument : nombre del instrumento (BOOM1000, CRASH1000, etc.)
    """

    def __init__(self, instrument: str = 'BOOM1000'):
        self.instrument = instrument
        self.fsm = MarketStateMachine()
        self.penalty = PenaltyEngine()
        self.entry = EntryEngine()
        self.formatter = OutputFormatter()

    def decide(
        self,
        # Probability Engine results
        total_score: float,
        trade_bias: str,
        no_trade_signal: bool,
        # Structure Engine results
        structure_bias: str,
        structure_score: float,
        last_event_type: str,
        last_event_significance: str,
        # Displacement
        has_displacement: bool,
        displacement_direction: str,
        # Liquidity Engine results
        sweep_present: bool,
        sweep_direction: str,
        sweep_quality: str,
        sweep_level: float,
        recent_sweeps: list,
        inducement_present: bool,
        inducement_direction: str,
        # Candle Engine results
        momentum_state: str,
        momentum_decay_score: float,
        exhaustion_signal: bool,
        pressure_dominant: str,
        pressure_quality: str,
        net_pressure: float,
        pressure_consistency: float,
        compression_active: bool,
        compression_score: float,
        breakout_pending: bool,
        absorption_detected: bool,
        rejection_detected: bool,
        rejection_quality: float = 0.0,
        # Liquidity data for entries
        liquidity_score: float = 0.0,
        active_equal_levels: list = None,
        recent_swing_highs: list = None,
        recent_swing_lows: list = None,
        # Price data
        current_price: float = 0.0,
        atr: float = 0.0,
        # DataFrame for entry calculation
        df = None,
        # HTF data (for multi-timeframe)
        htf_bias: str = None,
    ) -> DecisionOutput:
        """
        Produce la decisión completa del motor para el trader.

        Parameters
        ----------
        Todos los parámetros provienen de los motores internos del DeclanEngine.

        Returns
        -------
        DecisionOutput con la estructura mandatoria de salida.
        """
        active_levels = active_equal_levels or []
        swing_highs = recent_swing_highs or []
        swing_lows = recent_swing_lows or []

        # ── 1. RESOLVE MARKET STATE ────────────────────────────────
        market_state = self.fsm.resolve(
            structure_bias=structure_bias,
            structure_score=structure_score,
            last_event_type=last_event_type,
            last_event_significance=last_event_significance,
            has_displacement=has_displacement,
            displacement_direction=displacement_direction,
            sweep_present=sweep_present,
            sweep_direction=sweep_direction,
            sweep_quality=sweep_quality,
            inducement_present=inducement_present,
            inducement_direction=inducement_direction,
            momentum_state=momentum_state,
            momentum_decay_score=momentum_decay_score,
            exhaustion_signal=exhaustion_signal,
            pressure_dominant=pressure_dominant,
            pressure_quality=pressure_quality,
            net_pressure=net_pressure,
            compression_active=compression_active,
            compression_score=compression_score,
            breakout_pending=breakout_pending,
            absorption_detected=absorption_detected,
            rejection_detected=rejection_detected,
            liquidity_score=liquidity_score,
        )

        # ── 2. APPLY PENALTY ENGINE ────────────────────────────────
        penalty_assessment = self.penalty.assess(
            score=total_score,
            structure_bias=structure_bias,
            htf_bias=htf_bias,
            momentum_state=momentum_state,
            pressure_dominant=pressure_dominant,
            pressure_consistency=pressure_consistency,
            sweep_present=sweep_present,
            sweep_quality=sweep_quality,
            sweep_direction=sweep_direction,
            trade_direction=trade_bias,
            compression_active=compression_active,
            breakout_pending=breakout_pending,
            exhaustion_signal=exhaustion_signal,
            displacement_present=has_displacement,
            displacement_direction=displacement_direction,
        )

        adjusted_score = penalty_assessment.adjusted_score

        # ── 3. DETERMINE DIRECTION ─────────────────────────────────
        direccion = self._resolve_direction(market_state, trade_bias)

        # ── 4. DETERMINE MARKET CONDITION ──────────────────────────
        condicion = self._resolve_condition(market_state, momentum_state,
                                             compression_active, breakout_pending)

        # ── 5. DETERMINE RECOMMENDED ACTION ────────────────────────
        accion, accion_detalle = self._resolve_action(
            direccion, condicion, adjusted_score,
            no_trade_signal, market_state,
            sweep_present, compression_active, breakout_pending,
        )

        # ── 6. GENERATE SNIPER ENTRY (if applicable) ──────────────
        setup = None
        if accion in ('Buscar compras únicamente', 'Buscar ventas únicamente'):
            trade_dir = 'bullish' if accion == 'Buscar compras únicamente' else 'bearish'

            # Get the most recent displacement price if available
            disp_entry = None
            if df is not None and len(df) > 0:
                disp_entry = df['close'].iloc[-1]

            setup = self.entry.find_setup(
                df=df,
                current_price=current_price,
                atr=atr,
                sweep_present=sweep_present,
                sweep_direction=sweep_direction,
                sweep_quality=sweep_quality,
                sweep_level=sweep_level,
                structural_event_type=last_event_type,
                structural_event_direction=displacement_direction if displacement_direction else structure_bias,
                structural_event_significance=last_event_significance,
                displacement_present=has_displacement,
                displacement_direction=displacement_direction,
                displacement_entry_price=disp_entry,
                rejection_detected=rejection_detected,
                rejection_quality=rejection_quality,
                htf_aligned=(htf_bias == market_state.bias) if htf_bias else False,
                htf_direction=htf_bias or 'neutral',
                pressure_dominant=pressure_dominant,
                pressure_quality=pressure_quality,
                internal_liquidity_levels=self._format_liquidity_levels(
                    active_levels, current_price, trade_dir, 'internal'
                ),
                external_liquidity_levels=self._format_liquidity_levels(
                    active_levels, current_price, trade_dir, 'external'
                ),
                recent_swing_highs=swing_highs,
                recent_swing_lows=swing_lows,
                market_bias=market_state.bias,
                resolved_direction=trade_dir,  # FSM-resolved direction
            )

            # Invalidate setup if penalty-adjusted score too low
            if setup and setup.valid and adjusted_score < 50:
                setup.valid = False
                setup.quality_label = 'Inválido (score penalizado)'

        # ── 7. QUALITY SCORE ───────────────────────────────────────
        if setup and setup.valid:
            calidad = f"{setup.total_quality_score:.0f}% — {setup.quality_label}"
        else:
            calidad = "N/A — Sin setup válido"

        # ── 8. WARNING ─────────────────────────────────────────────
        advertencia = self._generate_warning(
            market_state, penalty_assessment,
            adjusted_score, no_trade_signal,
            inducement_present, exhaustion_signal,
        )

        # ── 9. CONTEXT ─────────────────────────────────────────────
        contexto = market_state.narrative

        return DecisionOutput(
            contexto=contexto,
            direccion_dominante=direccion,
            condicion_mercado=condicion,
            accion_recomendada=accion,
            accion_detalle=accion_detalle,
            setup=setup,
            calidad_setup=calidad,
            advertencia=advertencia,
            _market_state=market_state,
            _penalty_assessment=penalty_assessment,
            _adjusted_score=adjusted_score,
        )

    # ── INTERNAL METHODS ───────────────────────────────────────────

    def _resolve_direction(self, market_state: MarketState, trade_bias: str) -> str:
        """Traduce el bias interno a dirección dominante trader-facing."""
        if market_state.dominant_state == 'expansion_alcista':
            return 'ALCISTA'
        elif market_state.dominant_state == 'expansion_bajista':
            return 'BAJISTA'
        elif market_state.dominant_state in ('transicion',):
            return 'TRANSICIÓN'
        elif market_state.transition_flag in ('potential_reversal',):
            return 'TRANSICIÓN'
        elif trade_bias == 'bullish':
            return 'ALCISTA'
        elif trade_bias == 'bearish':
            return 'BAJISTA'
        else:
            return 'NEUTRAL'

    def _resolve_condition(
        self,
        market_state: MarketState,
        momentum_state: str,
        compression_active: bool,
        breakout_pending: bool,
    ) -> str:
        """Traduce el estado del mercado a condición trader-facing."""
        state = market_state.dominant_state

        condition_map = {
            'expansion_alcista': 'Tendencia saludable' if momentum_state in ('accelerating', 'stable') else 'Expansión agresiva',
            'expansion_bajista': 'Tendencia saludable' if momentum_state in ('accelerating', 'stable') else 'Expansión agresiva',
            'acumulacion': 'Acumulación',
            'distribucion': 'Distribución',
            'compresion': 'Compresión',
            'agotamiento': 'Agotamiento',
            'manipulacion': 'Rango manipulativo',
            'transicion': 'Posible reversión',
            'rango_manipulativo': 'Rango manipulativo',
            'neutral': 'Rango manipulativo',
        }

        return condition_map.get(state, 'Rango manipulativo')

    def _resolve_action(
        self,
        direction: str,
        condition: str,
        adjusted_score: float,
        no_trade_signal: bool,
        market_state: MarketState,
        sweep_present: bool,
        compression_active: bool,
        breakout_pending: bool,
    ) -> tuple:
        """Determina la acción recomendada para el trader."""
        # No trade conditions
        if no_trade_signal and adjusted_score < 50:
            return 'Evitar operar', 'Condiciones de mercado desfavorables. Score insuficiente.'

        if direction == 'NEUTRAL':
            if compression_active and not breakout_pending:
                return 'Esperar sweep de liquidez', 'Mercado en compresión. Esperar toma de liquidez antes de entrar.'
            return 'Mercado sin ventaja clara', 'Sin dirección dominante. No hay ventaja operacional.'

        if condition == 'Rango manipulativo':
            if market_state.transition_flag == 'manipulation_active':
                return 'Evitar operar', 'Manipulación activa detectada. Las señales pueden ser trampas.'
            return 'Mercado sin ventaja clara', 'Rango manipulativo sin ventaja direccional.'

        if condition == 'Agotamiento':
            return 'Esperar confirmación', 'Movimiento actual agotado. Esperar confirmación de reversión o continuación.'

        if condition == 'Compresión':
            if breakout_pending:
                return f'Esperar sweep de liquidez', 'Compresión extrema. Breakout inminente — esperar sweep de entrada.'
            return 'Esperar confirmación', 'Mercado comprimido. Esperar breakout direccional.'

        if condition == 'Posible reversión':
            return 'Esperar confirmación', 'Posible reversión en curso. Esperar confirmación estructural.'

        # Directional actions
        if direction == 'ALCISTA':
            if adjusted_score >= 60:
                return 'Buscar compras únicamente', 'Estructura alcista dominante. Evitar ventas contra tendencia.'
            else:
                return 'Esperar retroceso', 'Sesgo alcista pero score insuficiente. Esperar mejor zona de entrada.'

        if direction == 'BAJISTA':
            if adjusted_score >= 60:
                return 'Buscar ventas únicamente', 'Estructura bajista dominante. Evitar compras contra tendencia.'
            else:
                return 'Esperar retroceso', 'Sesgo bajista pero score insuficiente. Esperar mejor zona de entrada.'

        if direction == 'TRANSICIÓN':
            return 'Esperar confirmación', 'Mercado en transición. Esperar confirmación de nueva dirección.'

        return 'Mercado sin ventaja clara', 'Sin condiciones claras para operar.'

    def _generate_warning(
        self,
        market_state: MarketState,
        penalty: PenaltyAssessment,
        adjusted_score: float,
        no_trade_signal: bool,
        inducement_present: bool,
        exhaustion_signal: bool,
    ) -> str:
        """Genera advertencia contextual para el trader."""
        warnings = []

        if penalty.total_penalty < -15:
            warnings.append('Múltiples contradicciones detectadas en las señales.')

        if inducement_present:
            warnings.append('Inducement detectado — posibles trampas de liquidez activas.')

        if exhaustion_signal:
            warnings.append('Señal de agotamiento activa — el movimiento puede estar perdiendo fuerza.')

        if market_state.transition_flag == 'manipulation_active':
            warnings.append('Comportamiento manipulativo del mercado. Precaución extrema.')

        if adjusted_score < 50 and not no_trade_signal:
            warnings.append('Score ajustado bajo por penalidades. Calidad del setup dudosa.')

        if market_state.dominant_state in ('expansion_alcista', 'expansion_bajista'):
            if market_state.transition_flag == 'potential_reversal':
                warnings.append('Posible reversión dentro de tendencia dominante.')

        if not warnings:
            if market_state.dominant_state == 'expansion_alcista':
                warnings.append('La estructura alcista sigue intacta. Ventas agresivas continúan siendo vulnerables a barridas de liquidez.')
            elif market_state.dominant_state == 'expansion_bajista':
                warnings.append('La estructura bajista sigue intacta. Compras agresivas continúan siendo vulnerables a barridas de liquidez.')
            else:
                warnings.append('Mantener gestión de riesgo estricta.')

        return ' '.join(warnings)

    def _format_liquidity_levels(
        self,
        active_levels: list,
        current_price: float,
        direction: str,
        level_type: str,
    ) -> list:
        """
        Formatea niveles de liquidez para el Entry Engine.

        Clasificación institutional de liquidez:

        INTERNAL LIQUIDITY (TP1):
        - Liquidez accesible dentro del rango operativo actual
        - Recent swings, equal highs/lows internos, micro pools, imbalance fills
        - Criterio: dentro de ~1.5% del precio actual Y en dirección del trade

        EXTERNAL LIQUIDITY (TP2):
        - Liquidez macroestructural fuera del rango dominante
        - HTF highs/lows, major equal highs/lows, external range, major sweep zones
        - Criterio: más allá del 1.5% del precio actual O niveles marcados como HTF/major
        """
        # Umbral porcentual para distinguir interna de externa
        # Basado en volatilidad típica de synthetic indices
        internal_threshold = abs(current_price) * 0.015  # 1.5% del precio

        result = []
        for level in active_levels:
            if hasattr(level, 'price') and hasattr(level, 'level_type'):
                price = level.price
                ltype = level.level_type
                distance = abs(price - current_price)

                # Levels con atributo de scope (HTF/major = siempre externo)
                is_major = getattr(level, 'is_major', False) or getattr(level, 'htf_level', False)

                if level_type == 'internal':
                    # Internal: dentro del rango operativo y en dirección del trade
                    if direction == 'bullish' and price > current_price:
                        if distance <= internal_threshold and not is_major:
                            result.append({'price': price, 'level_type': ltype})
                    elif direction == 'bearish' and price < current_price:
                        if distance <= internal_threshold and not is_major:
                            result.append({'price': price, 'level_type': ltype})

                else:  # external
                    # External: fuera del rango operativo O marcados como HTF/major
                    if direction == 'bullish' and price > current_price:
                        if distance > internal_threshold or is_major:
                            result.append({'price': price, 'level_type': ltype})
                    elif direction == 'bearish' and price < current_price:
                        if distance > internal_threshold or is_major:
                            result.append({'price': price, 'level_type': ltype})

            elif isinstance(level, dict):
                # Dict format fallback
                price = level.get('price', 0)
                distance = abs(price - current_price)
                is_major = level.get('is_major', False) or level.get('htf_level', False)

                if level_type == 'internal':
                    if direction == 'bullish' and price > current_price and distance <= internal_threshold and not is_major:
                        result.append(level)
                    elif direction == 'bearish' and price < current_price and distance <= internal_threshold and not is_major:
                        result.append(level)
                else:
                    if direction == 'bullish' and price > current_price and (distance > internal_threshold or is_major):
                        result.append(level)
                    elif direction == 'bearish' and price < current_price and (distance > internal_threshold or is_major):
                        result.append(level)

        return result
