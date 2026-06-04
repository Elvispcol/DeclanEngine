"""
MARKET STATE MACHINE
DeclanEngine - Decision Engine

Maquina de estados del mercado con scored states y transition modifiers.

8 estados permitidos:
  - Expansión alcista
  - Expansión bajista
  - Acumulación
  - Distribución
  - Compresión
  - Agotamiento
  - Manipulación
  - Transición estructural

Jerarquía de prioridad (expert-defined):
  1. Expansión (ejecutiva — tiene prioridad sobre preparatoria)
  2. Agotamiento
  3. Manipulación
  4. Acumulación / Distribución
  5. Compresión (preparatoria — subordinada a expansión)
  6. Transición
  7. Rango manipulativo
  8. Neutral

Principio institucional:
El mercado NO tiene estados absolutos.
Contiene señales simultáneas que deben resolverse por score dominante
con transition modifiers para capturar combinaciones significativas.

Modelar COMPORTAMIENTO Wyckoff, no Wyckoff literal.
Boom/Crash acelera ciclos, comprime estructuras, distorsiona tiempos.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


# ── State Definitions ──────────────────────────────────────────────

VALID_STATES = [
    'expansion_alcista',
    'expansion_bajista',
    'acumulacion',
    'distribucion',
    'compresion',
    'agotamiento',
    'manipulacion',
    'transicion',
    'rango_manipulativo',
    'neutral',
]

# Priority: lower number = higher priority
STATE_PRIORITY = {
    'expansion_alcista':   1,
    'expansion_bajista':   1,
    'agotamiento':         2,
    'manipulacion':        3,
    'acumulacion':         4,
    'distribucion':        4,
    'compresion':          5,
    'transicion':          6,
    'rango_manipulativo':  7,
    'neutral':             8,
}


@dataclass
class MarketState:
    """Estado completo del mercado en un punto temporal."""
    dominant_state: str           # Estado dominante resuelto
    state_scores: Dict[str, float]  # Scores de todos los estados (0.0 - 1.0)
    confidence: float             # Confianza en el estado dominante (0.0 - 1.0)
    transition_flag: str          # Flag de transición especial (ej: "expansion_pending")
    bias: str                     # 'bullish' | 'bearish' | 'neutral'
    narrative: str                # Narrativa en lenguaje trader
    raw_signals: Dict[str, float] = field(default_factory=dict)  # Signals que alimentaron la FSM


class MarketStateMachine:
    """
    Maquina de estados del mercado con scored states y transition modifiers.

    Parameters
    ----------
    compression_expansion_threshold : score de compresión para activar transition modifier
    exhaustion_override_threshold   : score de agotamiento para override sobre tendencia
    """

    def __init__(
        self,
        compression_expansion_threshold: float = 0.60,
        exhaustion_override_threshold: float = 0.70,
    ):
        self.comp_thresh = compression_expansion_threshold
        self.exhaust_thresh = exhaustion_override_threshold

    def resolve(
        self,
        structure_bias: str,
        structure_score: float,
        last_event_type: str,
        last_event_significance: str,
        has_displacement: bool,
        displacement_direction: str,
        sweep_present: bool,
        sweep_direction: str,
        sweep_quality: str,
        inducement_present: bool,
        inducement_direction: str,
        momentum_state: str,
        momentum_decay_score: float,
        exhaustion_signal: bool,
        pressure_dominant: str,
        pressure_quality: str,
        net_pressure: float,
        compression_active: bool,
        compression_score: float,
        breakout_pending: bool,
        absorption_detected: bool,
        rejection_detected: bool,
        liquidity_score: float,
    ) -> MarketState:
        """
        Resuelve el estado dominante del mercado a partir de todas las señales.

        Returns
        -------
        MarketState con estado dominante, scores, narrativa y flags.
        """
        # ── 1. COMPUTE INDIVIDUAL STATE SCORES ─────────────────────
        scores = {}
        raw = {}

        # --- Expansión alcista ---
        exp_bull = 0.0
        if structure_bias == 'bullish':
            exp_bull += 0.25
        if has_displacement and displacement_direction == 'bullish':
            exp_bull += 0.25
        if momentum_state in ('accelerating', 'stable') and structure_bias == 'bullish':
            exp_bull += 0.20
        if last_event_type == 'BOS' and last_event_significance in ('strong', 'moderate'):
            exp_bull += 0.15
        if pressure_dominant == 'buyers' and pressure_quality in ('strong', 'moderate'):
            exp_bull += 0.15
        if rejection_detected and net_pressure > 0:
            exp_bull += 0.10
        scores['expansion_alcista'] = min(exp_bull, 1.0)

        # --- Expansión bajista ---
        exp_bear = 0.0
        if structure_bias == 'bearish':
            exp_bear += 0.25
        if has_displacement and displacement_direction == 'bearish':
            exp_bear += 0.25
        if momentum_state in ('accelerating', 'stable') and structure_bias == 'bearish':
            exp_bear += 0.20
        if last_event_type == 'BOS' and last_event_significance in ('strong', 'moderate'):
            exp_bear += 0.15
        if pressure_dominant == 'sellers' and pressure_quality in ('strong', 'moderate'):
            exp_bear += 0.15
        if rejection_detected and net_pressure < 0:
            exp_bear += 0.10
        scores['expansion_bajista'] = min(exp_bear, 1.0)

        # --- Acumulación ---
        accum = 0.0
        if compression_active and structure_bias in ('neutral', 'bullish'):
            accum += 0.25
        if absorption_detected:
            accum += 0.30
        if momentum_state in ('stable', 'decaying') and structure_bias != 'bearish':
            accum += 0.20
        if pressure_dominant == 'buyers' and pressure_quality == 'weak':
            accum += 0.10  # Absorción silenciosa
        if liquidity_score > 0.5 and structure_bias != 'bearish':
            accum += 0.15
        scores['acumulacion'] = min(accum, 1.0)

        # --- Distribución ---
        dist = 0.0
        if compression_active and structure_bias in ('neutral', 'bearish'):
            dist += 0.25
        if absorption_detected and structure_bias == 'bearish':
            dist += 0.30
        if momentum_state in ('stable', 'decaying') and structure_bias != 'bullish':
            dist += 0.20
        if pressure_dominant == 'sellers' and pressure_quality == 'weak':
            dist += 0.10
        if last_event_type == 'CHOCH' and last_event_significance in ('weak', 'moderate'):
            dist += 0.15
        scores['distribucion'] = min(dist, 1.0)

        # --- Compresión ---
        comp = 0.0
        if compression_active:
            comp += 0.35 + (compression_score * 0.30)
        if momentum_state in ('stable', 'decaying') and not exhaustion_signal:
            comp += 0.15
        if structure_bias == 'neutral':
            comp += 0.10
        if not has_displacement:
            comp += 0.10
        scores['compresion'] = min(comp, 1.0)

        # --- Agotamiento ---
        exhaust = 0.0
        if exhaustion_signal:
            exhaust += 0.40
        if momentum_state == 'exhausted':
            exhaust += 0.30
        elif momentum_state == 'decaying':
            exhaust += 0.15
        if momentum_decay_score >= 0.5:
            exhaust += momentum_decay_score * 0.20
        if has_displacement and structure_bias != 'neutral':
            # Displacement reciente con agotamiento = posible fin de movimiento
            exhaust += 0.10
        scores['agotamiento'] = min(exhaust, 1.0)

        # --- Manipulación ---
        manip = 0.0
        if inducement_present:
            manip += 0.30
        if sweep_present and sweep_quality in ('weak', 'standard'):
            manip += 0.20
        if structure_bias == 'neutral' and has_displacement:
            manip += 0.15
        if rejection_detected and not has_displacement:
            manip += 0.15
        if last_event_type == 'CHOCH' and last_event_significance == 'weak':
            manip += 0.20
        if compression_active and not breakout_pending:
            manip += 0.10
        scores['manipulacion'] = min(manip, 1.0)

        # --- Transición estructural ---
        trans = 0.0
        if last_event_type == 'CHOCH' and last_event_significance in ('moderate', 'strong'):
            trans += 0.40
        if sweep_present and sweep_quality == 'premium':
            trans += 0.25
        if has_displacement and displacement_direction != structure_bias:
            trans += 0.25
        if momentum_state == 'accelerating' and structure_bias != 'neutral':
            # Momentum acelerando contra tendencia = transición
            if ((displacement_direction == 'bullish' and structure_bias == 'bearish') or
                (displacement_direction == 'bearish' and structure_bias == 'bullish')):
                trans += 0.20
        scores['transicion'] = min(trans, 1.0)

        # --- Rango manipulativo ---
        rango = 0.0
        if structure_bias == 'neutral':
            rango += 0.30
        if not has_displacement and not sweep_present:
            rango += 0.25
        if compression_active and not breakout_pending:
            rango += 0.15
        if pressure_quality == 'weak' and structure_bias == 'neutral':
            rango += 0.20
        if momentum_state in ('stable', 'decaying') and not exhaustion_signal:
            rango += 0.10
        scores['rango_manipulativo'] = min(rango, 1.0)

        # --- Neutral ---
        neutral = 0.0
        if structure_bias == 'neutral':
            neutral += 0.30
        if not has_displacement and not sweep_present and not inducement_present:
            neutral += 0.30
        if momentum_state in ('stable',) and pressure_quality == 'weak':
            neutral += 0.20
        if not compression_active:
            neutral += 0.10
        if not absorption_detected and not rejection_detected:
            neutral += 0.10
        scores['neutral'] = min(neutral, 1.0)

        raw = {
            'structure_bias': structure_bias,
            'structure_score': structure_score,
            'momentum_state': momentum_state,
            'compression_active': compression_active,
            'breakout_pending': breakout_pending,
            'exhaustion_signal': exhaustion_signal,
        }

        # ── 2. TRANSITION MODIFIERS ────────────────────────────────
        # Compresión + momentum accelerating = expansión inminente
        if (compression_active and compression_score >= self.comp_thresh
                and momentum_state == 'accelerating'):
            # Boost expansión en dirección del momentum/estructura
            if structure_bias == 'bullish':
                scores['expansion_alcista'] = min(scores['expansion_alcista'] + 0.35, 1.0)
            elif structure_bias == 'bearish':
                scores['expansion_bajista'] = min(scores['expansion_bajista'] + 0.35, 1.0)
            else:
                # Sin bias claro: boost ambas expansiones moderadamente
                scores['expansion_alcista'] = min(scores['expansion_alcista'] + 0.15, 1.0)
                scores['expansion_bajista'] = min(scores['expansion_bajista'] + 0.15, 1.0)

        # Breakout pending desde compresión extrema = expansión inminente
        if breakout_pending and compression_active:
            if structure_bias == 'bullish':
                scores['expansion_alcista'] = min(scores['expansion_alcista'] + 0.25, 1.0)
            elif structure_bias == 'bearish':
                scores['expansion_bajista'] = min(scores['expansion_bajista'] + 0.25, 1.0)
            else:
                scores['transicion'] = min(scores['transicion'] + 0.20, 1.0)

        # Sweep premium + CHOCH = transición fuerte
        if sweep_present and sweep_quality == 'premium' and last_event_type == 'CHOCH':
            scores['transicion'] = min(scores['transicion'] + 0.30, 1.0)

        # Agotamiento + sweep = posible reversión
        if exhaustion_signal and sweep_present:
            scores['transicion'] = min(scores['transicion'] + 0.20, 1.0)

        # Agotamiento override: si agotamiento es muy alto, override expansión
        if scores.get('agotamiento', 0) >= self.exhaust_thresh:
            scores['expansion_alcista'] *= 0.5
            scores['expansion_bajista'] *= 0.5

        # Inducement + manipulación = rango manipulativo elevado
        if inducement_present and scores.get('manipulacion', 0) > 0.5:
            scores['rango_manipulativo'] = min(
                scores.get('rango_manipulativo', 0) + 0.20, 1.0
            )

        # ── 3. DOMINANT STATE RESOLUTION ───────────────────────────
        # Resolver por score más alto, con desempate por prioridad
        sorted_states = sorted(
            scores.items(),
            key=lambda x: (-x[1], STATE_PRIORITY.get(x[0], 99))
        )
        dominant_state = sorted_states[0][0]
        dominant_score = sorted_states[0][1]

        # ── 4. TRANSITION FLAGS ────────────────────────────────────
        transition_flag = ''
        if (compression_active and compression_score >= self.comp_thresh
                and momentum_state == 'accelerating'):
            direction = 'alcista' if structure_bias == 'bullish' else (
                'bajista' if structure_bias == 'bearish' else '')
            transition_flag = f"expansion_{direction}_pending" if direction else "expansion_pending"
        elif breakout_pending and compression_active:
            transition_flag = "breakout_imminent"
        elif exhaustion_signal and sweep_present:
            transition_flag = "potential_reversal"
        elif inducement_present and scores.get('manipulacion', 0) > 0.4:
            transition_flag = "manipulation_active"

        # ── 5. BIAS ────────────────────────────────────────────────
        if dominant_state == 'expansion_alcista':
            bias = 'bullish'
        elif dominant_state == 'expansion_bajista':
            bias = 'bearish'
        elif dominant_state in ('acumulacion',):
            bias = 'bullish'  # Acumulación tiende a bullish
        elif dominant_state in ('distribucion',):
            bias = 'bearish'  # Distribución tiende a bearish
        elif transition_flag == 'potential_reversal':
            # Reversión: bias opuesto a la estructura actual
            bias = 'bearish' if structure_bias == 'bullish' else (
                'bullish' if structure_bias == 'bearish' else 'neutral')
        elif last_event_type == 'CHOCH':
            bias = 'bullish' if displacement_direction == 'bullish' else (
                'bearish' if displacement_direction == 'bearish' else structure_bias)
        else:
            bias = structure_bias

        # ── 6. CONFIDENCE ──────────────────────────────────────────
        # Confianza basada en diferencia entre #1 y #2
        if len(sorted_states) >= 2:
            gap = dominant_score - sorted_states[1][1]
            confidence = min(0.50 + gap, 1.0) if dominant_score > 0.3 else 0.30
        else:
            confidence = 0.50

        # ── 7. NARRATIVE ───────────────────────────────────────────
        narrative = self._generate_narrative(
            dominant_state, bias, transition_flag,
            dominant_score, confidence,
            compression_active, breakout_pending,
            exhaustion_signal, sweep_present,
            inducement_present, absorption_detected,
        )

        return MarketState(
            dominant_state=dominant_state,
            state_scores=scores,
            confidence=round(confidence, 3),
            transition_flag=transition_flag,
            bias=bias,
            narrative=narrative,
            raw_signals=raw,
        )

    def _generate_narrative(
        self,
        dominant_state: str,
        bias: str,
        transition_flag: str,
        dominant_score: float,
        confidence: float,
        compression_active: bool,
        breakout_pending: bool,
        exhaustion_signal: bool,
        sweep_present: bool,
        inducement_present: bool,
        absorption_detected: bool,
    ) -> str:
        """Genera la narrativa del mercado en lenguaje trader."""

        # Base narrative por estado dominante
        state_narratives = {
            'expansion_alcista': 'El mercado mantiene una expansión alcista con continuidad estructural dominante.',
            'expansion_bajista': 'El mercado mantiene una expansión bajista saludable con continuidad estructural dominante.',
            'acumulacion': 'El mercado se encuentra en fase de acumulación con absorción compradora emergente.',
            'distribucion': 'El mercado muestra distribución con presión vendedora institucional latente.',
            'compresion': 'El mercado está en compresión de rango, acumulando energía para movimiento direccional.',
            'agotamiento': 'El movimiento actual muestra señales claras de agotamiento. Posible reversión o pausa.',
            'manipulacion': 'El mercado presenta comportamiento manipulativo. Las señales pueden ser trampas de liquidez.',
            'transicion': 'El mercado está en transición estructural. El control institucional puede estar cambiando.',
            'rango_manipulativo': 'El mercado opera en rango manipulativo sin ventaja direccional clara.',
            'neutral': 'El mercado no presenta condiciones definidas. Sin ventaja operacional identificable.',
        }

        base = state_narratives.get(dominant_state, 'Condición de mercado no clasificada.')

        # Add transition modifier context
        additions = []
        if transition_flag == 'expansion_alcista_pending':
            additions.append('Expansión alcista inminente desde compresión.')
        elif transition_flag == 'expansion_bajista_pending':
            additions.append('Expansión bajista inminente desde compresión.')
        elif transition_flag == 'breakout_imminent':
            additions.append('Breakout inminente por compresión extrema.')
        elif transition_flag == 'potential_reversal':
            additions.append('Posible reversión por agotamiento + sweep.')
        elif transition_flag == 'manipulation_active':
            additions.append('Actividad manipulativa detectada. Precaución extrema.')

        if exhaustion_signal and dominant_state not in ('agotamiento',):
            additions.append('Señales de agotamiento presentes en el movimiento actual.')

        if absorption_detected and dominant_state in ('compresion', 'acumulacion'):
            additions.append('Absorción institucional detectada en zona de liquidez.')

        full_narrative = base
        if additions:
            full_narrative += ' ' + ' '.join(additions)

        return full_narrative
