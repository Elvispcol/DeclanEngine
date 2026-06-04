"""
PROBABILITY ENGINE
DeclanEngine - Sprint 4

Combina todos los módulos en un score probabilístico 0-100.

Clasificación (según blueprint):
  50-60 : Weak
  60-75 : Moderate
  75-85 : Strong
  85-95 : Institutional Grade

Principio institucional:
El engine NUNCA piensa en certeza.
Piensa en PROBABILIDAD CONTEXTUAL.
Más alineación → mayor probabilidad.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from .instrument_profiles import InstrumentProfiles


@dataclass
class ProbabilityScore:
    """Score probabilístico completo para un momento de mercado."""
    instrument: str

    # Scores por componente (0.0 - 1.0)
    structure_score:  float = 0.0
    liquidity_score:  float = 0.0
    candle_score:     float = 0.0
    momentum_score:   float = 0.0
    mtf_score:        float = 0.0

    # Score final
    total_score:      float = 0.0      # 0-100
    classification:   str   = 'Weak'
    trade_bias:       str   = 'neutral'  # 'bullish' | 'bearish' | 'neutral'
    environment:      str   = 'low_quality'

    # Flags de calidad
    no_trade_signal:  bool  = False
    reasons:          list  = field(default_factory=list)
    weights_used:     dict  = field(default_factory=dict)


class ProbabilityEngine:
    """
    Calcula score probabilístico combinando todos los módulos.

    Parameters
    ----------
    instrument : nombre del instrumento (para pesos adaptativos)
    min_score  : score mínimo para considerar setup válido
    """

    CLASSIFICATIONS = [
        (85, 'Institutional Grade'),
        (75, 'Strong'),
        (60, 'Moderate'),
        (50, 'Weak'),
        (0,  'No Trade'),
    ]

    def __init__(self, instrument: str = 'BOOM1000', min_score: float = 60.0):
        self.instrument = instrument
        self.min_score = min_score
        self.weights = InstrumentProfiles.get_weights(instrument)

    def calculate(
        self,
        structure_summary:  dict,
        events_summary:     dict,
        liquidity_summary:  dict,
        sweep_summary:      dict,
        inducement_summary: dict,
        window_pressure,
        momentum_state,
        compression_state,
        recent_displacement: list,
        recent_sweeps:       list,
    ) -> ProbabilityScore:

        ps = ProbabilityScore(
            instrument=self.instrument,
            weights_used=self.weights
        )
        reasons = []

        # ── 1. STRUCTURE SCORE ────────────────────────────────────────────
        ss = 0.0
        bias = structure_summary.get('bias', 'neutral')

        if bias != 'neutral':
            ss += 0.35

        last_event = events_summary.get('last')
        if last_event:
            sig_map = {'strong': 0.35, 'moderate': 0.25, 'weak': 0.10}
            ss += sig_map.get(last_event.significance, 0.10)
            if last_event.event_type == 'CHOCH':
                ss += 0.15   # CHOCH = transición de estructura, mayor oportunidad
            elif last_event.event_type == 'BOS':
                ss += 0.10

        if recent_displacement:
            d = recent_displacement[-1]
            disp_map = {'premium': 0.15, 'standard': 0.08, 'weak': 0.03}
            ss += disp_map.get(d.quality, 0.03)

        ps.structure_score = round(min(ss, 1.0), 3)
        if ps.structure_score > 0.6:
            reasons.append(f"Estructura sólida [{bias}]")

        # ── 2. LIQUIDITY SCORE ────────────────────────────────────────────
        ls = 0.0

        if liquidity_summary.get('premium', 0) > 0:
            ls += 0.30
        elif liquidity_summary.get('active', 0) > 0:
            ls += 0.15

        if recent_sweeps:
            s = recent_sweeps[-1]
            sweep_map = {'premium': 0.40, 'standard': 0.25, 'weak': 0.10}
            ls += sweep_map.get(s.sweep_quality, 0.10)
            reasons.append(f"Sweep {s.sweep_type} [{s.sweep_quality}]")

        ind_prem = inducement_summary.get('premium', 0)
        if ind_prem > 0:
            ls += min(ind_prem * 0.10, 0.30)
            reasons.append(f"Inducement premium x{ind_prem}")

        ps.liquidity_score = round(min(ls, 1.0), 3)

        # ── 3. CANDLE SCORE ───────────────────────────────────────────────
        cs = 0.0
        if window_pressure:
            net = abs(window_pressure.net_pressure)
            cs += net * 0.50

            quality_map = {'strong': 0.30, 'moderate': 0.20, 'weak': 0.05}
            cs += quality_map.get(window_pressure.quality, 0.05)

            if window_pressure.absorption_detected:
                cs += 0.10
                reasons.append("Absorción detectada")
            if window_pressure.rejection_detected:
                cs += 0.10
                reasons.append("Rechazo detectado")

        ps.candle_score = round(min(cs, 1.0), 3)

        # ── 4. MOMENTUM SCORE ─────────────────────────────────────────────
        ms = 0.0
        if momentum_state:
            state_map = {
                'accelerating': 0.85,
                'stable':       0.55,
                'decaying':     0.30,
                'exhausted':    0.10,
            }
            ms = state_map.get(momentum_state.state, 0.50)

            if momentum_state.exhaustion_signal:
                ms *= 0.5   # Agotamiento = penalizar fuerte
                reasons.append("⚠ Agotamiento de momentum")
            elif momentum_state.state == 'decaying':
                reasons.append("Momentum en decay")

        ps.momentum_score = round(ms, 3)

        # ── 5. MTF SCORE (simulado en Sprint 4, real en Sprint 5) ─────────
        # Por ahora usa estructura + momentum como proxy de alineación HTF/LTF
        mtf = 0.0
        if bias != 'neutral' and momentum_state:
            bias_match = (
                (bias == 'bullish' and momentum_state.state in ('accelerating', 'stable'))
                or
                (bias == 'bearish' and momentum_state.state in ('accelerating', 'stable'))
            )
            mtf = 0.70 if bias_match else 0.35

        if compression_state and compression_state.get('breakout_pending'):
            mtf = max(mtf, 0.65)
            reasons.append("⚡ Compresión extrema — breakout pendiente")

        ps.mtf_score = round(min(mtf, 1.0), 3)

        # ── TOTAL SCORE ───────────────────────────────────────────────────
        w = self.weights
        raw = (
            ps.structure_score  * w['structure'] +
            ps.liquidity_score  * w['liquidity'] +
            ps.candle_score     * w['candle']    +
            ps.momentum_score   * w['momentum']  +
            ps.mtf_score        * w['mtf']
        )
        ps.total_score = round(raw * 100, 1)

        # ── CLASIFICACIÓN ─────────────────────────────────────────────────
        for threshold, label in self.CLASSIFICATIONS:
            if ps.total_score >= threshold:
                ps.classification = label
                break

        # ── BIAS DE TRADE ─────────────────────────────────────────────────
        if recent_sweeps:
            ps.trade_bias = recent_sweeps[-1].probable_direction
        elif bias != 'neutral':
            ps.trade_bias = bias

        # ── AMBIENTE ─────────────────────────────────────────────────────
        if ps.total_score >= 75:
            ps.environment = 'high_quality'
        elif ps.total_score >= 60:
            ps.environment = 'moderate_quality'
        else:
            ps.environment = 'low_quality'

        # ── NO TRADE SIGNAL ───────────────────────────────────────────────
        ps.no_trade_signal = (
            ps.total_score < self.min_score
            or bias == 'neutral'
            or (momentum_state and momentum_state.exhaustion_signal and ps.liquidity_score < 0.4)
        )

        if ps.no_trade_signal and ps.total_score < self.min_score:
            reasons.append("Score bajo — entorno de baja calidad")

        ps.reasons = reasons
        return ps

    def format_output(self, ps: ProbabilityScore) -> str:
        bar = self._score_bar(ps.total_score)
        lines = [
            f"  Instrumento : {ps.instrument}",
            f"  Score       : {ps.total_score:.1f}/100  {bar}",
            f"  Clasificación: {ps.classification}",
            f"  Bias        : {ps.trade_bias.upper()}",
            f"  Ambiente    : {ps.environment.replace('_',' ').upper()}",
            f"  No Trade    : {'SÍ ⛔' if ps.no_trade_signal else 'NO ✅'}",
            f"",
            f"  Desglose de scores (pesos {self._weight_str()}):",
            f"    Estructura : {ps.structure_score:.0%}",
            f"    Liquidez   : {ps.liquidity_score:.0%}",
            f"    Velas      : {ps.candle_score:.0%}",
            f"    Momentum   : {ps.momentum_score:.0%}",
            f"    MTF        : {ps.mtf_score:.0%}",
        ]
        if ps.reasons:
            lines += ["", "  Factores clave:"]
            for r in ps.reasons:
                lines.append(f"    • {r}")
        return "\n".join(lines)

    def _weight_str(self) -> str:
        w = self.weights
        return (f"S={w['structure']:.0%} L={w['liquidity']:.0%} "
                f"C={w['candle']:.0%} M={w['momentum']:.0%} T={w['mtf']:.0%}")

    @staticmethod
    def _score_bar(score: float) -> str:
        filled = int(score / 10)
        bar = '█' * filled + '░' * (10 - filled)
        return f"[{bar}]"
