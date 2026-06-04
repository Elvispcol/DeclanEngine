"""
PENALTY ENGINE
DeclanEngine - Penalty Engine

Penaliza contradicciones agresivamente.
El engine actual recompensa confluencias correctamente
pero NO penaliza contradicciones.

Principio institucional:
Es preferible subestimar un setup que sobreestimarlo.
Proteger al trader de señales infladas.

Penalidades (expert-defined):
  Conflicto HTF/LTF             : -12
  Momentum débil                : -6
  Presión mixta                 : -8
  Sweep ambiguo                 : -10
  Compresión sin breakout       : -5
  Señales de agotamiento        : -9
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PenaltyAssessment:
    """Evaluación de penalidades aplicadas a un setup."""
    total_penalty: float
    penalty_details: Dict[str, float]  # nombre → valor de cada penalidad
    adjusted_score: float              # Score después de penalidades
    original_score: float              # Score antes de penalidades
    warnings: List[str] = field(default_factory=list)


class PenaltyEngine:
    """
    Evalúa contradicciones entre señales y aplica penalidades cuantificadas.

    Parameters
    ----------
    penalties : dict con valores de penalidad personalizables
    """

    DEFAULT_PENALTIES = {
        'htf_ltf_conflict':     12,
        'weak_momentum':        6,
        'mixed_pressure':       8,
        'ambiguous_sweep':      10,
        'compression_no_break': 5,
        'exhaustion_signals':   9,
    }

    def __init__(self, penalties: Dict[str, float] = None):
        self.penalties = penalties or self.DEFAULT_PENALTIES.copy()

    def assess(
        self,
        score: float,
        # Inputs para evaluación de penalidades
        structure_bias: str = 'neutral',
        htf_bias: str = None,
        momentum_state: str = 'stable',
        pressure_dominant: str = 'neutral',
        pressure_consistency: float = 0.0,
        sweep_present: bool = False,
        sweep_quality: str = None,
        sweep_direction: str = None,
        trade_direction: str = None,
        compression_active: bool = False,
        breakout_pending: bool = False,
        exhaustion_signal: bool = False,
        displacement_present: bool = False,
        displacement_direction: str = None,
    ) -> PenaltyAssessment:
        """
        Evalúa todas las contradicciones y aplica penalidades.

        Parameters
        ----------
        score : Score original del ProbabilityEngine (0-100)
        Otros parámetros: señales de los motores para evaluar contradicciones.

        Returns
        -------
        PenaltyAssessment con penalidades detalladas y score ajustado.
        """
        penalty_details = {}
        warnings = []

        # ── 1. HTF/LTF CONFLICT ────────────────────────────────────
        if htf_bias is not None and structure_bias != 'neutral' and htf_bias != 'neutral':
            if structure_bias != htf_bias:
                p = self.penalties['htf_ltf_conflict']
                penalty_details['htf_ltf_conflict'] = -p
                warnings.append(
                    f"Conflicto temporalidades: estructura {structure_bias} vs HTF {htf_bias}"
                )

        # ── 2. WEAK MOMENTUM ───────────────────────────────────────
        if momentum_state in ('decaying', 'exhausted'):
            p = self.penalties['weak_momentum']
            if momentum_state == 'exhausted':
                p = int(p * 1.5)  # Exhausted = 50% más penalidad
            penalty_details['weak_momentum'] = -p
            warnings.append(f"Momentum débil: {momentum_state}")

        # ── 3. MIXED PRESSURE ──────────────────────────────────────
        if pressure_consistency < 0.40 and pressure_dominant != 'neutral':
            p = self.penalties['mixed_pressure']
            # Consistencia muy baja = más penalidad
            if pressure_consistency < 0.25:
                p = int(p * 1.3)
            penalty_details['mixed_pressure'] = -p
            warnings.append(
                f"Presión mixta: consistencia {pressure_consistency:.0%} ({pressure_dominant})"
            )

        # ── 4. AMBIGUOUS SWEEP ────────────────────────────────────
        if sweep_present and sweep_quality == 'weak':
            p = self.penalties['ambiguous_sweep']
            penalty_details['ambiguous_sweep'] = -p
            warnings.append("Sweep ambiguo de calidad débil")

        # Sweep contra dirección del trade
        if (sweep_present and trade_direction is not None
                and sweep_direction is not None):
            # sweep_high → probable_direction bearish; si trade es bullish, contradicción
            sweep_implies = 'bearish' if sweep_direction == 'sweep_high' else 'bullish'
            if sweep_implies != trade_direction and sweep_quality in ('weak', 'standard'):
                if 'ambiguous_sweep' not in penalty_details:
                    p = int(self.penalties['ambiguous_sweep'] * 0.7)
                    penalty_details['sweep_against_trade'] = -p
                    warnings.append(
                        f"Sweep contradice dirección del trade ({sweep_direction})"
                    )

        # ── 5. COMPRESSION WITHOUT BREAKOUT ────────────────────────
        if compression_active and not breakout_pending:
            p = self.penalties['compression_no_break']
            penalty_details['compression_no_break'] = -p
            warnings.append("Compresión activa sin señal de breakout")

        # ── 6. EXHAUSTION SIGNALS ──────────────────────────────────
        if exhaustion_signal:
            p = self.penalties['exhaustion_signals']
            penalty_details['exhaustion_signals'] = -p
            warnings.append("Señal de agotamiento de momentum activa")

        # Displacement contra trade direction = sospechoso
        if (displacement_present and trade_direction is not None
                and displacement_direction is not None
                and displacement_direction != trade_direction):
            p = 7  # Penalidad adicional por displacement contra trade
            penalty_details['displacement_against_trade'] = -p
            warnings.append(
                f"Desplazamiento contradice dirección ({displacement_direction} vs {trade_direction})"
            )

        # ── CALCULATE ADJUSTED SCORE ───────────────────────────────
        total_penalty = sum(penalty_details.values())
        adjusted = max(0.0, score + total_penalty)  # Floor en 0

        return PenaltyAssessment(
            total_penalty=total_penalty,
            penalty_details=penalty_details,
            adjusted_score=round(adjusted, 1),
            original_score=score,
            warnings=warnings,
        )

    def assess_setup_quality(
        self,
        score: float,
        **kwargs
    ) -> float:
        """
        Shortcut: retorna solo el score ajustado.

        Útil para integrar rápidamente en el ProbabilityEngine.
        """
        assessment = self.assess(score, **kwargs)
        return assessment.adjusted_score
