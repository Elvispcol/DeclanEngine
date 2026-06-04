"""
OUTPUT FORMATTER
DeclanEngine - Decision Engine

Traduce señales internas a lenguaje trader-facing.
NUNCA expone raw diagnostics directamente.

El trader debe sentir:
  - guiado
  - protegido
  - filtrado
  - informado
  - operacionalmente claro

El sistema debe sentir como:
  un analista institucional simplificando el mercado para ejecución.

NO como:
  - consola de debug técnico
  - engine de código
  - bot de señales retail
"""

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .decision_engine import DecisionOutput
from entry_engine.entry_engine import SniperEntry


class OutputFormatter:
    """
    Formatea la salida del Decision Engine para el trader final.
    """

    def format(self, decision: 'DecisionOutput', instrument: str) -> str:
        """
        Formatea la decisión completa en output trader-facing.

        Parameters
        ----------
        decision : Output completo del DecisionEngine
        instrument : Nombre del instrumento

        Returns
        -------
        String formateado listo para mostrar al trader.
        """
        lines = []

        # Header
        lines.append("=" * 55)
        lines.append(f"  DECLAN ENGINE | {instrument}")
        lines.append("=" * 55)
        lines.append("")

        # 1. CONTEXTO DEL MERCADO
        lines.append("CONTEXTO DEL MERCADO:")
        lines.append(f"  {decision.contexto}")
        lines.append("")

        # 2. DIRECCIÓN DOMINANTE
        lines.append("DIRECCIÓN DOMINANTE:")
        lines.append(f"  {decision.direccion_dominante}")
        lines.append("")

        # 3. CONDICIÓN DEL MERCADO
        lines.append("CONDICIÓN DEL MERCADO:")
        lines.append(f"  {decision.condicion_mercado}")
        lines.append("")

        # 4. ACCIÓN RECOMENDADA
        lines.append("ACCIÓN RECOMENDADA:")
        lines.append(f"  {decision.accion_recomendada}")
        if decision.accion_detalle:
            lines.append(f"  {decision.accion_detalle}")
        lines.append("")

        # 5. SNIPER ENTRY SETUP
        if decision.setup and decision.setup.valid:
            setup = decision.setup
            lines.append("ENTRADA PREMIUM:")
            lines.append(f"  {setup.entry_price:.2f}")
            lines.append("")
            lines.append("STOP LOSS:")
            lines.append(f"  {setup.stop_loss:.2f}")
            lines.append(f"  ({setup.sl_reason})")
            lines.append("")
            lines.append("TAKE PROFIT 1:")
            lines.append(f"  {setup.take_profit_1:.2f}")
            lines.append(f"  ({setup.tp1_reason})")
            lines.append("")
            lines.append("TAKE PROFIT 2:")
            lines.append(f"  {setup.take_profit_2:.2f}")
            lines.append(f"  ({setup.tp2_reason})")
            lines.append("")
            lines.append("RIESGO / BENEFICIO:")
            lines.append(f"  1:{setup.risk_reward}")
            lines.append("")

        # 6. CALIDAD DEL SETUP
        lines.append("CALIDAD DEL SETUP:")
        lines.append(f"  {decision.calidad_setup}")
        lines.append("")

        # 7. ADVERTENCIA
        lines.append("ADVERTENCIA:")
        lines.append(f"  {decision.advertencia}")
        lines.append("")

        lines.append("=" * 55)

        return "\n".join(lines)

    def format_compact(self, decision: 'DecisionOutput', instrument: str) -> str:
        """
        Formato compacto ideal para Telegram/Discord/mobile.
        """
        lines = []
        lines.append(f"📊 {instrument}")
        lines.append(f"📍 {decision.direccion_dominante} | {decision.condicion_mercado}")
        lines.append(f"⚡ {decision.accion_recomendada}")

        if decision.setup and decision.setup.valid:
            s = decision.setup
            lines.append(f"🎯 Entry: {s.entry_price:.2f}")
            lines.append(f"🛑 SL: {s.stop_loss:.2f}")
            lines.append(f"✅ TP1: {s.take_profit_1:.2f}")
            lines.append(f"✅ TP2: {s.take_profit_2:.2f}")
            lines.append(f"📐 R:B = 1:{s.risk_reward}")

        lines.append(f"📋 {decision.calidad_setup}")
        lines.append(f"⚠️ {decision.advertencia}")

        return "\n".join(lines)

    def format_debug(self, decision: 'DecisionOutput') -> str:
        """
        Formato extendido con información interna para desarrollo/debug.
        NO para el trader final.
        """
        lines = []
        lines.append("── DEBUG OUTPUT (interno) ──")
        lines.append("")

        # Market State
        if decision._market_state:
            ms = decision._market_state
            lines.append(f"Estado dominante: {ms.dominant_state}")
            lines.append(f"Confianza: {ms.confidence:.2f}")
            lines.append(f"Bias: {ms.bias}")
            lines.append(f"Transition flag: {ms.transition_flag}")
            lines.append("State scores:")
            for state, score in sorted(ms.state_scores.items(), key=lambda x: -x[1]):
                if score > 0.1:
                    lines.append(f"  {state}: {score:.3f}")
            lines.append("")

        # Penalty Assessment
        if decision._penalty_assessment:
            pa = decision._penalty_assessment
            lines.append(f"Score original: {pa.original_score:.1f}")
            lines.append(f"Penalidades totales: {pa.total_penalty}")
            lines.append(f"Score ajustado: {pa.adjusted_score:.1f}")
            if pa.penalty_details:
                lines.append("Detalle penalidades:")
                for name, value in pa.penalty_details.items():
                    lines.append(f"  {name}: {value}")
            if pa.warnings:
                lines.append("Warnings:")
                for w in pa.warnings:
                    lines.append(f"  - {w}")
            lines.append("")

        # Setup details
        if decision.setup:
            s = decision.setup
            lines.append(f"Setup válido: {s.valid}")
            lines.append(f"Dirección: {s.direction}")
            lines.append(f"Entry type: {s.entry_type}")
            lines.append(f"Foundation: {s.foundation_score}")
            lines.append(f"Confirmation: {s.confirmation_score}")
            lines.append(f"Core conditions: {s.core_conditions_met}")
            lines.append(f"Confirmation layers: {s.confirmation_layers_met}")

        return "\n".join(lines)
