"""
HTF ANALYZER
DeclanEngine - Multi-Timeframe Analysis

Análisis real de alineación Higher Timeframe.

Principio institucional:
El timeframe operativo (LTF) define la entrada.
El Higher Timeframe (HTF) define la dirección.
Si ambos están alineados → confluencia → alta probabilidad.
Si están en conflicto → HTF gobierna → no operar contra HTF.

HTF NO significa que el M15 es HTF del M5.
HTF significa el timeframe que define la MACRO estructura.
Para synthetic indices en M5: HTF = M15/H1
Para synthetic indices en M1: HTF = M5/M15

El HTF Analyzer:
  1. Analiza la estructura del HTF (bias, último evento)
  2. Evalúa la alineación LTF→HTF
  3. Genera un HTF bias y alignment score
  4. Identifica niveles de liquidez HTF (para TP2)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import pandas as pd


@dataclass
class HTFResult:
    """Resultado del análisis Higher Timeframe."""
    htf_bias: str               # 'bullish' | 'bearish' | 'neutral'
    htf_strength: float         # 0.0 - 1.0 (fuerza del bias HTF)
    alignment: str              # 'aligned' | 'neutral' | 'conflicting'
    alignment_score: float      # 0.0 - 1.0 (score de alineación)
    htf_last_event: str         # 'BOS' | 'CHOCH' | ''
    htf_last_event_dir: str     # 'bullish' | 'bearish' | ''
    htf_swing_highs: List[float] = field(default_factory=list)
    htf_swing_lows: List[float] = field(default_factory=list)
    htf_liquidity_levels: List[Dict] = field(default_factory=list)
    narrative: str = ''


class HTFAnalyzer:
    """
    Analiza la alineación Higher Timeframe para el Decision Engine.

    Parameters
    ----------
    htf_timeframe : timeframe considerado como HTF
    """

    def __init__(self, htf_timeframe: str = 'H1'):
        self.htf_timeframe = htf_timeframe

    def analyze(
        self,
        htf_structure_bias: str,
        htf_structure_score: float,
        htf_last_event_type: str,
        htf_last_event_direction: str,
        htf_last_event_significance: str,
        htf_has_displacement: bool,
        htf_displacement_direction: str,
        htf_momentum_state: str,
        htf_pressure_dominant: str,
        htf_swing_highs: List[float] = None,
        htf_swing_lows: List[float] = None,
        htf_equal_levels: list = None,
        # LTF data for alignment check
        ltf_structure_bias: str = 'neutral',
    ) -> HTFResult:
        """
        Produce el análisis HTF completo.

        Parameters
        ----------
        La mayoría de parámetros provienen de correr los engines
        existentes (Structure, Liquidity, Candle) sobre el DataFrame HTF.

        Returns
        -------
        HTFResult con bias, alineación y niveles de liquidez HTF.
        """
        swing_highs = htf_swing_highs or []
        swing_lows = htf_swing_lows or []
        equal_levels = htf_equal_levels or []

        # ── 1. HTF BIAS ────────────────────────────────────────────
        htf_bias = htf_structure_bias
        htf_strength = 0.0

        if htf_bias != 'neutral':
            htf_strength += 0.30

        # BOS/CHOCH en HTF refuerza el bias
        if htf_last_event_type == 'BOS':
            sig_map = {'strong': 0.25, 'moderate': 0.15, 'weak': 0.05}
            htf_strength += sig_map.get(htf_last_event_significance, 0.05)
        elif htf_last_event_type == 'CHOCH':
            sig_map = {'strong': 0.30, 'moderate': 0.20, 'weak': 0.08}
            htf_strength += sig_map.get(htf_last_event_significance, 0.08)

        # Displacement HTF refuerza
        if htf_has_displacement:
            if htf_displacement_direction == htf_bias:
                htf_strength += 0.20
            elif htf_displacement_direction != 'neutral':
                # Displacement contra bias = debilidad
                htf_strength *= 0.7

        # Momentum HTF
        mom_map = {'accelerating': 0.15, 'stable': 0.10, 'decaying': -0.05, 'exhausted': -0.15}
        htf_strength += mom_map.get(htf_momentum_state, 0.0)

        # Pressure HTF
        if htf_pressure_dominant != 'neutral':
            pressure_dir = 'bullish' if htf_pressure_dominant == 'buyers' else 'bearish'
            if pressure_dir == htf_bias:
                htf_strength += 0.10
            else:
                htf_strength -= 0.05

        htf_strength = max(0.0, min(htf_strength, 1.0))

        # ── 2. ALIGNMENT ───────────────────────────────────────────
        if ltf_structure_bias == 'neutral' or htf_bias == 'neutral':
            alignment = 'neutral'
            alignment_score = 0.35
        elif ltf_structure_bias == htf_bias:
            alignment = 'aligned'
            alignment_score = 0.50 + (htf_strength * 0.50)
        else:
            alignment = 'conflicting'
            alignment_score = max(0.0, 0.30 - (htf_strength * 0.30))

        # ── 3. HTF LIQUIDITY LEVELS (para TP2) ─────────────────────
        htf_liquidity = []

        # Swing highs/lows del HTF son niveles macroestructurales
        for h in swing_highs:
            htf_liquidity.append({
                'price': h,
                'level_type': 'htf_swing_high',
                'is_major': True,
            })
        for l in swing_lows:
            htf_liquidity.append({
                'price': l,
                'level_type': 'htf_swing_low',
                'is_major': True,
            })

        # Equal levels del HTF
        for level in equal_levels:
            if hasattr(level, 'price') and hasattr(level, 'level_type'):
                htf_liquidity.append({
                    'price': level.price,
                    'level_type': f"htf_{level.level_type}",
                    'is_major': True,
                })
            elif isinstance(level, dict):
                htf_liquidity.append({**level, 'is_major': True})

        # ── 4. NARRATIVE ───────────────────────────────────────────
        if alignment == 'aligned':
            narrative = f"Alineación HTF confirmada: estructura {htf_bias} en {self.htf_timeframe}. Confluencia temporal activa."
        elif alignment == 'conflicting':
            narrative = f"Conflicto temporal: LTF {ltf_structure_bias} vs HTF {htf_bias}. Precaución — no operar contra HTF."
        else:
            narrative = f"HTF sin bias claro en {self.htf_timeframe}. Sin confluencia temporal disponible."

        return HTFResult(
            htf_bias=htf_bias,
            htf_strength=round(htf_strength, 3),
            alignment=alignment,
            alignment_score=round(alignment_score, 3),
            htf_last_event=htf_last_event_type,
            htf_last_event_dir=htf_last_event_direction,
            htf_swing_highs=swing_highs,
            htf_swing_lows=swing_lows,
            htf_liquidity_levels=htf_liquidity,
            narrative=narrative,
        )

    def analyze_from_dataframes(
        self,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame,
    ) -> HTFResult:
        """
        Análisis HTF completo corriendo todos los engines sobre ambos DataFrames.

        Este es el método de conveniencia que corre Structure Engine,
        Candle Engine y extrae la información necesaria de ambos timeframes.

        Parameters
        ----------
        ltf_df : DataFrame del timeframe operativo (M1, M5)
        htf_df : DataFrame del higher timeframe (M15, H1)

        Returns
        -------
        HTFResult completo.
        """
        from structure_engine import (
            SwingDetector, StructureClassifier,
            BOSCHOCHDetector, DisplacementDetector
        )
        from candle_engine import PressureAnalyzer, MomentumDetector

        # ── Run engines on HTF DataFrame ───────────────────────────
        sd = SwingDetector(left_bars=5, right_bars=5)
        sh, sl = sd.get_swing_list(htf_df)
        sc = StructureClassifier()
        sp = sc.classify(sh, sl)
        sm = sc.get_structure_summary(sp)
        bd = BOSCHOCHDetector()
        evts = bd.detect(htf_df, sp)
        ev = bd.get_events_summary(evts)
        dd = DisplacementDetector()
        dsps = dd.detect(htf_df)
        rd = dd.get_recent_displacement(dsps, lookback=10)

        pa = PressureAnalyzer(window_size=5)
        rdgs = pa.analyze_all(htf_df)
        wp = pa.get_current_pressure(htf_df, rdgs)
        md = MomentumDetector(window=6)
        msts = md.detect(htf_df)
        mom = md.get_current(msts)

        # Extract HTF data
        htf_bias = sm['bias']
        htf_score = 0.5  # Placeholder
        htf_last_event = ev['last'].event_type if ev['last'] else ''
        htf_last_dir = ev['last'].direction if ev['last'] else ''
        htf_last_sig = ev['last'].significance if ev['last'] else ''
        htf_has_disp = len(rd) > 0
        htf_disp_dir = rd[-1].direction if rd else ''
        htf_mom = mom.state if mom else 'stable'
        htf_pressure = wp.dominant_side

        # LTF structure
        sd_ltf = SwingDetector(left_bars=3, right_bars=3)
        sh_ltf, sl_ltf = sd_ltf.get_swing_list(ltf_df)
        sc_ltf = StructureClassifier()
        sp_ltf = sc_ltf.classify(sh_ltf, sl_ltf)
        sm_ltf = sc_ltf.get_structure_summary(sp_ltf)
        ltf_bias = sm_ltf['bias']

        # HTF swing levels
        htf_highs = sh['price'].tolist() if len(sh) > 0 else []
        htf_lows = sl['price'].tolist() if len(sl) > 0 else []

        return self.analyze(
            htf_structure_bias=htf_bias,
            htf_structure_score=htf_score,
            htf_last_event_type=htf_last_event,
            htf_last_event_direction=htf_last_dir,
            htf_last_event_significance=htf_last_sig,
            htf_has_displacement=htf_has_disp,
            htf_displacement_direction=htf_disp_dir,
            htf_momentum_state=htf_mom,
            htf_pressure_dominant=htf_pressure,
            htf_swing_highs=htf_highs,
            htf_swing_lows=htf_lows,
            ltf_structure_bias=ltf_bias,
        )
