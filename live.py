"""
DECLAN ENGINE - Live Analysis
Análisis con datos reales de Deriv/MT5 + Decision Engine + HTF Analysis

Uso:
    python live.py
    python live.py CRASH1000
    python live.py BOOM1000 M15
    python live.py BOOM1000 M5 --debug
"""

import sys
from mt5_connector import MT5Connector
from data.sample_generator import generate_boom_crash_data
from structure_engine import SwingDetector, StructureClassifier, BOSCHOCHDetector, DisplacementDetector
from liquidity_engine import EqualLevelsDetector, SweepDetector, InducementDetector
from candle_engine import PressureAnalyzer, MomentumDetector, CompressionDetector
from probability_engine import ProbabilityEngine
from decision_engine import DecisionEngine, OutputFormatter
from mtf_engine import HTFAnalyzer


def run_analysis(df, instrument: str, timeframe: str, debug: bool = False,
                 htf_df=None):
    """Corre el engine completo sobre un DataFrame OHLC con Decision Engine."""

    # ── Sprint 1 — Structure Engine ────────────────────────────────
    sd = SwingDetector(left_bars=3, right_bars=3)
    sh, sl = sd.get_swing_list(df)
    sc = StructureClassifier()
    sp = sc.classify(sh, sl)
    sm = sc.get_structure_summary(sp)
    bd = BOSCHOCHDetector()
    evts = bd.detect(df, sp)
    ev = bd.get_events_summary(evts)
    dd = DisplacementDetector()
    dsps = dd.detect(df)
    rd = dd.get_recent_displacement(dsps, lookback=15)

    # ── Sprint 2 — Liquidity Engine ────────────────────────────────
    eld = EqualLevelsDetector()
    eql = eld.detect(df, sh, sl)
    ls = eld.get_summary(eql)
    swd = SweepDetector()
    swp = swd.detect_from_swings(df, sh, sl)
    ss = swd.get_summary(swp)
    rswp = swd.get_recent_sweeps(swp, lookback=20)
    ind = InducementDetector()
    izs = ind.detect(df, evts, dsps)
    iz = ind.get_summary(izs)

    # ── Sprint 3 — Candle Engine ───────────────────────────────────
    pa = PressureAnalyzer(window_size=5)
    rdgs = pa.analyze_all(df)
    wp = pa.get_current_pressure(df, rdgs)
    md = MomentumDetector(window=6)
    msts = md.detect(df)
    mom = md.get_current(msts)
    cd = CompressionDetector()
    csts = cd.detect(df)
    cmp = cd.get_summary(csts)

    # ── Sprint 4 — Probability Engine ──────────────────────────────
    pe = ProbabilityEngine(instrument=instrument, min_score=60.0)
    score = pe.calculate(
        structure_summary=sm, events_summary=ev,
        liquidity_summary=ls, sweep_summary=ss,
        inducement_summary=iz, window_pressure=wp,
        momentum_state=mom, compression_state=cmp,
        recent_displacement=rd, recent_sweeps=rswp,
    )

    # ── MTF Engine — HTF Analysis ──────────────────────────────────
    htf_result = None

    if htf_df is not None:
        # Real HTF analysis from MT5 data
        htf_analyzer = HTFAnalyzer(htf_timeframe=_get_htf_timeframe(timeframe))
        htf_result = htf_analyzer.analyze_from_dataframes(df, htf_df)
    else:
        # Fallback: simulated HTF
        htf_n_candles = max(len(df) // 3, 100)
        htf_sim_df = generate_boom_crash_data(htf_n_candles, instrument, seed=99)
        htf_analyzer = HTFAnalyzer(htf_timeframe=_get_htf_timeframe(timeframe))
        htf_result = htf_analyzer.analyze_from_dataframes(df, htf_sim_df)

    # Recalcular probability con HTF real
    if htf_result and htf_result.alignment_score is not None:
        score = pe.calculate(
            structure_summary=sm, events_summary=ev,
            liquidity_summary=ls, sweep_summary=ss,
            inducement_summary=iz, window_pressure=wp,
            momentum_state=mom, compression_state=cmp,
            recent_displacement=rd, recent_sweeps=rswp,
            htf_alignment_score=htf_result.alignment_score,
        )

    if debug and htf_result:
        print(f"\n── HTF Analysis ({htf_analyzer.htf_timeframe}) ──")
        print(f"  Bias: {htf_result.htf_bias}")
        print(f"  Strength: {htf_result.htf_strength:.2f}")
        print(f"  Alignment: {htf_result.alignment} ({htf_result.alignment_score:.2f})")
        print(f"  Narrative: {htf_result.narrative}")
        print()

    # ── Decision Engine ────────────────────────────────────────────
    last_event_type = ev['last'].event_type if ev['last'] else ''
    last_event_sig = ev['last'].significance if ev['last'] else ''

    has_displacement = len(rd) > 0
    displacement_direction = rd[-1].direction if rd else ''

    sweep_present = len(rswp) > 0
    sweep_direction = rswp[-1].sweep_type if rswp else ''
    sweep_quality = rswp[-1].sweep_quality if rswp else ''
    sweep_level = rswp[-1].level_swept if rswp else 0.0

    inducement_present = iz['total'] > 0
    inducement_direction = iz['last'].probable_move if iz['last'] else ''

    atr = _compute_current_atr(df)
    current_price = df['close'].iloc[-1]

    active_levels = eld.get_active_levels(eql)

    # Add HTF liquidity levels for TP2
    htf_liquidity = htf_result.htf_liquidity_levels if htf_result else []
    all_liquidity_levels = list(active_levels) + htf_liquidity

    swing_highs = sh['price'].tolist() if len(sh) > 0 else []
    swing_lows = sl['price'].tolist() if len(sl) > 0 else []

    rejection_quality = 0.0
    if wp.rejection_detected:
        rejection_quality = 0.7 if wp.quality == 'strong' else 0.4

    de = DecisionEngine(instrument=instrument)
    decision = de.decide(
        total_score=score.total_score,
        trade_bias=score.trade_bias,
        no_trade_signal=score.no_trade_signal,
        structure_bias=sm['bias'],
        structure_score=score.structure_score,
        last_event_type=last_event_type,
        last_event_significance=last_event_sig,
        has_displacement=has_displacement,
        displacement_direction=displacement_direction,
        sweep_present=sweep_present,
        sweep_direction=sweep_direction,
        sweep_quality=sweep_quality,
        sweep_level=sweep_level,
        recent_sweeps=rswp,
        inducement_present=inducement_present,
        inducement_direction=inducement_direction,
        momentum_state=mom.state if mom else 'stable',
        momentum_decay_score=mom.decay_score if mom else 0.0,
        exhaustion_signal=mom.exhaustion_signal if mom else False,
        pressure_dominant=wp.dominant_side,
        pressure_quality=wp.quality,
        net_pressure=wp.net_pressure,
        pressure_consistency=wp.consistency,
        compression_active=cmp['compressed'],
        compression_score=cmp.get('score', 0),
        breakout_pending=cmp.get('breakout_pending', False),
        absorption_detected=wp.absorption_detected,
        rejection_detected=wp.rejection_detected,
        rejection_quality=rejection_quality,
        liquidity_score=score.liquidity_score,
        active_equal_levels=all_liquidity_levels,
        recent_swing_highs=swing_highs,
        recent_swing_lows=swing_lows,
        current_price=current_price,
        atr=atr,
        df=df,
        htf_bias=htf_result.htf_bias if htf_result else None,
    )

    # ── OUTPUT (Trader-Facing) ─────────────────────────────────────
    formatter = OutputFormatter()
    output = formatter.format(decision, instrument)

    # Add timeframe info
    tf_line = f"  Temporalidad: {timeframe} | Barras: {len(df)}"
    if 'time' in df.columns:
        tf_line += f" | Última: {df['time'].iloc[-1]}"
    # Insert timeframe info after header
    lines = output.split('\n')
    lines.insert(2, tf_line)
    print('\n'.join(lines))

    # Debug output
    if debug:
        print(formatter.format_debug(decision))
        print(f"\n── Probability Engine ──")
        print(pe.format_output(score))


def _get_htf_timeframe(ltf: str) -> str:
    """Determina el HTF apropiado para el timeframe operativo dado."""
    htf_map = {
        'M1': 'M15',
        'M5': 'H1',
        'M15': 'H4',
        'M30': 'H4',
        'H1': 'D1',
    }
    return htf_map.get(ltf, 'H1')


def _compute_current_atr(df, period: int = 14) -> float:
    """Computa el ATR actual del DataFrame."""
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atr = 0.0
    for i in range(1, min(len(df), period + 1)):
        tr = max(
            highs[-i] - lows[-i],
            abs(highs[-i] - closes[-i - 1]),
            abs(lows[-i] - closes[-i - 1])
        )
        if i == 1:
            atr = tr
        else:
            atr = (atr * (period - 1) + tr) / period
    return atr if atr > 0 else 1.0


def main():
    instrument = sys.argv[1].upper() if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'BOOM1000'
    timeframe  = sys.argv[2].upper() if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'M5'
    n_bars     = int(sys.argv[3])    if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 300
    debug      = '--debug' in sys.argv

    print(f"\n  Intentando conectar con MT5...")
    connector = MT5Connector()

    if connector.connect():
        df = connector.get_ohlc(instrument, timeframe, n_bars)

        # Try to get HTF data from MT5
        htf_timeframe = _get_htf_timeframe(timeframe)
        htf_df = connector.get_ohlc(instrument, htf_timeframe, n_bars // 2)

        connector.disconnect()

        if df is not None:
            run_analysis(df, instrument, timeframe, debug=debug, htf_df=htf_df)
        else:
            print(f"\n  ⚠ No se pudo obtener datos de MT5.")
            print(f"  Usando datos simulados para {instrument}...\n")
            df = generate_boom_crash_data(n_bars, instrument)
            df['time'] = df['time'].astype(str)
            run_analysis(df, instrument, timeframe, debug=debug)
    else:
        print(f"\n  MT5 no disponible. Usando datos simulados...\n")
        df = generate_boom_crash_data(n_bars, instrument)
        run_analysis(df, instrument, timeframe, debug=debug)


if __name__ == "__main__":
    main()
