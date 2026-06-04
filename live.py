"""
DECLAN ENGINE - Live Analysis
Análisis con datos reales de Deriv/MT5 + Decision Engine

Uso:
    python live.py
    python live.py CRASH1000
    python live.py BOOM1000 M15
"""

import sys
from mt5_connector import MT5Connector
from data.sample_generator import generate_boom_crash_data
from structure_engine import SwingDetector, StructureClassifier, BOSCHOCHDetector, DisplacementDetector
from liquidity_engine import EqualLevelsDetector, SweepDetector, InducementDetector
from candle_engine import PressureAnalyzer, MomentumDetector, CompressionDetector
from probability_engine import ProbabilityEngine
from decision_engine import DecisionEngine, OutputFormatter


def run_analysis(df, instrument: str, timeframe: str):
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
        active_equal_levels=active_levels,
        recent_swing_highs=swing_highs,
        recent_swing_lows=swing_lows,
        current_price=current_price,
        atr=atr,
        df=df,
    )

    # ── OUTPUT (Trader-Facing) ─────────────────────────────────────
    formatter = OutputFormatter()
    output = formatter.format(decision, instrument)

    # Add timeframe info
    tf_line = f"  Temporalidad: {timeframe} | Barras: {len(df)}"
    if 'time' in df.columns:
        tf_line += f" | Última: {df['time'].iloc[-1]}"
    output = output.replace("=" * 55, "=" * 55, 1)
    # Insert timeframe info after header
    lines = output.split('\n')
    lines.insert(2, tf_line)
    print('\n'.join(lines))


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
    instrument = sys.argv[1].upper() if len(sys.argv) > 1 else 'BOOM1000'
    timeframe  = sys.argv[2].upper() if len(sys.argv) > 2 else 'M5'
    n_bars     = int(sys.argv[3])    if len(sys.argv) > 3 else 300

    print(f"\n  Intentando conectar con MT5...")
    connector = MT5Connector()

    if connector.connect():
        df = connector.get_ohlc(instrument, timeframe, n_bars)
        connector.disconnect()

        if df is not None:
            run_analysis(df, instrument, timeframe)
        else:
            print(f"\n  ⚠ No se pudo obtener datos de MT5.")
            print(f"  Usando datos simulados para {instrument}...\n")
            df = generate_boom_crash_data(n_bars, instrument)
            df['time'] = df['time'].astype(str)
            run_analysis(df, instrument, timeframe)
    else:
        print(f"\n  MT5 no disponible. Usando datos simulados...\n")
        df = generate_boom_crash_data(n_bars, instrument)
        run_analysis(df, instrument, timeframe)


if __name__ == "__main__":
    main()
