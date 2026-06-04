"""
DECLAN ENGINE - Live Analysis
Análisis con datos reales de Deriv/MT5

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


def run_analysis(df, instrument: str, timeframe: str):
    """Corre el engine completo sobre un DataFrame OHLC."""

    # Sprint 1 — Structure
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

    # Sprint 2 — Liquidity
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

    # Sprint 3 — Candle Pressure
    pa = PressureAnalyzer(window_size=5)
    rdgs = pa.analyze_all(df)
    wp = pa.get_current_pressure(df, rdgs)
    md = MomentumDetector(window=6)
    msts = md.detect(df)
    mom = md.get_current(msts)
    cd = CompressionDetector()
    csts = cd.detect(df)
    cmp = cd.get_summary(csts)

    # Sprint 4 — Probability
    pe = ProbabilityEngine(instrument=instrument, min_score=60.0)
    score = pe.calculate(
        structure_summary=sm, events_summary=ev,
        liquidity_summary=ls, sweep_summary=ss,
        inducement_summary=iz, window_pressure=wp,
        momentum_state=mom, compression_state=cmp,
        recent_displacement=rd, recent_sweeps=rswp,
    )

    # Output
    print(f"\n{'='*55}")
    print(f"  DECLAN ENGINE | {instrument} {timeframe}")
    print(f"  Barras: {len(df)} | Última: {df['time'].iloc[-1]}")
    print(f"{'='*55}")

    print(f"\n  [ESTRUCTURA]")
    print(f"  Bias: {sm['bias'].upper()} | HH:{sm['hh_count']} HL:{sm['hl_count']} LH:{sm['lh_count']} LL:{sm['ll_count']}")
    if ev['last']:
        e = ev['last']
        print(f"  {e.event_type} {e.direction.upper()} [{e.significance}] | BOS:{ev['bos']} CHOCH:{ev['choch']}")
    if rd:
        d = rd[-1]
        print(f"  Desplaz: {d.direction.upper()} [{d.quality}] body={d.body_ratio:.0%}")

    print(f"\n  [LIQUIDEZ]")
    print(f"  EqLevels activos:{ls['active']} premium:{ls['premium']}")
    if rswp:
        s = rswp[-1]
        print(f"  Último sweep: {s.sweep_type.upper()} [{s.sweep_quality}] → {s.probable_direction.upper()}")
    if iz['last']:
        z = iz['last']
        print(f"  Inducement [{z.trap_quality}] → {z.probable_move.upper()}")

    print(f"\n  [PRESIÓN]")
    print(f"  {wp.dominant_side.upper()} [{wp.quality}] score={wp.net_pressure:+.2f}", end="")
    if wp.absorption_detected: print(" | ABSORCIÓN", end="")
    if wp.rejection_detected:  print(" | RECHAZO", end="")
    print()
    if mom:
        print(f"  Momentum: {mom.state.upper()}", end="")
        if mom.exhaustion_signal: print(" ⚠ AGOTAMIENTO", end="")
        print()
    if cmp.get('breakout_pending'):
        print(f"  ⚡ COMPRESIÓN EXTREMA — BREAKOUT PENDIENTE")

    print(f"\n  [PROBABILIDAD]")
    print(pe.format_output(score))
    print(f"\n{'='*55}\n")


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
