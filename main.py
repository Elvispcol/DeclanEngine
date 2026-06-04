"""
DECLAN ENGINE - Main Entry Point
Sprint 1 + 2 + 3: Structure + Liquidity + Candle Pressure

Declan Trader | Porciento Trading
"""

from data.sample_generator import generate_boom_crash_data
from structure_engine import SwingDetector, StructureClassifier, BOSCHOCHDetector, DisplacementDetector
from liquidity_engine import EqualLevelsDetector, SweepDetector, InducementDetector
from candle_engine import PressureAnalyzer, MomentumDetector, CompressionDetector


def analyze(instrument: str = 'BOOM1000', n_candles: int = 300):
    print(f"\n{'='*55}")
    print(f"  DECLAN ENGINE | {instrument}")
    print(f"{'='*55}")

    df = generate_boom_crash_data(n_candles, instrument)

    # Sprint 1 — Structure
    sd = SwingDetector(left_bars=3, right_bars=3)
    sh, sl = sd.get_swing_list(df)
    sc = StructureClassifier()
    sp = sc.classify(sh, sl)
    sm = sc.get_structure_summary(sp)
    bd = BOSCHOCHDetector()
    events = bd.detect(df, sp)
    ev = bd.get_events_summary(events)
    dd = DisplacementDetector()
    disps = dd.detect(df)
    rd = dd.get_recent_displacement(disps, lookback=15)

    print(f"\n  [ESTRUCTURA]")
    print(f"  Bias: {sm['bias'].upper()} | HH:{sm['hh_count']} HL:{sm['hl_count']} LH:{sm['lh_count']} LL:{sm['ll_count']}")
    print(f"  BOS:{ev['bos']} CHOCH:{ev['choch']}", end="")
    if ev['last']:
        e = ev['last']
        print(f" | {e.event_type} {e.direction.upper()} [{e.significance}]")
    else:
        print()
    if rd:
        d = rd[-1]
        print(f"  Desplaz: {d.direction.upper()} [{d.quality}] body={d.body_ratio:.0%} {d.range_vs_atr:.1f}xATR")

    # Sprint 2 — Liquidity
    eld = EqualLevelsDetector()
    eql = eld.detect(df, sh, sl)
    ls  = eld.get_summary(eql)
    swd = SweepDetector()
    swp = swd.detect_from_swings(df, sh, sl)
    ss  = swd.get_summary(swp)
    rswp = swd.get_recent_sweeps(swp, lookback=20)
    ind = InducementDetector()
    izs = ind.detect(df, events, disps)
    iz  = ind.get_summary(izs)

    print(f"\n  [LIQUIDEZ]")
    print(f"  EqLevels: Activos:{ls['active']} Premium:{ls['premium']} Barridos:{ls['swept']}")
    print(f"  Sweeps: Total:{ss['total']} Premium:{ss['premium']}", end="")
    if rswp:
        s = rswp[-1]
        print(f" | {s.sweep_type.upper()} [{s.sweep_quality}] → {s.probable_direction.upper()}")
    else:
        print()
    print(f"  Inducement: Total:{iz['total']} Premium:{iz['premium']}", end="")
    if iz['last']:
        z = iz['last']
        print(f" | {z.inducement_type} [{z.trap_quality}] → {z.probable_move.upper()}")
    else:
        print()

    # Sprint 3 — Candle Pressure
    pa   = PressureAnalyzer(window_size=5)
    rdgs = pa.analyze_all(df)
    wp   = pa.get_current_pressure(df, rdgs)
    md   = MomentumDetector(window=6)
    msts = md.detect(df)
    mom  = md.get_current(msts)
    cd   = CompressionDetector()
    csts = cd.detect(df)
    cmp  = cd.get_summary(csts)

    print(f"\n  [PRESIÓN DE VELAS]")
    print(f"  Presión: {wp.dominant_side.upper()} [{wp.quality}] "
          f"score={wp.net_pressure:+.2f} consistencia={wp.consistency:.0%}")
    print(f"  Tendencia presión: {wp.pressure_trend.upper()}", end="")
    if wp.absorption_detected: print(" | ABSORCIÓN detectada", end="")
    if wp.rejection_detected:  print(" | RECHAZO detectado", end="")
    print()

    if mom:
        print(f"  Momentum: {mom.state.upper()} decay={mom.decay_score:.0%}", end="")
        if mom.exhaustion_signal: print(" ⚠ AGOTAMIENTO", end="")
        print()

    print(f"  Compresión: {'SÍ' if cmp['compressed'] else 'NO'} "
          f"[{cmp['quality']}] score={cmp.get('score', 0):.0%}", end="")
    if cmp.get('breakout_pending'): print(" ⚡ BREAKOUT PENDIENTE", end="")
    print()

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    analyze('BOOM1000')
    analyze('CRASH1000')
