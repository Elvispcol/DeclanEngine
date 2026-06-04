"""
DECLAN ENGINE - Main Entry Point
Sprint 1 + Sprint 2: Structure + Liquidity Engine

Institutional Synthetic Indices Context Engine
Declan Trader | Porciento Trading
"""

from data.sample_generator import generate_boom_crash_data
from structure_engine import (
    SwingDetector, StructureClassifier,
    BOSCHOCHDetector, DisplacementDetector
)
from liquidity_engine import (
    EqualLevelsDetector, SweepDetector, InducementDetector
)


def analyze(instrument: str = 'BOOM1000', n_candles: int = 300):
    print(f"\n{'='*55}")
    print(f"  DECLAN ENGINE | {instrument}")
    print(f"{'='*55}")

    df = generate_boom_crash_data(n_candles, instrument)

    # --- Sprint 1: Structure ---
    sd = SwingDetector(left_bars=3, right_bars=3)
    swing_highs, swing_lows = sd.get_swing_list(df)

    sc = StructureClassifier()
    structure_points = sc.classify(swing_highs, swing_lows)
    summary = sc.get_structure_summary(structure_points)

    bd = BOSCHOCHDetector()
    events = bd.detect(df, structure_points)
    ev_sum = bd.get_events_summary(events)

    dd = DisplacementDetector()
    displacements = dd.detect(df)
    recent_disp = dd.get_recent_displacement(displacements, lookback=15)

    print(f"\n  [ESTRUCTURA]")
    print(f"  Bias : {summary['bias'].upper()}")
    print(f"  HH:{summary['hh_count']} HL:{summary['hl_count']} "
          f"LH:{summary['lh_count']} LL:{summary['ll_count']}")
    print(f"  BOS:{ev_sum['bos']} | CHOCH:{ev_sum['choch']}", end="")
    if ev_sum['last']:
        e = ev_sum['last']
        print(f" | Último: {e.event_type} {e.direction.upper()} [{e.significance}]")
    else:
        print()

    if recent_disp:
        d = recent_disp[-1]
        print(f"  Desplaz: {d.direction.upper()} [{d.quality}] "
              f"body={d.body_ratio:.0%} {d.range_vs_atr:.1f}xATR")

    # --- Sprint 2: Liquidity ---
    eld = EqualLevelsDetector(atr_multiplier=0.10, min_touches=2)
    eq_levels = eld.detect(df, swing_highs, swing_lows)
    liq_sum = eld.get_summary(eq_levels)

    sweep_det = SweepDetector()
    sweeps = sweep_det.detect_from_swings(df, swing_highs, swing_lows)
    sweep_sum = sweep_det.get_summary(sweeps)
    recent_sweeps = sweep_det.get_recent_sweeps(sweeps, lookback=20)

    ind_det = InducementDetector()
    inducements = ind_det.detect(df, events, displacements)
    ind_sum = ind_det.get_summary(inducements)

    print(f"\n  [LIQUIDEZ]")
    print(f"  Equal Levels — Activos:{liq_sum['active']} "
          f"(EH:{liq_sum['equal_highs_active']} EL:{liq_sum['equal_lows_active']}) "
          f"Premium:{liq_sum['premium']} Barridos:{liq_sum['swept']}")

    if liq_sum['nearest_high']:
        nh = liq_sum['nearest_high']
        print(f"  EqualHigh más reciente: {nh.price:.2f} "
              f"[{nh.touches} toques | {nh.quality_label}]")
    if liq_sum['nearest_low']:
        nl = liq_sum['nearest_low']
        print(f"  EqualLow más reciente : {nl.price:.2f} "
              f"[{nl.touches} toques | {nl.quality_label}]")

    print(f"\n  Sweeps — Total:{sweep_sum['total']} "
          f"Premium:{sweep_sum['premium']} "
          f"(SH:{sweep_sum.get('sweep_highs',0)} SL:{sweep_sum.get('sweep_lows',0)})")
    if recent_sweeps:
        s = recent_sweeps[-1]
        print(f"  Último sweep: {s.sweep_type.upper()} @ {s.level_swept:.2f} "
              f"[{s.sweep_quality}] → probable: {s.probable_direction.upper()}")

    print(f"\n  Inducement — Total:{ind_sum['total']} Premium:{ind_sum['premium']}")
    if ind_sum['last']:
        iz = ind_sum['last']
        print(f"  Último: {iz.inducement_type} @ bar {iz.bar_index} "
              f"[{iz.trap_quality}] → move: {iz.probable_move.upper()}")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    analyze('BOOM1000')
    analyze('CRASH1000')
