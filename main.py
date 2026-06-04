"""
DECLAN ENGINE - Main Entry Point
Sprint 1: Structure Engine

Institutional Synthetic Indices Context Engine
Declan Trader | Porciento Trading

Uso:
    python main.py
"""

from data.sample_generator import generate_boom_crash_data
from structure_engine import (
    SwingDetector,
    StructureClassifier,
    BOSCHOCHDetector,
    DisplacementDetector
)


def analyze(instrument: str = 'BOOM1000', n_candles: int = 300):
    print(f"\n{'='*55}")
    print(f"  DECLAN ENGINE | {instrument}")
    print(f"{'='*55}")

    # Datos
    df = generate_boom_crash_data(n_candles, instrument)

    # Swings
    sd = SwingDetector(left_bars=3, right_bars=3)
    swing_highs, swing_lows = sd.get_swing_list(df)

    # Estructura
    sc = StructureClassifier()
    structure_points = sc.classify(swing_highs, swing_lows)
    summary = sc.get_structure_summary(structure_points)

    # BOS / CHOCH
    bd = BOSCHOCHDetector()
    events = bd.detect(df, structure_points)
    ev_sum = bd.get_events_summary(events)

    # Desplazamiento
    dd = DisplacementDetector()
    displacements = dd.detect(df)
    recent_disp = dd.get_recent_displacement(displacements, lookback=15)

    # Output
    print(f"\n  ESTRUCTURA")
    print(f"  Bias actual : {summary['bias'].upper()}")
    print(f"  HH: {summary['hh_count']}  HL: {summary['hl_count']}  "
          f"LH: {summary['lh_count']}  LL: {summary['ll_count']}")

    print(f"\n  EVENTOS ESTRUCTURALES")
    print(f"  BOS: {ev_sum['bos']}  |  CHOCH: {ev_sum['choch']}")
    if ev_sum['last']:
        e = ev_sum['last']
        print(f"  Último: {e.event_type} {e.direction.upper()} [{e.significance}]")

    print(f"\n  DESPLAZAMIENTO")
    print(f"  Total: {len(displacements)}  |  Recientes: {len(recent_disp)}")
    if recent_disp:
        d = recent_disp[-1]
        print(f"  Último: {d.direction.upper()} [{d.quality}] "
              f"body={d.body_ratio:.0%} rango={d.range_vs_atr:.1f}xATR")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    analyze('BOOM1000')
    analyze('CRASH1000')
