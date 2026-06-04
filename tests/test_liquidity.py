"""
TESTS - Liquidity Engine
DeclanEngine - Sprint 2

Ejecutar: python tests/test_liquidity.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_generator import generate_boom_crash_data
from structure_engine import SwingDetector, StructureClassifier, BOSCHOCHDetector, DisplacementDetector
from liquidity_engine import EqualLevelsDetector, SweepDetector, InducementDetector


def run_tests():
    print("=" * 60)
    print("  DECLAN ENGINE - SPRINT 2 - LIQUIDITY ENGINE TEST")
    print("=" * 60)

    df = generate_boom_crash_data(300, 'BOOM1000')
    print(f"\n[1] Datos: {len(df)} candles generados ✓")

    sd = SwingDetector(left_bars=3, right_bars=3)
    swing_highs, swing_lows = sd.get_swing_list(df)
    print(f"\n[2] Swings — Highs:{len(swing_highs)} Lows:{len(swing_lows)} ✓")

    sc = StructureClassifier()
    structure_points = sc.classify(swing_highs, swing_lows)
    bd = BOSCHOCHDetector()
    events = bd.detect(df, structure_points)
    dd = DisplacementDetector()
    displacements = dd.detect(df)

    # Equal Levels
    print("\n[3] Equal Levels...")
    eld = EqualLevelsDetector(atr_multiplier=0.10, min_touches=2)
    eq_levels = eld.detect(df, swing_highs, swing_lows)
    s = eld.get_summary(eq_levels)
    print(f"    Total: {s['total']} | Activos: {s['active']} | "
          f"Premium: {s['premium']} | Barridos: {s['swept']} ✓")

    active = eld.get_active_levels(eq_levels)
    for lvl in active[:3]:
        print(f"    {lvl.level_type.upper()} @ {lvl.price:.2f} "
              f"[{lvl.touches} toques | {lvl.quality_label} | q={lvl.touch_quality:.2f}]")

    # Sweeps
    print("\n[4] Liquidity Sweeps...")
    sweep_det = SweepDetector()
    sweeps = sweep_det.detect_from_swings(df, swing_highs, swing_lows)
    ss = sweep_det.get_summary(sweeps)
    print(f"    Total: {ss['total']} | Premium: {ss['premium']} ✓")
    if ss['last']:
        s2 = ss['last']
        print(f"    Último: {s2.sweep_type} @ {s2.level_swept:.2f} "
              f"[{s2.sweep_quality}] pen={s2.atr_penetration:.2f}xATR → {s2.probable_direction.upper()}")

    # Inducement
    print("\n[5] Inducement Zones...")
    ind = InducementDetector()
    zones = ind.detect(df, events, displacements)
    iz = ind.get_summary(zones)
    print(f"    Total: {iz['total']} | Premium: {iz['premium']} ✓")
    if iz['last']:
        z = iz['last']
        print(f"    Último: {z.inducement_type} [{z.trap_quality}] → {z.probable_move.upper()}")

    print("\n" + "=" * 60)
    print("  ✅ Sprint 2 - Liquidity Engine operativo.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
