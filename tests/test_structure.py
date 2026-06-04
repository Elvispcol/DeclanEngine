"""
TESTS - Structure Engine
DeclanEngine - Sprint 1

Pruebas básicas del motor de estructura.
Ejecutar: python tests/test_structure.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.sample_generator import generate_boom_crash_data
from structure_engine import (
    SwingDetector,
    StructureClassifier,
    BOSCHOCHDetector,
    DisplacementDetector
)


def run_tests():
    print("=" * 60)
    print("  DECLAN ENGINE - SPRINT 1 - TEST SUITE")
    print("=" * 60)

    # Generar datos de prueba
    print("\n[1] Generando datos OHLC simulados (BOOM1000)...")
    df = generate_boom_crash_data(300, 'BOOM1000')
    print(f"    ✓ {len(df)} candles generados")

    # Test Swing Detector
    print("\n[2] Detectando swings...")
    swing_detector = SwingDetector(left_bars=3, right_bars=3)
    swing_highs, swing_lows = swing_detector.get_swing_list(df)
    print(f"    ✓ Swing Highs detectados: {len(swing_highs)}")
    print(f"    ✓ Swing Lows detectados:  {len(swing_lows)}")

    # Test Structure Classifier
    print("\n[3] Clasificando estructura (HH/HL/LH/LL)...")
    classifier = StructureClassifier()
    structure_points = classifier.classify(swing_highs, swing_lows)
    summary = classifier.get_structure_summary(structure_points)

    print(f"    ✓ Puntos de estructura: {summary['total_points']}")
    print(f"    ✓ HH: {summary['hh_count']}  HL: {summary['hl_count']}  "
          f"LH: {summary['lh_count']}  LL: {summary['ll_count']}")
    print(f"    ✓ Bias actual: {summary['bias'].upper()}")

    if summary['last_point']:
        lp = summary['last_point']
        print(f"    ✓ Último punto: {lp.point_type} @ {lp.price:.2f} (bar {lp.bar_index})")

    # Test BOS/CHOCH Detector
    print("\n[4] Detectando BOS y CHOCH...")
    bos_detector = BOSCHOCHDetector(require_close_beyond=True, min_body_ratio=0.4)
    events = bos_detector.detect(df, structure_points)
    events_summary = bos_detector.get_events_summary(events)

    print(f"    ✓ Total eventos: {events_summary['total']}")
    print(f"    ✓ BOS:   {events_summary['bos']}")
    print(f"    ✓ CHOCH: {events_summary['choch']}")

    if events_summary['last']:
        e = events_summary['last']
        print(f"    ✓ Último evento: {e.event_type} {e.direction.upper()} "
              f"@ bar {e.bar_index} | nivel {e.broken_level:.2f} | {e.significance}")

    # Test Displacement Detector
    print("\n[5] Detectando candles de desplazamiento...")
    disp_detector = DisplacementDetector(
        min_body_ratio=0.6,
        min_range_vs_atr=1.2,
        max_opposite_wick_ratio=0.25
    )
    displacements = disp_detector.detect(df)
    recent = disp_detector.get_recent_displacement(displacements, lookback=20)

    premium = [d for d in displacements if d.quality == 'premium']
    standard = [d for d in displacements if d.quality == 'standard']

    print(f"    ✓ Total desplazamientos: {len(displacements)}")
    print(f"    ✓ Premium:  {len(premium)}")
    print(f"    ✓ Standard: {len(standard)}")
    print(f"    ✓ Recientes (últimos 20 bars): {len(recent)}")

    # Resumen Final
    print("\n" + "=" * 60)
    print("  RESUMEN DEL ESTADO ACTUAL DEL MERCADO")
    print("=" * 60)
    print(f"  Instrumento : BOOM1000")
    print(f"  Candles     : {len(df)}")
    print(f"  Bias        : {summary['bias'].upper()}")

    if events:
        last_e = events[-1]
        print(f"  Último BOS/CHOCH: {last_e.event_type} {last_e.direction.upper()} "
              f"[{last_e.significance}]")

    if recent:
        last_d = recent[-1]
        print(f"  Último desplaz.: {last_d.direction.upper()} "
              f"[{last_d.quality}] body={last_d.body_ratio:.0%}")

    print("\n  ✅ Sprint 1 - Structure Engine operativo.\n")


if __name__ == "__main__":
    run_tests()
