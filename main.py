"""
DECLAN ENGINE - Main Entry Point
Decision Engine Implementation

Motor de interpretación institucional → Decisiones operacionales para traders.

Declan Trader | Porciento Trading
"""

from data.sample_generator import generate_boom_crash_data
from structure_engine import SwingDetector, StructureClassifier, BOSCHOCHDetector, DisplacementDetector
from liquidity_engine import EqualLevelsDetector, SweepDetector, InducementDetector
from candle_engine import PressureAnalyzer, MomentumDetector, CompressionDetector
from probability_engine import ProbabilityEngine
from decision_engine import DecisionEngine, OutputFormatter


def analyze(instrument: str = 'BOOM1000', n_candles: int = 300):
    df = generate_boom_crash_data(n_candles, instrument)

    # ── Sprint 1 — Structure Engine ────────────────────────────────
    sd   = SwingDetector(left_bars=3, right_bars=3)
    sh, sl = sd.get_swing_list(df)
    sc   = StructureClassifier()
    sp   = sc.classify(sh, sl)
    sm   = sc.get_structure_summary(sp)
    bd   = BOSCHOCHDetector()
    evts = bd.detect(df, sp)
    ev   = bd.get_events_summary(evts)
    dd   = DisplacementDetector()
    dsps = dd.detect(df)
    rd   = dd.get_recent_displacement(dsps, lookback=15)

    # ── Sprint 2 — Liquidity Engine ────────────────────────────────
    eld  = EqualLevelsDetector()
    eql  = eld.detect(df, sh, sl)
    ls   = eld.get_summary(eql)
    swd  = SweepDetector()
    swp  = swd.detect_from_swings(df, sh, sl)
    ss   = swd.get_summary(swp)
    rswp = swd.get_recent_sweeps(swp, lookback=20)
    ind  = InducementDetector()
    izs  = ind.detect(df, evts, dsps)
    iz   = ind.get_summary(izs)

    # ── Sprint 3 — Candle Engine ───────────────────────────────────
    pa   = PressureAnalyzer(window_size=5)
    rdgs = pa.analyze_all(df)
    wp   = pa.get_current_pressure(df, rdgs)
    md   = MomentumDetector(window=6)
    msts = md.detect(df)
    mom  = md.get_current(msts)
    cd   = CompressionDetector()
    csts = cd.detect(df)
    cmp  = cd.get_summary(csts)

    # ── Sprint 4 — Probability Engine ──────────────────────────────
    pe = ProbabilityEngine(instrument=instrument, min_score=60.0)
    score = pe.calculate(
        structure_summary   = sm,
        events_summary      = ev,
        liquidity_summary   = ls,
        sweep_summary       = ss,
        inducement_summary  = iz,
        window_pressure     = wp,
        momentum_state      = mom,
        compression_state   = cmp,
        recent_displacement = rd,
        recent_sweeps       = rswp,
    )

    # ── Decision Engine ────────────────────────────────────────────
    # Extract data needed for Decision Engine
    last_event_type = ev['last'].event_type if ev['last'] else ''
    last_event_sig = ev['last'].significance if ev['last'] else ''
    last_event_dir = ev['last'].direction if ev['last'] else ''

    has_displacement = len(rd) > 0
    displacement_direction = rd[-1].direction if rd else ''

    sweep_present = len(rswp) > 0
    sweep_direction = rswp[-1].sweep_type if rswp else ''
    sweep_quality = rswp[-1].sweep_quality if rswp else ''
    sweep_level = rswp[-1].level_swept if rswp else 0.0

    inducement_present = iz['total'] > 0
    inducement_direction = iz['last'].probable_move if iz['last'] else ''

    # Get ATR for entry calculations
    atr = _compute_current_atr(df)

    # Current price
    current_price = df['close'].iloc[-1]

    # Active equal levels for TP calculation
    active_levels = eld.get_active_levels(eql)

    # Recent swing highs/lows for SL
    swing_highs = sh['price'].tolist() if len(sh) > 0 else []
    swing_lows = sl['price'].tolist() if len(sl) > 0 else []

    # Rejection quality from pressure
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
    print(formatter.format(decision, instrument))

    # Debug output (optional — descomentar para desarrollo)
    # print(formatter.format_debug(decision))


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


if __name__ == "__main__":
    analyze('BOOM1000')
    analyze('CRASH1000')
