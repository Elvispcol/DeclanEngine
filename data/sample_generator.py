"""
SAMPLE DATA GENERATOR
DeclanEngine - Data Module

Genera datos OHLC sintéticos para pruebas del motor de estructura.
Simula comportamiento tipo Boom/Crash con ciclos de acumulación,
markup, distribución y markdown.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_boom_crash_data(
    n_candles: int = 500,
    instrument: str = 'BOOM1000',
    seed: int = 42
) -> pd.DataFrame:
    """
    Genera datos OHLC simulados tipo Boom/Crash.
    
    Parameters
    ----------
    n_candles  : Número de candles a generar
    instrument : Nombre del instrumento
    seed       : Semilla para reproducibilidad
    
    Returns
    -------
    DataFrame con columnas: ['time', 'open', 'high', 'low', 'close', 'instrument']
    """
    # Seed diferente por instrumento para comportamiento único
    instrument_seeds = {
        'BOOM1000': seed,
        'BOOM500': seed + 7,
        'CRASH1000': seed + 13,
        'CRASH500': seed + 19,
    }
    np.random.seed(instrument_seeds.get(instrument, seed))

    base_price = 10000.0
    prices = [base_price]

    is_boom = instrument.startswith('BOOM')
    is_crash = instrument.startswith('CRASH')
    is_fast = '500' in instrument   # 500 = más volátil, reversiones más rápidas

    # Parámetros por tipo de instrumento
    spike_prob     = 0.025 if is_fast else 0.015
    spike_mag      = 0.018 if is_fast else 0.013
    trend_strength = 0.0025 if not is_fast else 0.0035

    phase_length = n_candles // 4

    for i in range(1, n_candles):
        phase = (i // phase_length) % 4

        if phase == 0:   # Acumulación
            drift = 0.0001
            vol = 0.003 if not is_fast else 0.005
        elif phase == 1:  # Markup
            drift = trend_strength if is_boom else -trend_strength * 0.5
            vol = 0.004
        elif phase == 2:  # Distribución
            drift = 0.0002 if is_boom else -0.0003
            vol = 0.005 if not is_fast else 0.007
        else:             # Markdown
            drift = -trend_strength * 0.5 if is_boom else -trend_strength
            vol = 0.004

        # Spikes: BOOM sube, CRASH baja
        spike = 0
        if is_boom and np.random.random() < spike_prob:
            spike = abs(np.random.normal(0, spike_mag))
        elif is_crash and np.random.random() < spike_prob:
            spike = -abs(np.random.normal(0, spike_mag))

        change = drift + np.random.normal(0, vol) + spike
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))

    # Construir OHLC
    records = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    for i in range(len(prices) - 1):
        o = prices[i]
        c = prices[i + 1]
        
        candle_vol = abs(c - o) * np.random.uniform(1.0, 2.5)
        h = max(o, c) + abs(np.random.normal(0, candle_vol * 0.3))
        l = min(o, c) - abs(np.random.normal(0, candle_vol * 0.3))

        records.append({
            'time': start_time + timedelta(minutes=i),
            'open': round(o, 5),
            'high': round(h, 5),
            'low': round(l, 5),
            'close': round(c, 5),
            'instrument': instrument
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_boom_crash_data(500, 'BOOM1000')
    df.to_csv('data/sample_boom1000.csv', index=False)
    print(f"Generados {len(df)} candles. Guardado en data/sample_boom1000.csv")
    print(df.head())
