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
    np.random.seed(seed)

    base_price = 10000.0
    prices = [base_price]
    
    # Generar precio base con ciclos
    phase_length = n_candles // 4
    
    for i in range(1, n_candles):
        phase = (i // phase_length) % 4
        
        if phase == 0:   # Acumulación
            drift = 0.0001
            vol = 0.003
        elif phase == 1:  # Markup
            drift = 0.002
            vol = 0.004
        elif phase == 2:  # Distribución
            drift = 0.0002
            vol = 0.005
        else:             # Markdown
            drift = -0.002
            vol = 0.004

        # Spike ocasional para Boom/Crash
        spike = 0
        if instrument.startswith('BOOM') and np.random.random() < 0.02:
            spike = abs(np.random.normal(0, 0.015))
        elif instrument.startswith('CRASH') and np.random.random() < 0.02:
            spike = -abs(np.random.normal(0, 0.015))

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
