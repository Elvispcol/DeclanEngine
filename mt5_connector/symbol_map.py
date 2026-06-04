"""
SYMBOL MAP
DeclanEngine - Sprint 5 | MT5 Connector

Mapeo de nombres internos del engine a símbolos reales de Deriv en MT5.
Los nombres exactos pueden variar según el broker/servidor de Deriv.
"""


class SymbolMap:
    """
    Mapeo entre nombres del engine y símbolos MT5 de Deriv.
    Ajustar si los símbolos en su terminal son diferentes.
    """

    # Nombres exactos en Deriv MT5
    SYMBOLS = {
        'BOOM1000':  'Boom 1000 Index',
        'BOOM500':   'Boom 500 Index',
        'CRASH1000': 'Crash 1000 Index',
        'CRASH500':  'Crash 500 Index',
        'BOOM300':   'Boom 300 Index',
        'CRASH300':  'Crash 300 Index',
        'STEP':      'Step Index',
        'VOL10':     'Volatility 10 Index',
        'VOL25':     'Volatility 25 Index',
        'VOL50':     'Volatility 50 Index',
        'VOL75':     'Volatility 75 Index',
        'VOL100':    'Volatility 100 Index',
    }

    # Timeframes MT5
    TIMEFRAMES = {
        'M1':  1,
        'M5':  5,
        'M15': 15,
        'M30': 30,
        'H1':  16385,
        'H4':  16388,
        'D1':  16408,
    }

    @classmethod
    def get_symbol(cls, instrument: str) -> str:
        return cls.SYMBOLS.get(instrument.upper(), instrument)

    @classmethod
    def get_timeframe(cls, tf: str) -> int:
        return cls.TIMEFRAMES.get(tf.upper(), 5)

    @classmethod
    def list_instruments(cls) -> list:
        return list(cls.SYMBOLS.keys())
