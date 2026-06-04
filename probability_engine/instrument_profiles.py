"""
INSTRUMENT PROFILES
DeclanEngine - Sprint 4 | Probability Engine

Pesos semi-adaptativos por instrumento, según doctrina del GPT especialista:
  - Un framework global con modificadores por instrumento
  - NO sistemas paralelos independientes (imposible calibrar)

Base weights (global):
  structure   : 0.30
  liquidity   : 0.20
  candle      : 0.15
  momentum    : 0.15
  mtf         : 0.20

Modificadores por instrumento aplicados sobre base.
"""


class InstrumentProfiles:
    """
    Retorna pesos calibrados por instrumento.
    """

    BASE_WEIGHTS = {
        "structure": 0.30,
        "liquidity": 0.20,
        "candle":    0.15,
        "momentum":  0.15,
        "mtf":       0.20
    }

    # Modificadores aditivos sobre base — deben sumar 0
    MODIFIERS = {
        "BOOM1000": {
            # Reversiones por absorción/sweep — liquidez y velas más importantes
            "liquidity": +0.08,
            "candle":    +0.05,
            "momentum":  -0.08,
            "structure": -0.03,
            "mtf":       -0.02
        },
        "BOOM500": {
            # Más volátil, reversiones más rápidas — velas dominan
            "candle":    +0.08,
            "liquidity": +0.05,
            "momentum":  -0.08,
            "structure": -0.03,
            "mtf":       -0.02
        },
        "CRASH1000": {
            # Markdown eficiente — estructura y momentum más importantes
            "structure": +0.05,
            "momentum":  +0.08,
            "candle":    -0.05,
            "liquidity": -0.05,
            "mtf":       -0.03
        },
        "CRASH500": {
            # Muy volátil en markdown — momentum lidera
            "momentum":  +0.10,
            "structure": +0.05,
            "candle":    -0.08,
            "liquidity": -0.05,
            "mtf":       -0.02
        }
    }

    @classmethod
    def get_weights(cls, instrument: str) -> dict:
        """
        Retorna pesos finales para el instrumento.
        Normaliza para garantizar que sumen exactamente 1.0.
        """
        weights = cls.BASE_WEIGHTS.copy()
        mods = cls.MODIFIERS.get(instrument, {})

        for key, delta in mods.items():
            weights[key] = max(0.05, weights[key] + delta)

        # Normalizar
        total = sum(weights.values())
        return {k: round(v / total, 4) for k, v in weights.items()}

    @classmethod
    def describe(cls, instrument: str) -> str:
        w = cls.get_weights(instrument)
        return (f"structure={w['structure']:.0%} liquidity={w['liquidity']:.0%} "
                f"candle={w['candle']:.0%} momentum={w['momentum']:.0%} mtf={w['mtf']:.0%}")
