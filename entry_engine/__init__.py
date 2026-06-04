"""
ENTRY ENGINE
DeclanEngine - Entry Engine

Genera setups de entrada basados en liquidez y estructura.
Core Conditions + Confirmation Layers.
SL y TP basados en liquidez (nunca distancia fija).
"""

from .entry_engine import EntryEngine, SniperEntry

__all__ = ['EntryEngine', 'SniperEntry']
