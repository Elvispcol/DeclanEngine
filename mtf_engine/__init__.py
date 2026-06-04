"""
MTF ENGINE
DeclanEngine - Multi-Timeframe Analysis

Análisis real multi-timeframe que reemplaza el MTF simulado del Probability Engine.

El MTF score ya NO es un proxy — es un análisis real de alineación temporal.
"""

from .htf_analyzer import HTFAnalyzer, HTFResult

__all__ = ['HTFAnalyzer', 'HTFResult']
