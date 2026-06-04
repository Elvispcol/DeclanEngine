"""
PENALTY ENGINE
DeclanEngine - Penalty Engine

Penaliza contradicciones agresivamente para prevenir scores inflados.
"""

from .penalty_engine import PenaltyEngine, PenaltyAssessment

__all__ = ['PenaltyEngine', 'PenaltyAssessment']
