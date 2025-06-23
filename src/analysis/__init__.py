"""
Analysis submodule for media bias network analysis.

This module provides specialized analysis components while maintaining
backward compatibility with the original ResultsAnalyzer class.
"""

from .core import ResultsAnalyzer

# Maintain backward compatibility - users can still import directly
__all__ = ['ResultsAnalyzer']