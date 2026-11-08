"""
Causal Fixed-Income Module.

Provides structural link assumptions, PCMCI+ causal discovery engine,
cross-regime invariance evaluation, and artifact serialization.
"""

from .priors import EXPECTED_SIGNS, build_link_assumptions
from .engine import CausalEngine, CausalResult
from .macro_discovery import MacroRegime, MacroCausalDiscovery, DEFAULT_REGIMES
from .panel_discovery import PanelCausalDiscovery, PanelDiscoveryResult, PANEL_CAUSAL_VARS

__all__ = [
    "EXPECTED_SIGNS",
    "build_link_assumptions",
    "CausalEngine",
    "CausalResult",
    "MacroRegime",
    "MacroCausalDiscovery",
    "DEFAULT_REGIMES",
    "PanelCausalDiscovery",
    "PanelDiscoveryResult",
    "PANEL_CAUSAL_VARS",
]

