"""Attention-direction detection for smart-home interaction.

The base package deliberately imports only dependency-light modules. Heavy model and
hardware dependencies are loaded by their command modules only when needed.
"""

from data_preparation.schema import CLASS_NAMES, NODE_NAMES, WINDOW_SIZE

__all__ = ["CLASS_NAMES", "NODE_NAMES", "WINDOW_SIZE"]
__version__ = "1.0.0"
