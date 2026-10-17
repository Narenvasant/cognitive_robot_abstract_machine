"""
What can go wrong setting a RoboCasa task up for Tiago.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from krrood.exceptions import DataclassException


@dataclass
class NoClearStandingPose(DataclassException):
    """
    Raised when Tiago touches the kitchen wherever it is stood to reach a target.
    """

    target: np.ndarray
    """
    What it was to reach.
    """

    tried: int
    """
    How many stands were tried.
    """

    def error_message(self) -> str:
        return (
            f"Tiago touches the kitchen at every one of the {self.tried} stands tried "
            f"for reaching {np.round(self.target, 3)}."
        )

    def suggest_correction(self) -> str:
        return "Look further back, or work with the other arm."
