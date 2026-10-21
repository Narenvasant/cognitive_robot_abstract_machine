"""
Where a causal query finds the causal structure learned for the class it asks about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from probabilistic_model.causal_structure.learning import LearnedCausalStructure
from typing_extensions import Type

from krrood.parametrization.parameterizer import ModelQueryParameters

# %% registries


@dataclass
class CausalStructureRegistry(ABC):
    """
    Resolves a query to the causal structure learned for the class it asks about.
    """

    @abstractmethod
    def get_structure(self, parameters: ModelQueryParameters) -> LearnedCausalStructure:
        """
        :param parameters: The parameters extracted from the queried statement.
        :return: The structure learned for the queried class.
        """


@dataclass
class DictCausalStructureRegistry(CausalStructureRegistry):
    """
    Maps each queried class to the causal structure learned for it.
    """

    structures: dict[Type, LearnedCausalStructure]
    """
    The learned structure of each class.
    """

    def get_structure(self, parameters: ModelQueryParameters) -> LearnedCausalStructure:
        return self.structures[parameters.queried_class]
