"""
RoboCasa tasks narrowed to what Tiago's hand can take: RoboCasa samples their objects
from fewer categories than the tasks they narrow.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import Any

from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)

with robocasa_version_assertions_relaxed():
    from robocasa.environments.kitchen.composite.chopping_food.organize_vegetables import (
        OrganizeVegetables,
    )

# %% what fits between Tiago's fingers


class SmallFruit(StrEnum):
    """
    The kinds of fruit RoboCasa has whose every model is narrower than the gap Tiago's
    open fingers leave, by RoboCasa's names for them.
    """

    APRICOT = "apricot"
    """
    An apricot.
    """

    KIWI = "kiwi"
    """
    A kiwi.
    """

    STRAWBERRY = "strawberry"
    """
    A strawberry.
    """

    DATES = "dates"
    """
    Dates.
    """


# %% the narrowed tasks


class OrganizeSmallFruit(OrganizeVegetables):
    """
    RoboCasa's :class:`OrganizeVegetables`, with two pieces of small fruit for the
    vegetables.
    """

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        configurations = super()._get_obj_cfgs()
        for configuration in configurations:
            if configuration["graspable"]:
                configuration["obj_groups"] = [fruit.value for fruit in SmallFruit]
        return configurations
