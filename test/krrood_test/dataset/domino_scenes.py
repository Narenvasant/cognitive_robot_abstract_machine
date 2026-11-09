"""
Drawing scenes of pieces in a row that topple each other where they touch, together
with the true effects of setting a relation between two of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .example_classes import (
    DominoBand,
    DominoContact,
    DominoEnvironment,
    DominoPiece,
    DominoScene,
)

# %% the process


@dataclass
class DominoSceneGenerator:
    """
    Draws scenes of pieces in a row. Each pair of neighbours is a candidate contact. The
    environment decides how many contacts touch, how close the pieces stand and how
    easily they fall; a fallen piece knocks over each piece it touches with
    :attr:`spillover`, and the target comes up less often the more pairs touch.
    """

    spillover: float = 0.3
    """
    How likely a piece knocked over by itself knocks over a piece it touches.
    """

    piece_counts: tuple[int, ...] = (3, 4, 5)
    """
    The numbers of pieces a scene is drawn with, uniformly.
    """

    touching_rate: dict[DominoEnvironment, float] = field(
        default_factory=lambda: {DominoEnvironment.BIN: 0.6, DominoEnvironment.TABLE: 0.25}
    )
    """
    How likely a candidate contact touches, per environment.
    """

    near_rate: dict[DominoEnvironment, float] = field(
        default_factory=lambda: {DominoEnvironment.BIN: 0.7, DominoEnvironment.TABLE: 0.3}
    )
    """
    How likely a piece stands near the target, per environment.
    """

    fall_rate: dict[DominoBand, float] = field(
        default_factory=lambda: {DominoBand.NEAR: 0.45, DominoBand.FAR: 0.1}
    )
    """
    How likely a piece is knocked over by itself, per band, on a table.
    """

    fall_rate_increase_in_a_bin: float = 0.15
    """
    How much more likely a piece in a bin is knocked over by itself.
    """

    lift_rate: dict[DominoEnvironment, float] = field(
        default_factory=lambda: {DominoEnvironment.BIN: 0.75, DominoEnvironment.TABLE: 0.9}
    )
    """
    How likely the target comes up with no touching pair, per environment.
    """

    lift_rate_loss_per_touching_pair: float = 0.15
    """
    How much less likely the target comes up for each touching pair.
    """

    def scene(
        self,
        random_state: np.random.Generator,
        piece_count: int | None = None,
        touching: dict[int, bool] | None = None,
        bands: dict[int, DominoBand] | None = None,
    ) -> DominoScene:
        """
        :param random_state: Source of randomness.
        :param piece_count: The number of pieces, or ``None`` to draw it.
        :param touching: Contacts set by intervention, by the position of their first
            piece.
        :param bands: Bands set by intervention, by the position of the piece.
        :return: One scene.
        """
        touching = touching or {}
        bands = bands or {}
        environment = list(DominoEnvironment)[random_state.integers(2)]
        if piece_count is None:
            piece_count = int(random_state.choice(self.piece_counts))
        piece_bands = [
            bands.get(
                position,
                (
                    DominoBand.NEAR
                    if random_state.random() < self.near_rate[environment]
                    else DominoBand.FAR
                ),
            )
            for position in range(piece_count)
        ]
        contact_touching = [
            touching.get(
                position,
                bool(random_state.random() < self.touching_rate[environment]),
            )
            for position in range(piece_count - 1)
        ]
        increase = (
            self.fall_rate_increase_in_a_bin
            if environment is DominoEnvironment.BIN
            else 0.0
        )
        fell_by_itself = [
            bool(random_state.random() < self.fall_rate[band] + increase)
            for band in piece_bands
        ]
        disturbed = list(fell_by_itself)
        for position in range(piece_count - 1):
            if not contact_touching[position]:
                continue
            if fell_by_itself[position] and random_state.random() < self.spillover:
                disturbed[position + 1] = True
            if fell_by_itself[position + 1] and random_state.random() < self.spillover:
                disturbed[position] = True
        pieces = [
            DominoPiece(band=band, disturbed=fell)
            for band, fell in zip(piece_bands, disturbed)
        ]
        contacts = [
            DominoContact(pieces[position], pieces[position + 1], value)
            for position, value in enumerate(contact_touching)
        ]
        lift_rate = self.lift_rate[environment] - (
            self.lift_rate_loss_per_touching_pair * sum(contact_touching)
        )
        return DominoScene(
            environment=environment,
            pieces=pieces,
            contacts=contacts,
            target_lifted=bool(random_state.random() < max(lift_rate, 0.0)),
        )

    def scenes(self, scene_count: int, seed: int = 0) -> list[DominoScene]:
        """
        :param scene_count: How many scenes to draw.
        :param seed: Seed of the draw.
        :return: The scenes.
        """
        random_state = np.random.default_rng(seed)
        return [self.scene(random_state) for _ in range(scene_count)]
