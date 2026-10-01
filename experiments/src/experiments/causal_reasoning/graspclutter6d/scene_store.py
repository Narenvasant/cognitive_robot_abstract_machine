"""
Keeping the built scenes in a database, so that a run reads them from there instead of
from the dataset server and the grasp labels every time.

The scenes are written through the ``experiments`` package's generated ORM interface, one
row per scene, object instance and viewpoint, in the order the dataset lists the scenes;
reading them back gives the same list in the same order, which the seeded splits rely on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from typing_extensions import List, Self

from experiments.causal_reasoning.graspclutter6d.domain import GraspClutterScene


class SceneStoreVariable(StrEnum):
    """
    Environment variables describing where the scenes are kept.
    """

    DATABASE_URI = "GRASPCLUTTER6D_DATABASE_URI"


DEFAULT_DATABASE_FILE = "scenes.sqlite"
"""
The database file kept beside the dataset when no database is named.
"""


@dataclass
class SceneStore:
    """
    The built scenes as rows of a database.
    """

    database_uri: str
    """
    Where the database is; any address SQLAlchemy accepts.
    """

    engine: Engine = field(init=False, repr=False)
    """
    The connection to it.
    """

    def __post_init__(self) -> None:
        self.engine = create_engine(self.database_uri)

    @classmethod
    def from_environment(cls, dataset_directory: Path) -> Self:
        """
        The store the environment names, or a database file beside the dataset.

        :param dataset_directory: Where the dataset is kept on this machine.
        :return: The store.
        """
        uri = os.environ.get(SceneStoreVariable.DATABASE_URI)
        if not uri:
            uri = f"sqlite:///{dataset_directory / DEFAULT_DATABASE_FILE}"
        return cls(database_uri=uri)

    @property
    def scene_count(self) -> int:
        """
        How many scenes the store holds; zero before anything was written.
        """
        from experiments.orm.ormatic_interface import Base, GraspClutterSceneDAO

        if not self.engine.dialect.has_table(
            self.engine.connect(), GraspClutterSceneDAO.__tablename__
        ):
            return 0
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            return len(session.scalars(select(GraspClutterSceneDAO.database_id)).all())

    def write(self, scenes: List[GraspClutterScene]) -> None:
        """
        Replace whatever the store holds with these scenes, in this order.

        :param scenes: The scenes to keep.
        """
        from experiments.orm.ormatic_interface import Base

        drop_database(self.engine)
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            session.add_all([to_dao(scene) for scene in scenes])
            session.commit()

    def read(self) -> List[GraspClutterScene]:
        """
        :return: The scenes the store holds, in the order they were written.
        """
        from experiments.orm.ormatic_interface import GraspClutterSceneDAO

        with Session(self.engine) as session:
            rows = session.scalars(
                select(GraspClutterSceneDAO).order_by(GraspClutterSceneDAO.database_id)
            ).all()
            return [row.from_dao() for row in rows]
