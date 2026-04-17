import os
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine

DATABASE_URI = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "postgresql://semantic_digital_twin:naren@localhost:5432/demo_robot_plans",
)

QUERY = """
WITH plan_actions AS (
    SELECT
        sp.database_id AS plan_id,
        sp.current_node_id,

        MIN(CASE WHEN d.polymorphic_type='PickUpActionDAO'
            THEN dnm.designator_ref_id END) AS pickup_id,

        MIN(CASE WHEN d.polymorphic_type='PlaceActionDAO'
            THEN dnm.designator_ref_id END) AS place_id,

        MIN(CASE WHEN d.polymorphic_type='NavigateActionDAO'
            THEN dnm.designator_ref_id END) AS pick_nav_id,

        MAX(CASE WHEN d.polymorphic_type='NavigateActionDAO'
            THEN dnm.designator_ref_id END) AS place_nav_id

    FROM "SequentialPlanDAO" sp

    JOIN "PlanNodeMappingDAO" pn
        ON pn.database_id = sp.current_node_id

    JOIN "PlanEdgeDAO" pe
        ON pe.parent_id = pn.database_id

    JOIN "PlanNodeMappingDAO" child
        ON child.database_id = pe.child_id

    JOIN "ActionNodeMappingDAO" anm
        ON anm.database_id = child.database_id

    JOIN "DesignatorNodeMappingDAO" dnm
        ON dnm.database_id = anm.database_id

    JOIN "DesignatorDescriptionDAO" d
        ON d.database_id = dnm.designator_ref_id

    WHERE d.polymorphic_type IN (
        'PickUpActionDAO',
        'PlaceActionDAO',
        'NavigateActionDAO'
    )

    GROUP BY sp.database_id, sp.current_node_id
)

SELECT
    pa.plan_id,

    pu.arm AS pick_arm,

    v_pick.x AS pick_x,
    v_pick.y AS pick_y,

    v_place.x AS place_x,
    v_place.y AS place_y,

    v_end.x AS end_x,
    v_end.y AS end_y,
    v_end.z AS end_z

FROM plan_actions pa

JOIN "PickUpActionDAO" pu
    ON pu.database_id = pa.pickup_id

JOIN "NavigateActionDAO" np
    ON np.database_id = pa.pick_nav_id

JOIN "PoseMappingDAO" pm_pick
    ON pm_pick.database_id = np.target_location_id

JOIN "Vector3MappingDAO" v_pick
    ON v_pick.database_id = pm_pick.position_id

JOIN "NavigateActionDAO" np2
    ON np2.database_id = pa.place_nav_id

JOIN "PoseMappingDAO" pm_place
    ON pm_place.database_id = np2.target_location_id

JOIN "Vector3MappingDAO" v_place
    ON v_place.database_id = pm_place.position_id

JOIN "PlaceActionDAO" pl
    ON pl.database_id = pa.place_id

JOIN "PoseMappingDAO" pm_end
    ON pm_end.database_id = pl.target_location_id

JOIN "Vector3MappingDAO" v_end
    ON v_end.database_id = pm_end.position_id

WHERE v_end.z < 0.9
ORDER BY pa.plan_id;
"""

print("Connecting...")
engine = create_engine(DATABASE_URI)
session = Session(engine)

print("Running query...")
result = session.execute(text(QUERY))

df = pd.DataFrame(result.fetchall(), columns=result.keys())
session.close()

# -----------------------
# post process
# -----------------------
df["pick_arm"] = df["pick_arm"].str.split(".").str[-1]
df["arm_encoded"] = (df["pick_arm"] == "RIGHT").astype(int)

df = df[
    [
        "plan_id",
        "pick_arm",
        "arm_encoded",
        "pick_x",
        "pick_y",
        "place_x",
        "place_y",
        "end_x",
        "end_y",
        "end_z",
    ]
]

print(f"\nShape: {df.shape}")
print(f"Nulls: {df.isnull().sum().sum()}")
print(df["pick_arm"].value_counts())

df.to_csv("pick_and_place_dataframe.csv", index=False)
print("Saved -> pick_and_place_dataframe.csv")