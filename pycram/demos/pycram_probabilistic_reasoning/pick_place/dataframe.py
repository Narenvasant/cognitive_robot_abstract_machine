"""
Pick-and-place DataFrame builder.
Run on your machine where PostgreSQL is accessible.

Usage:
    python build_dataframe.py

Output:
    pick_and_place_dataframe.csv
"""

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
        MIN(CASE WHEN d.polymorphic_type='PickUpActionDAO'   THEN dnm.designator_ref_id END) AS pickup_id,
        MIN(CASE WHEN d.polymorphic_type='PlaceActionDAO'    THEN dnm.designator_ref_id END) AS place_id,
        MIN(CASE WHEN d.polymorphic_type='NavigateActionDAO' THEN dnm.designator_ref_id END) AS pick_nav_id,
        MAX(CASE WHEN d.polymorphic_type='NavigateActionDAO' THEN dnm.designator_ref_id END) AS place_nav_id
    FROM "SequentialPlanDAO" sp
    JOIN "PlanMappingDAO_nodes_association"  na  ON na.source_planmappingdao_id = sp.database_id
    JOIN "ActionNodeMappingDAO"             anm ON anm.database_id = na.target_plannodemappingdao_id
    JOIN "DesignatorNodeMappingDAO"         dnm ON dnm.database_id = anm.database_id
    JOIN "DesignatorDescriptionDAO"         d   ON d.database_id   = dnm.designator_ref_id
    WHERE (sp.current_plan_id IS NULL OR sp.current_plan_id != sp.database_id)
      AND d.polymorphic_type IN ('PickUpActionDAO','PlaceActionDAO','NavigateActionDAO')
    GROUP BY sp.database_id
    HAVING
        COUNT(DISTINCT CASE WHEN d.polymorphic_type='PickUpActionDAO'   THEN dnm.designator_ref_id END) = 1
        AND COUNT(DISTINCT CASE WHEN d.polymorphic_type='PlaceActionDAO'    THEN dnm.designator_ref_id END) = 1
        AND COUNT(DISTINCT CASE WHEN d.polymorphic_type='NavigateActionDAO' THEN dnm.designator_ref_id END) = 2
)
SELECT
    pa.plan_id,
    pu.arm                      AS pick_arm,
    v_pn.x                      AS pick_approach_x,
    v_pn.y                      AS pick_approach_y,
    v_ln.x                      AS place_approach_x,
    v_ln.y                      AS place_approach_y,
    v_end.x                     AS milk_end_x,
    v_end.y                     AS milk_end_y,
    v_end.z                     AS milk_end_z
FROM plan_actions pa
JOIN "PickUpActionDAO"   pu       ON pu.database_id        = pa.pickup_id
JOIN "NavigateActionDAO" na_pick  ON na_pick.database_id   = pa.pick_nav_id
JOIN "PoseStampedDAO"    ps_pn    ON ps_pn.database_id     = na_pick.target_location_id
JOIN "PyCramPoseDAO"     po_pn    ON po_pn.database_id     = ps_pn.pose_id
JOIN "PyCramVector3DAO"  v_pn     ON v_pn.database_id      = po_pn.position_id
JOIN "NavigateActionDAO" na_place ON na_place.database_id  = pa.place_nav_id
JOIN "PoseStampedDAO"    ps_ln    ON ps_ln.database_id     = na_place.target_location_id
JOIN "PyCramPoseDAO"     po_ln    ON po_ln.database_id     = ps_ln.pose_id
JOIN "PyCramVector3DAO"  v_ln     ON v_ln.database_id      = po_ln.position_id
JOIN "DesignatorNodeMappingDAO" dnm_pl ON dnm_pl.designator_ref_id = pa.place_id
JOIN "ActionNodeMappingDAO"     anm_pl ON anm_pl.database_id       = dnm_pl.database_id
JOIN "ExecutionDataDAO"         ed     ON ed.database_id            = anm_pl.execution_data_id
JOIN "PoseStampedDAO"    ps_end   ON ps_end.database_id    = ed.manipulated_body_pose_end_id
JOIN "PyCramPoseDAO"     po_end   ON po_end.database_id    = ps_end.pose_id
JOIN "PyCramVector3DAO"  v_end    ON v_end.database_id     = po_end.position_id
WHERE v_end.z < 0.9
ORDER BY pa.plan_id;
"""

print("Connecting...")
engine  = create_engine(DATABASE_URI)
session = Session(engine)
print("Running query...")
result  = session.execute(text(QUERY))
df      = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
session.close()

# Post-process arm column: "pycram.robot_descriptions...Arms.LEFT" -> "LEFT"
df["pick_arm"]    = df["pick_arm"].str.split(".").str[-1]
df["arm_encoded"] = (df["pick_arm"] == "RIGHT").astype(int)

# Column order
df = df[["plan_id",
         "pick_arm", "arm_encoded",
         "pick_approach_x", "pick_approach_y",
         "place_approach_x", "place_approach_y",
         "milk_end_x", "milk_end_y", "milk_end_z"]]

print(f"\nShape        : {df.shape}")
print(f"Null values  : {df.isnull().sum().sum()}")
print(f"\nArm distribution:\n{df['pick_arm'].value_counts().to_string()}")
print(f"\nDescriptive stats:\n{df.describe().round(4).to_string()}")

df.to_csv("pick_and_place_dataframe.csv", index=False)
print("\nSaved -> pick_and_place_dataframe.csv")