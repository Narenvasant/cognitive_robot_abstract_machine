"""
Full Database Export — All DAOs and Their Values

Exports every table and every DAO stored in the PostgreSQL database used by
the pick-and-place experiments, including all semantic digital twin ORM
tables, pycram plan tables, and any other tables present in the schema.

The export produces a structured, self-contained directory that can be
copied directly to a USB stick. Every table is exported as a CSV. Binary
columns (numpy arrays, trimesh geometry) are decoded into human-readable
form where possible. A full pg_dump SQL file is included for complete
database restoration on any other machine.

DAO Hierarchy stored by the demos
----------------------------------
The semantic digital twin ORM stores the following DAO types, all of which
are exported:

    SequentialPlanDAO               — top-level plan records (one per iteration)
    WorldMappingDAO                 — full world snapshots
    WorldStateMappingDAO            — joint state vectors (DOF values + UUIDs)
    Point3MappingDAO                — 3D positions (x, y, z + reference frame)
    QuaternionMappingDAO            — orientations (x, y, z, w + reference frame)
    HomogeneousTransformationMatrixMappingDAO — 4x4 transforms (position + rotation)
    PoseMappingDAO                  — poses (position + orientation)
    Vector3MappingDAO               — 3D vectors
    RotationMatrixMappingDAO        — rotation matrices (stored as quaternion)
    KinematicStructureEntityDAO     — bodies and connections in the world
    ConnectionDAO / subtype DAOs    — joints, fixed connections, OmniDrive, etc.
    DegreeOfFreedomDAO              — revolute/prismatic DOFs with limits
    SemanticAnnotationDAO           — Table, Milk, and other semantic labels
    ... and all other tables discovered in the schema

Output Structure
----------------
    database_export/
        00_export_summary.txt           — manifest, row counts, DAO descriptions
        01_table_inventory.txt          — all tables with column schemas
        plans/
            SequentialPlanDAO.csv       — primary plan records
        world/
            WorldMappingDAO.csv
            WorldStateMappingDAO.csv
            KinematicStructureEntityDAO.csv (or equivalent)
            ConnectionDAO.csv (and subtypes)
            DegreeOfFreedomDAO.csv
            SemanticAnnotationDAO.csv (and subtypes)
        spatial/
            Point3MappingDAO.csv
            QuaternionMappingDAO.csv
            HomogeneousTransformationMatrixMappingDAO.csv
            PoseMappingDAO.csv
            Vector3MappingDAO.csv
            RotationMatrixMappingDAO.csv
        all_tables/
            <every_table>.csv           — one CSV per table, unfiltered
        full_database_dump.sql          — pg_dump for full restoration

Usage
-----
    python export_database_full.py

    SEMANTIC_DIGITAL_TWIN_DATABASE_URI=postgresql://user:pass@host/db \
        python export_database_full.py

    EXPORT_OUTPUT_DIR=/path/to/output python export_database_full.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session


# =============================================================================
# Configuration
# =============================================================================

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "postgresql://semantic_digital_twin:naren@localhost:5432/probabilistic_reasoning",
)

OUTPUT_DIRECTORY: str = os.environ.get(
    "EXPORT_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "database_export"),
)

ZIP_OUTPUT_PATH: str = OUTPUT_DIRECTORY.rstrip("/\\") + ".zip"

# DAO group assignments: each key is a subdirectory name, the value is a list
# of table name substrings. Tables matching any substring are placed in that
# group. Unmatched tables go into all_tables/ only.
DAO_GROUPS: Dict[str, List[str]] = {
    "plans": [
        "SequentialPlan",
        "ActionDescription",
        "NavigateAction",
        "PickUpAction",
        "PlaceAction",
        "ParkArms",
    ],
    "world": [
        "WorldMapping",
        "WorldState",
        "KinematicStructure",
        "Connection",
        "DegreeOfFreedom",
        "SemanticAnnotation",
        "Body",
        "World",
        "Table",
        "Milk",
    ],
    "spatial": [
        "Point3Mapping",
        "QuaternionMapping",
        "HomogeneousTransformation",
        "PoseMapping",
        "Vector3Mapping",
        "RotationMatrix",
        "Pose",
    ],
}

# Known binary or structured column types that need special handling
BINARY_COLUMN_HINTS: List[str] = [
    "data",
    "ids",
    "rotation",
    "matrix",
    "mesh",
    "visual",
    "collision",
    "geometry",
]


# =============================================================================
# DAO Schema Documentation
# =============================================================================

DAO_DESCRIPTIONS: Dict[str, str] = {
    "SequentialPlanDAO": (
        "Top-level plan record. One row per successful pick-and-place execution. "
        "Contains references to all child action nodes in the plan graph."
    ),
    "WorldMappingDAO": (
        "Full world snapshot serialised at plan time. Contains references to all "
        "kinematic structure entities, connections, semantic annotations, degrees "
        "of freedom, and the current world state."
    ),
    "WorldStateMappingDAO": (
        "Serialised joint state vector. 'data' column contains a flat array of "
        "DOF values; 'ids' column contains the corresponding DOF UUIDs in the "
        "same order."
    ),
    "Point3MappingDAO": (
        "3D position (x, y, z) with an optional reference frame body UUID. "
        "Used for approach positions, place targets, and waypoints."
    ),
    "QuaternionMappingDAO": (
        "Orientation as a unit quaternion (x, y, z, w) with optional reference "
        "frame. Used for all pose orientations in the world."
    ),
    "HomogeneousTransformationMatrixMappingDAO": (
        "4x4 rigid body transform stored as position (Point3) + rotation "
        "(Quaternion) + reference frame + child frame. Used for connection "
        "origins and robot pose."
    ),
    "PoseMappingDAO": (
        "Pose combining a Point3 position and Quaternion orientation with an "
        "optional reference frame. Used for NavigateAction target locations "
        "and PlaceAction target poses."
    ),
    "Vector3MappingDAO": (
        "3D vector (x, y, z) with optional reference frame. Used for forces, "
        "velocities, and direction vectors."
    ),
    "RotationMatrixMappingDAO": (
        "Rotation matrix stored as a Quaternion with optional reference frame."
    ),
}


# =============================================================================
# Utilities
# =============================================================================

def _print_section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def _ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def _safe_filename(table_name: str) -> str:
    """Convert a table name to a safe filename, preserving readability."""
    return table_name.replace('"', "").replace("/", "_").replace(" ", "_")


def _classify_table(table_name: str) -> Optional[str]:
    """
    Return the group subdirectory for a table name, or None if unclassified.
    Matching is case-insensitive substring search.
    """
    lower = table_name.lower()
    for group_name, patterns in DAO_GROUPS.items():
        for pattern in patterns:
            if pattern.lower() in lower:
                return group_name
    return None


# =============================================================================
# Database Connection
# =============================================================================

def _connect(database_uri: str) -> Tuple[Any, Session]:
    """
    Establish a SQLAlchemy engine and session.

    Returns
    -------
    tuple[Engine, Session]
    """
    print(f"[export] Connecting to: {database_uri}")
    engine  = create_engine(database_uri, pool_pre_ping=True)
    session = Session(engine)
    session.execute(text("SELECT 1"))
    print("[export] Connection verified.")
    return engine, session


# =============================================================================
# Schema Inspection
# =============================================================================

def _inspect_full_schema(engine: Any) -> Dict[str, Any]:
    """
    Inspect the complete database schema: all tables, their columns, types,
    primary keys, foreign keys, and indices.

    Returns a dictionary keyed by table name, each value being a dict of
    column metadata.
    """
    inspector = inspect(engine)
    schema: Dict[str, Any] = {}

    for table_name in sorted(inspector.get_table_names()):
        columns      = inspector.get_columns(table_name)
        primary_keys = inspector.get_pk_constraint(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        indices      = inspector.get_indexes(table_name)

        schema[table_name] = {
            "columns":      columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "indices":      indices,
        }

    return schema


def _write_table_inventory(
    output_directory: str,
    schema:           Dict[str, Any],
    row_counts:       Dict[str, int],
) -> None:
    """
    Write a detailed table inventory including column names, types, primary
    keys, and foreign key relationships for every table in the database.
    """
    lines = [
        "Database Table Inventory",
        "=" * 72,
        f"Generated : {datetime.datetime.now().isoformat()}",
        f"Database  : {DATABASE_URI}",
        f"Tables    : {len(schema)}",
        f"Total rows: {sum(c for c in row_counts.values() if c >= 0)}",
        "",
    ]

    for table_name, table_info in sorted(schema.items()):
        row_count = row_counts.get(table_name, -1)
        group     = _classify_table(table_name)
        dao_desc  = DAO_DESCRIPTIONS.get(table_name, "")

        lines += [
            f"TABLE: {table_name}",
            f"  Rows    : {row_count if row_count >= 0 else 'ERROR'}",
            f"  Group   : {group or 'all_tables'}",
        ]
        if dao_desc:
            lines.append(f"  Purpose : {dao_desc}")

        lines.append(f"  Columns :")
        for column in table_info["columns"]:
            nullable = "nullable" if column.get("nullable", True) else "not null"
            lines.append(
                f"    {column['name']:<40} "
                f"{str(column['type']):<25} {nullable}"
            )

        primary_key_columns = table_info["primary_keys"].get("constrained_columns", [])
        if primary_key_columns:
            lines.append(f"  Primary key: {', '.join(primary_key_columns)}")

        for foreign_key in table_info["foreign_keys"]:
            local_cols  = ", ".join(foreign_key.get("constrained_columns", []))
            ref_table   = foreign_key.get("referred_table", "?")
            ref_cols    = ", ".join(foreign_key.get("referred_columns", []))
            lines.append(f"  Foreign key: {local_cols} -> {ref_table}.{ref_cols}")

        lines.append("")

    inventory_path = os.path.join(output_directory, "01_table_inventory.txt")
    _write_text(inventory_path, "\n".join(lines))
    print(f"[export] Table inventory written: {inventory_path}")


# =============================================================================
# Row Counting
# =============================================================================

def _count_all_rows(
    session:     Session,
    table_names: List[str],
) -> Dict[str, int]:
    """Return row counts for all tables. Tables that fail return -1."""
    row_counts: Dict[str, int] = {}
    for table_name in table_names:
        try:
            count = session.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            ).scalar()
            row_counts[table_name] = int(count)
        except Exception as error:
            print(f"[export] WARNING: count failed for {table_name}: {error}")
            session.rollback()
            row_counts[table_name] = -1
    return row_counts


# =============================================================================
# Column Value Decoding
# =============================================================================

def _decode_binary_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to decode binary and structured columns into human-readable form.

    Columns matching BINARY_COLUMN_HINTS that contain bytes or memoryview
    objects are decoded as follows:
        - JSON bytes -> expanded into readable string
        - Raw bytes  -> hex string with byte length annotation
        - numpy arrays stored as bytes -> decoded via numpy frombuffer

    This is a best-effort operation. Columns that cannot be decoded are
    left unchanged and noted in the column header via rename.
    """
    import numpy as np

    decoded = dataframe.copy()

    for column_name in decoded.columns:
        if not any(hint in column_name.lower() for hint in BINARY_COLUMN_HINTS):
            continue

        sample = decoded[column_name].dropna()
        if sample.empty:
            continue

        first_value = sample.iloc[0]

        if isinstance(first_value, (bytes, memoryview)):
            raw_bytes = bytes(first_value) if isinstance(first_value, memoryview) else first_value

            # Attempt JSON decode
            try:
                json.loads(raw_bytes)

                def decode_json(value: Any) -> str:
                    if value is None:
                        return ""
                    try:
                        return json.dumps(json.loads(bytes(value)), indent=None)
                    except Exception:
                        return f"<binary {len(bytes(value))} bytes>"

                decoded[column_name] = decoded[column_name].apply(decode_json)
                continue
            except Exception:
                pass

            # Attempt numpy float64 array decode
            try:
                np.frombuffer(raw_bytes, dtype=np.float64)

                def decode_numpy(value: Any) -> str:
                    if value is None:
                        return ""
                    try:
                        array = np.frombuffer(bytes(value), dtype=np.float64)
                        return "[" + ", ".join(f"{v:.6f}" for v in array) + "]"
                    except Exception:
                        return f"<binary {len(bytes(value))} bytes>"

                decoded[column_name] = decoded[column_name].apply(decode_numpy)
                continue
            except Exception:
                pass

            # Fallback: hex representation
            def decode_hex(value: Any) -> str:
                if value is None:
                    return ""
                raw = bytes(value) if isinstance(value, memoryview) else value
                return f"<{len(raw)} bytes: {raw[:32].hex()}{'...' if len(raw) > 32 else ''}>"

            decoded[column_name] = decoded[column_name].apply(decode_hex)

    return decoded


# =============================================================================
# Single Table Export
# =============================================================================

def _export_table(
    session:     Session,
    table_name:  str,
    output_path: str,
) -> Tuple[int, bool]:
    """
    Export a single table to a CSV file with binary column decoding.

    Returns (row_count, success).
    """
    try:
        dataframe = pd.read_sql(
            sql=text(f'SELECT * FROM "{table_name}"'),
            con=session.bind,
        )
        decoded_dataframe = _decode_binary_columns(dataframe)
        decoded_dataframe.to_csv(output_path, index=False, encoding="utf-8")
        return len(dataframe), True

    except Exception as error:
        print(f"[export] WARNING: export failed for {table_name}: {error}")
        session.rollback()
        _write_text(output_path, f"Export failed: {error}\n")
        return -1, False


# =============================================================================
# Grouped Export
# =============================================================================

def _export_all_tables(
    session:          Session,
    schema:           Dict[str, Any],
    output_directory: str,
) -> Dict[str, int]:
    """
    Export every table to both its group subdirectory and the all_tables/
    directory. Returns a mapping of table name to exported row count.
    """
    all_tables_directory = os.path.join(output_directory, "all_tables")
    _ensure_directory(all_tables_directory)

    # Ensure all group directories exist
    for group_name in DAO_GROUPS:
        _ensure_directory(os.path.join(output_directory, group_name))

    exported_counts: Dict[str, int] = {}
    table_names = sorted(schema.keys())

    print(f"\n[export] Exporting {len(table_names)} tables ...")

    for table_name in table_names:
        safe_name    = _safe_filename(table_name) + ".csv"
        all_tables_path = os.path.join(all_tables_directory, safe_name)

        row_count, success = _export_table(session, table_name, all_tables_path)
        exported_counts[table_name] = row_count

        status = f"{row_count} rows" if success else "FAILED"
        print(f"[export]   {table_name:<55} {status}")

        # Also copy to the appropriate group directory
        group = _classify_table(table_name)
        if group is not None:
            group_path = os.path.join(output_directory, group, safe_name)
            if success:
                shutil.copy2(all_tables_path, group_path)

    print(f"[export] All tables exported to {all_tables_directory}")
    return exported_counts


# =============================================================================
# Schema JSON Export
# =============================================================================

def _export_schema_json(
    output_directory: str,
    schema:           Dict[str, Any],
    row_counts:       Dict[str, int],
) -> None:
    """
    Write the full schema as a machine-readable JSON file. Column types are
    serialised as strings since SQLAlchemy type objects are not JSON-serialisable.
    """
    serialisable_schema: Dict[str, Any] = {}

    for table_name, table_info in schema.items():
        serialisable_schema[table_name] = {
            "row_count": row_counts.get(table_name, -1),
            "group":     _classify_table(table_name) or "all_tables",
            "columns": [
                {
                    "name":     col["name"],
                    "type":     str(col["type"]),
                    "nullable": col.get("nullable", True),
                }
                for col in table_info["columns"]
            ],
            "primary_keys": table_info["primary_keys"].get("constrained_columns", []),
            "foreign_keys": [
                {
                    "local":      fk.get("constrained_columns", []),
                    "references": fk.get("referred_table", ""),
                    "columns":    fk.get("referred_columns", []),
                }
                for fk in table_info["foreign_keys"]
            ],
        }

    schema_path = os.path.join(output_directory, "schema.json")
    with open(schema_path, "w", encoding="utf-8") as file:
        json.dump(serialisable_schema, file, indent=2)
    print(f"[export] Schema JSON written: {schema_path}")


# =============================================================================
# PostgreSQL Dump
# =============================================================================

def _run_pg_dump(database_uri: str, output_path: str) -> bool:
    """
    Run pg_dump to produce a portable plain-text SQL dump.

    Uses --no-owner and --no-acl so the dump can be restored to any
    PostgreSQL instance regardless of the original role configuration.

    Returns True on success.
    """
    if shutil.which("pg_dump") is None:
        print(
            "[export] pg_dump not found on PATH — SQL dump skipped.\n"
            "[export] Install postgresql-client to enable full dumps."
        )
        _write_text(
            output_path,
            "pg_dump was not available on PATH.\n"
            "To produce a full SQL dump, install postgresql-client and re-run.\n"
            "Alternatively, use the CSV files in all_tables/ for data access.\n",
        )
        return False

    print("[export] Running pg_dump ...")
    try:
        result = subprocess.run(
            [
                "pg_dump",
                "--no-owner",
                "--no-acl",
                "--format=plain",
                "--file", output_path,
                database_uri,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[export] SQL dump written: {output_path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"[export] pg_dump failed (exit {result.returncode}): {result.stderr[:500]}")
            _write_text(
                output_path,
                f"pg_dump failed (exit {result.returncode}).\n"
                f"stderr:\n{result.stderr}\n",
            )
            return False
    except subprocess.TimeoutExpired:
        print("[export] pg_dump timed out after 600 seconds.")
        return False
    except Exception as error:
        print(f"[export] pg_dump exception: {error}")
        return False


# =============================================================================
# Export Summary
# =============================================================================

def _write_export_summary(
    output_directory: str,
    schema:           Dict[str, Any],
    row_counts:       Dict[str, int],
    exported_counts:  Dict[str, int],
    dump_succeeded:   bool,
    export_started:   datetime.datetime,
) -> None:
    """
    Write a comprehensive human-readable export summary that serves as the
    manifest for the archive. Includes DAO descriptions, row counts per table,
    group assignments, and restoration instructions.
    """
    export_finished   = datetime.datetime.now()
    duration_seconds  = (export_finished - export_started).total_seconds()
    total_rows        = sum(c for c in row_counts.values() if c >= 0)
    total_exported    = sum(c for c in exported_counts.values() if c >= 0)
    tables_by_group: Dict[str, List[str]] = {}

    for table_name in sorted(schema.keys()):
        group = _classify_table(table_name) or "all_tables"
        tables_by_group.setdefault(group, []).append(table_name)

    lines = [
        "=" * 72,
        "  Full Database Export — Semantic Digital Twin / Pick-and-Place",
        "  Causally-Aware Robot Action Verification via Interventional",
        "  Probabilistic Circuits — SPAI @ IJCAI 2026",
        "=" * 72,
        "",
        f"  Source database   : {DATABASE_URI}",
        f"  Export started    : {export_started.isoformat()}",
        f"  Export finished   : {export_finished.isoformat()}",
        f"  Duration          : {duration_seconds:.1f}s",
        "",
        f"  Tables discovered : {len(schema)}",
        f"  Total rows        : {total_rows}",
        f"  Rows exported     : {total_exported}",
        f"  SQL dump          : {'yes (full_database_dump.sql)' if dump_succeeded else 'not available'}",
        "",
        "=" * 72,
        "  Archive Structure",
        "=" * 72,
        "",
        "  00_export_summary.txt       — this file",
        "  01_table_inventory.txt      — full column schema for every table",
        "  schema.json                 — machine-readable schema",
        "  full_database_dump.sql      — pg_dump (restore with psql)",
        "  plans/                      — plan execution DAOs",
        "  world/                      — world, body, connection, annotation DAOs",
        "  spatial/                    — pose, transform, quaternion DAOs",
        "  all_tables/                 — every table as CSV (unfiltered)",
        "",
        "=" * 72,
        "  DAO Descriptions",
        "=" * 72,
        "",
    ]

    for dao_name, description in DAO_DESCRIPTIONS.items():
        row_count = row_counts.get(dao_name, "not present")
        lines += [
            f"  {dao_name}",
            f"    Rows    : {row_count}",
            f"    Purpose : {description}",
            "",
        ]

    lines += [
        "=" * 72,
        "  Row Counts by Group",
        "=" * 72,
        "",
    ]

    for group_name in sorted(tables_by_group.keys()):
        group_tables = tables_by_group[group_name]
        group_total  = sum(row_counts.get(t, 0) for t in group_tables if row_counts.get(t, 0) >= 0)
        lines.append(f"  {group_name}/  ({group_total} total rows)")

        for table_name in sorted(group_tables):
            count = row_counts.get(table_name, -1)
            count_display = str(count) if count >= 0 else "ERROR"
            lines.append(f"    {table_name:<58} {count_display:>8} rows")
        lines.append("")

    lines += [
        "=" * 72,
        "  Restoration Instructions",
        "=" * 72,
        "",
        "  Full restore from SQL dump:",
        "    createdb target_database",
        "    psql target_database < full_database_dump.sql",
        "",
        "  Load a specific table in Python:",
        "    import pandas as pd",
        "    df = pd.read_csv('all_tables/SequentialPlanDAO.csv')",
        "",
        "  Load from database directly:",
        "    from sqlalchemy import create_engine, text",
        "    engine = create_engine('<uri>')",
        "    df = pd.read_sql(text('SELECT * FROM \"SequentialPlanDAO\"'), engine)",
        "",
        "=" * 72,
    ]

    summary_path = os.path.join(output_directory, "00_export_summary.txt")
    _write_text(summary_path, "\n".join(lines))
    print(f"[export] Export summary written: {summary_path}")


# =============================================================================
# Archive Creation
# =============================================================================

def _create_zip_archive(source_directory: str, zip_path: str) -> float:
    """
    Create a ZIP archive of the export directory.
    Returns the archive size in megabytes.
    """
    directory_name = os.path.basename(source_directory.rstrip("/\\"))
    print(f"\n[export] Creating ZIP archive: {zip_path} ...")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(source_directory):
            for filename in files:
                absolute_path = os.path.join(root, filename)
                relative_path = os.path.relpath(absolute_path, source_directory)
                archive_path  = os.path.join(directory_name, relative_path)
                archive.write(absolute_path, archive_path)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[export] Archive created: {zip_path} ({size_mb:.1f} MB)")
    return size_mb


# =============================================================================
# Main Entry Point
# =============================================================================

def export_full_database() -> None:
    """
    Export every DAO and its values from the experiment PostgreSQL database.

    Steps
    -----
    1. Connect and verify the database is reachable.
    2. Inspect the full schema: all tables, columns, types, foreign keys.
    3. Count rows in every table.
    4. Export every table as a decoded CSV to all_tables/ and its group dir.
    5. Export the full schema as schema.json for programmatic access.
    6. Write a detailed table inventory (01_table_inventory.txt).
    7. Run pg_dump for a complete restorable SQL dump.
    8. Write the export summary manifest (00_export_summary.txt).
    9. Compress everything into database_export.zip.
    """
    export_started = datetime.datetime.now()

    _print_section("Full Database Export")
    print(f"  Database  : {DATABASE_URI}")
    print(f"  Output    : {OUTPUT_DIRECTORY}")
    print(f"  Archive   : {ZIP_OUTPUT_PATH}")
    print(f"  Started   : {export_started.isoformat()}")

    # Prepare output directory
    if os.path.exists(OUTPUT_DIRECTORY):
        print(f"\n[export] Removing existing output directory ...")
        shutil.rmtree(OUTPUT_DIRECTORY)
    _ensure_directory(OUTPUT_DIRECTORY)

    # Connect
    try:
        engine, session = _connect(DATABASE_URI)
    except Exception as error:
        print(f"\n[export] FATAL: Cannot connect to database: {error}")
        print("[export] Verify that PostgreSQL is running and DATABASE_URI is correct.")
        sys.exit(1)

    try:
        # Step 1: Full schema inspection
        _print_section("Step 1 of 7: Schema Inspection")
        schema = _inspect_full_schema(engine)
        print(f"[export] {len(schema)} tables found in schema.")

        # Step 2: Row counts
        _print_section("Step 2 of 7: Row Counting")
        row_counts = _count_all_rows(session, list(schema.keys()))
        total_rows = sum(c for c in row_counts.values() if c >= 0)
        print(f"[export] Total rows across all tables: {total_rows}")

        # Step 3: Table inventory
        _print_section("Step 3 of 7: Table Inventory")
        _write_table_inventory(OUTPUT_DIRECTORY, schema, row_counts)

        # Step 4: CSV export of all tables
        _print_section("Step 4 of 7: CSV Export (all tables)")
        exported_counts = _export_all_tables(session, schema, OUTPUT_DIRECTORY)

        # Step 5: Schema JSON
        _print_section("Step 5 of 7: Schema JSON")
        _export_schema_json(OUTPUT_DIRECTORY, schema, row_counts)

        # Step 6: PostgreSQL dump
        _print_section("Step 6 of 7: Full Database Dump (pg_dump)")
        dump_path      = os.path.join(OUTPUT_DIRECTORY, "full_database_dump.sql")
        dump_succeeded = _run_pg_dump(DATABASE_URI, dump_path)

        # Step 7: Export summary
        _print_section("Step 7 of 7: Export Summary")
        _write_export_summary(
            output_directory=OUTPUT_DIRECTORY,
            schema=schema,
            row_counts=row_counts,
            exported_counts=exported_counts,
            dump_succeeded=dump_succeeded,
            export_started=export_started,
        )

    finally:
        session.close()
        engine.dispose()

    # Create ZIP
    _print_section("Creating ZIP Archive")
    archive_size_mb = _create_zip_archive(OUTPUT_DIRECTORY, ZIP_OUTPUT_PATH)

    # Final report
    duration = (datetime.datetime.now() - export_started).total_seconds()
    _print_section("Export Complete")
    print(f"  Tables exported   : {len(schema)}")
    print(f"  Total rows        : {sum(c for c in row_counts.values() if c >= 0)}")
    print(f"  Archive size      : {archive_size_mb:.1f} MB")
    print(f"  Duration          : {duration:.1f}s")
    print(f"  Output directory  : {OUTPUT_DIRECTORY}")
    print(f"  ZIP archive       : {ZIP_OUTPUT_PATH}")
    print()
    print("  Copy to USB stick:")
    print(f"    cp {ZIP_OUTPUT_PATH} /media/<your-usb>/")
    print()
    print("  Restore on another machine:")
    print("    createdb target_database")
    print("    psql target_database < database_export/full_database_dump.sql")


if __name__ == "__main__":
    export_full_database()