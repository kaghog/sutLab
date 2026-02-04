import numpy as np
import pandas as pd
from shapely.geometry import Point

"""
Yield work location candidates for Germany.
"""


def configure(context):
    context.stage("hannover.data.osm.locations")
    context.stage("data.spatial.municipalities")

    context.config("synthetic_work_min_radius_km", 10.0)
    context.config("synthetic_work_max_radius_km", 30.0)
    context.config("synthetic_work_locations_per_km", 3)


def execute(context):
    # Load data
    df = context.stage("hannover.data.osm.locations")
    df = df[df["location_type"] == "work"].copy()

    df["employees"] = df["area"] * df["floors"]
    df["fake"] = False

    center_x = df.geometry.x.mean()
    center_y = df.geometry.y.mean()

    # Fill missing municipalities
    df_municipalities = context.stage("data.spatial.municipalities")
    df_fake = df_municipalities[
        ~df_municipalities["commune_id"].isin(df["commune_id"])
    ].copy()

    df_fake["geometry"] = df_fake["geometry"].centroid
    df_fake["iris_id"] = df_fake["commune_id"].astype(str) + "0000"
    df_fake["iris_id"] = df_fake["iris_id"].astype("category")
    df_fake["employees"] = 1
    df_fake["fake"] = True

    # ========== SYNTHETIC OUTER RING LOCATIONS ==========
    # Adds work locations beyond region boundaries (10-30km) to match HTS trip distances.
    # Places locations on spatial grid, ignoring municipality boundaries.
    # Remove this section if reverting to OSM-only coverage.

    min_radius_km = context.config("synthetic_work_min_radius_km")
    max_radius_km = context.config("synthetic_work_max_radius_km")
    locations_per_km = context.config("synthetic_work_locations_per_km")

    median_employees = df["employees"].median()

    synthetic_locations = []
    for radius_km in range(int(min_radius_km), int(max_radius_km) + 1, 1):
        radius_m = radius_km * 1000
        circumference = 2 * np.pi * radius_m
        n_locations = int((circumference / 1000) * locations_per_km)

        angles = np.linspace(0, 2 * np.pi, n_locations, endpoint=False)

        for angle in angles:
            x = center_x + radius_m * np.cos(angle)
            y = center_y + radius_m * np.sin(angle)

            # Gentler decay to maintain substantial employee counts at distance
            decay = 1.0 - 0.5 * (
                (radius_km - min_radius_km) / (max_radius_km - min_radius_km)
            )
            employees = int(median_employees * decay)

            synthetic_locations.append(
                {
                    "geometry": Point(x, y),
                    "employees": max(employees, 100),
                    "fake": True,
                    "commune_id": "99999",
                    "iris_id": "999990000",
                }
            )

    df_synthetic = pd.DataFrame(synthetic_locations)
    print(
        f"Added {len(df_synthetic)} synthetic work locations in rings {min_radius_km}-{max_radius_km}km"
    )

    # ========== END SYNTHETIC LOCATIONS ==========

    # Merge
    df = pd.concat(
        [
            df[["employees", "fake", "commune_id", "iris_id", "geometry"]],
            df_fake[["employees", "fake", "commune_id", "iris_id", "geometry"]],
            df_synthetic[["employees", "fake", "commune_id", "iris_id", "geometry"]],
        ],
        ignore_index=True,
    )

    # Identifiers
    df["location_id"] = np.arange(len(df))
    df["location_id"] = "work_" + df["location_id"].astype(str)

    df["iris_id"] = df["iris_id"].astype("category")

    return df
