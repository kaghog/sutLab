import numpy as np
import pandas as pd
from shapely.geometry import Point

"""
Yield education location candidates for Germany.
"""


def configure(context):
    context.stage("hannover.data.osm.locations")
    context.stage("data.spatial.municipalities")

    context.config("synthetic_education_min_radius_km", 10.0)
    context.config("synthetic_education_max_radius_km", 30.0)
    context.config("synthetic_education_locations_per_km", 1)


MINIMUM_AREA = 20


def execute(context):
    # Load data
    df = context.stage("hannover.data.osm.locations")
    df = df[df["location_type"] == "education"].copy()
    df["fake"] = False

    # Weight
    df["weight"] = np.maximum(df["area"], MINIMUM_AREA) * df["floors"]

    center_x = df.geometry.x.mean()
    center_y = df.geometry.y.mean()

    # Handle types
    for education_type in ["kindergarten", "school", "university"]:
        f = df["building"] == education_type
        df.loc[f, "education_type"] = education_type

    for education_type in ["kindergarten", "school", "university"]:
        f = df["amenity"] == education_type
        df.loc[f, "education_type"] = education_type

    df = df[~df["education_type"].isna()].copy()

    # Need this for the IDF logic, not for the Germany logic
    df_municipalities = context.stage("data.spatial.municipalities")
    df_fake = df_municipalities[
        ~df_municipalities["commune_id"].isin(df["commune_id"])
    ].copy()

    df_fake["geometry"] = df_fake["geometry"].centroid

    df_fake["iris_id"] = df_fake["commune_id"].astype(str) + "0000"
    df_fake["iris_id"] = df_fake["iris_id"].astype("category")

    df_fake["fake"] = True

    df_fake["education_type"] = "unknown"
    df_fake["weight"] = 1.0

    # ========== SYNTHETIC OUTER RING LOCATIONS ==========
    # Adds education locations beyond region boundaries (10-30km) to match HTS trip distances.
    # Places locations on spatial grid, ignoring municipality boundaries.
    # Remove this section if reverting to OSM-only coverage.

    min_radius_km = context.config("synthetic_education_min_radius_km")
    max_radius_km = context.config("synthetic_education_max_radius_km")
    locations_per_km = context.config("synthetic_education_locations_per_km")

    median_weight = df["weight"].median()

    synthetic_locations = []
    for radius_km in range(int(min_radius_km), int(max_radius_km) + 1, 1):
        radius_m = radius_km * 1000
        circumference = 2 * np.pi * radius_m
        n_locations = int((circumference / 1000) * locations_per_km)

        angles = np.linspace(0, 2 * np.pi, n_locations, endpoint=False)

        for angle in angles:
            x = center_x + radius_m * np.cos(angle)
            y = center_y + radius_m * np.sin(angle)

            # Gentler decay for education (schools can be far from center)
            decay = 1.0 - 0.4 * (
                (radius_km - min_radius_km) / (max_radius_km - min_radius_km)
            )
            weight = int(median_weight * decay)

            # Add one of each type to cover all age groups
            for edu_type in ["school", "kindergarten", "university"]:
                synthetic_locations.append(
                    {
                        "geometry": Point(x, y),
                        "weight": max(weight, 100),
                        "fake": True,
                        "commune_id": "99999",
                        "iris_id": "999990000",
                        "education_type": edu_type,
                    }
                )

    df_synthetic = pd.DataFrame(synthetic_locations)
    print(
        f"Added {len(df_synthetic)} synthetic education locations in rings {min_radius_km}-{max_radius_km}km"
    )

    # ========== END SYNTHETIC LOCATIONS ==========

    # Merge
    df = pd.concat(
        [
            df[
                [
                    "fake",
                    "commune_id",
                    "iris_id",
                    "education_type",
                    "weight",
                    "geometry",
                ]
            ],
            df_fake[
                [
                    "fake",
                    "commune_id",
                    "iris_id",
                    "education_type",
                    "weight",
                    "geometry",
                ]
            ],
            df_synthetic[
                [
                    "fake",
                    "commune_id",
                    "iris_id",
                    "education_type",
                    "weight",
                    "geometry",
                ]
            ],
        ],
        ignore_index=True,
    )

    # Convert to category
    df["education_type"] = df["education_type"].astype("category")

    # Identifiers
    df["location_id"] = np.arange(len(df))
    df["location_id"] = "edu_" + df["location_id"].astype(str)

    # TODO compare with actual school locations data given. Cehck also for buildings file

    return df
