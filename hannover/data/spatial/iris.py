"""
Generates the IRIS zoning system that is not used in Germany. Instead, we create one
fake IRIS for each municipality in Germany. See the `codes` stage for more information.
"""

import geopandas as gpd


def configure(context):
    context.stage("hannover.data.spatial.admin_boundary")
    context.stage("hannover.data.spatial.codes")


def execute(context):
    # Load administrative codes
    df = context.stage("hannover.data.spatial.codes").copy()

    # Generate fake IRIS ID (13 chars: kreis + commune + suffix)
    # This is the main purpose of this stage
    df["iris_id"] = (
        df["kreis_code"].astype(str) + df["commune_id"].astype(str) + "0000"
    ).astype("category")

    # Load geometry from admin_boundary (GeoDataFrame)
    df_geometry = context.stage("hannover.data.spatial.admin_boundary")[
        ["mikrobezirk_code", "geometry"]
    ]

    # Rename mikrobezirk_code to commune_id for merge
    df_geometry = df_geometry.rename(columns={"mikrobezirk_code": "commune_id"})

    # Merge and ensure result is a GeoDataFrame
    df = gpd.GeoDataFrame(
        df.merge(df_geometry, on="commune_id", how="left"),
        geometry="geometry",
        crs=df_geometry.crs,
    )

    return df[["iris_id", "commune_id", "departement_id", "kreis_code", "geometry"]]
