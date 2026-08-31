import numpy as np
import pandas as pd
import geopandas as gpd

"""
Yield home zones for Spain based on synthetic population data.
"""

def configure(context):
    context.config("asuncion.work_locations_data", "locations/unidades_economicas.geojson")
    context.config("data_path")

    context.stage("asuncion.data.spatial.iris")
    context.config("pipeline_crs")


def execute(context):
    # Load data
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.work_locations_data')}"
    work_locations = gpd.read_file(FILE_URL, dtype=str)
    work_locations["type_category"] = work_locations["desccodref"]
    work_locations["type"] = work_locations["descref"]

    work_locations = work_locations[["geometry", "type", "type_category"]].copy()

    work_locations["employees"] = 1.0
    work_locations["fake"] = False
    work_locations["location_id"] = np.arange(len(work_locations))
    work_locations["location_id"] = "work_" + work_locations["location_id"].astype(str)
    
    work_locations = work_locations.to_crs(context.config("pipeline_crs"))
    work_locations["geometry"] = work_locations.geometry.centroid

    # add spatial identifiers
    gdf_spatial = context.stage("asuncion.data.spatial.iris")
    work_locations = work_locations.to_crs(gdf_spatial.crs)

    work_locations = gpd.sjoin(work_locations, gdf_spatial, predicate="within", how="inner")

    assert len(work_locations) != 0

    return work_locations[["location_id", "employees", "fake", "commune_id", "iris_id", "geometry", "type", "type_category"]]
