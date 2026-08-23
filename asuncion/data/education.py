import numpy as np
import pandas as pd
import geopandas as gpd

"""
Yield home zones for Spain based on synthetic population data.
"""

def configure(context):
    context.config("asuncion.edu_locations_data", "locations/locales_educativos.geojson")
    context.config("data_path")

    context.stage("asuncion.data.spatial.iris")

EDUCATION_MAP = {
    "EDUCACION BASICA": "school",
    "COLEGIOS": "school",
    "UNIVERSIDADES": "university",
    "OTROS":"other",
}

def execute(context):
    # Load data
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.edu_locations_data')}"
    edu_locations = gpd.read_file(FILE_URL, dtype=str)

    edu_locations = edu_locations.to_crs("EPSG:4674")
    edu_locations["education_type"] = edu_locations["desc_class"].map(EDUCATION_MAP)

    edu_locations = edu_locations[["geometry", "education_type"]].copy()


    edu_locations["students"] = 1.0
    edu_locations["fake"] = False
    edu_locations["location_id"] = np.arange(len(edu_locations))
    edu_locations["location_id"] = "edu_" + edu_locations["location_id"].astype(str)
    edu_locations = edu_locations.to_crs(context.config("pipeline_crs"))
    edu_locations["geometry"] = edu_locations.geometry.centroid

    # add spatial identifiers
    gdf_spatial = context.stage("asuncion.data.spatial.iris")
    edu_locations = edu_locations.to_crs(gdf_spatial.crs)
    edu_locations = gpd.sjoin(edu_locations, gdf_spatial, predicate="within", how="inner")


    edu_locations["weight"] = edu_locations["students"]
    assert len(edu_locations) != 0

    return edu_locations[["location_id", "education_type", "weight", "fake", "commune_id", "iris_id", "geometry"]]
