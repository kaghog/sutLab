import os
import geopandas as gpd
import zipfile
import numpy as np
import pandas as pd

"""
This stages loads a spatial data of administrative units used in the pipeline.

NOTE: This is stage is alternatively called "population.raw" in other pipelines (e.g., Hannover). 
For asuncion, the stage is renamed to better reflect its nature since it is no longer related to population data.
"""

def configure(context):
    context.config("data_path")
    context.config("asuncion.asuncion_shp","spatial/00 ASUNCION/Barrios Localidades_Asuncion.shp")
    context.config("asuncion.central_shp", "spatial/11 CENTRAL/Distritos_Central.shp")


# EPSG:4674

def execute(context):
    # Load Asuncion departement
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.asuncion_shp')}"
    gdf_asuncion = gpd.read_file(FILE_URL, dtype=str)
    ASUNCION_COLUMNS = {
        "DPTO_DESC": "departement",
        "DIST_DESC_": "district",
        "BARLO_DESC": "borough"
    }
    gdf_asuncion = gdf_asuncion[list(ASUNCION_COLUMNS.keys()) + ["geometry"]]
    gdf_asuncion = gdf_asuncion.rename(columns=ASUNCION_COLUMNS)


    # Load Central departement
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.central_shp')}"
    gdf_central = gpd.read_file(FILE_URL, dtype=str)
    CENTRAL_COLUMNS = {
        "DPTO_DESC": "departement",
        "DIST_DESC_": "district",
    }
    gdf_central = gdf_central[list(CENTRAL_COLUMNS.keys()) + ["geometry"]]
    gdf_central = gdf_central.rename(columns=CENTRAL_COLUMNS)

    gdf_central["borough"] = gdf_central["district"]


    # Merge both dataframes
    assert gdf_asuncion.crs == gdf_central.crs

    gdf_spatial = pd.concat([gdf_asuncion, gdf_central])

    gdf_spatial = gdf_spatial.to_crs("EPSG:4674")

    # make all upper_case
    gdf_spatial["departement"] = gdf_spatial["departement"].str.upper()
    gdf_spatial["district"] = gdf_spatial["district"].str.upper()
    gdf_spatial["borough"] = gdf_spatial["borough"].str.upper()

    # Return
    return gdf_spatial[["departement", "district", "borough", "geometry"]]


def validate(context):
    filenames = [
        "asuncion.asuncion_shp",
        "asuncion.central_shp",
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
