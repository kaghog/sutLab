import os
import geopandas as gpd
import zipfile
import numpy as np

"""
This stages loads a file containing population data for Germany including the adminstrative codes and filter population data for Hannover
Hannover's official AGS (ARS) starts with: 03241
"""

def configure(context):
    context.config("data_path")
    context.config("germany.population_path", "germany/vg250-ew_12-31.utm32s.gpkg.ebenen.zip")
    context.config("germany.population_source", "vg250-ew_12-31.utm32s.gpkg.ebenen/vg250-ew_ebenen_1231/DE_VG250.gpkg")

def execute(context):
    # Load IRIS registry
    with zipfile.ZipFile(
        "{}/{}".format(context.config("data_path"), context.config("germany.population_path"))) as archive:
        with archive.open(context.config("germany.population_source")) as f:
            df_population = gpd.read_file(f, layer = "vg250_gem")[[
                "ARS", "EWZ", "geometry"
            ]]

    # Filter for prefix starting with 03241 (Hannover)
    df_population = df_population[df_population["ARS"].str.startswith("03241")].copy()
    
    # Rename
    df_population = df_population.rename(columns = { 
        "ARS": "municipality_code",
        "EWZ": "population"
    })
    
    return df_population

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("germany.population_path"))):
        raise RuntimeError("German population data is not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("germany.population_path")))
