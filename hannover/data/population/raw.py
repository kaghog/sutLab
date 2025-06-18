import os
import geopandas as gpd
import zipfile
import numpy as np

"""
This stages loads a file containing population data for city Hannover including the Mikrobezirk codes
"""

def configure(context):
    context.config("data_path")
    context.config("hannover.population_shp", "hannover/SKH20_Mikrobezirke.shp")

def execute(context):
    # Load shapes
    gdf_mikrobezirke = gpd.read_file(os.path.join(context.config("data_path"), context.config("hannover.population_shp")))[["MIKROBZNR", "geometry"]]

    # Rename
    df_population = gdf_mikrobezirke.rename(columns = { 
        "MIKROBZNR": "mikrobezirk_code",
    })
    
    # Clean
    df_population = df_population[df_population["mikrobezirk_code"].astype(str).str.isdigit()].copy()
    
    # Sort by code
    df_population["mikrobezirk_code"] = df_population["mikrobezirk_code"].astype(int)
    df_population = df_population.sort_values("mikrobezirk_code").reset_index(drop=True)
    
    # Pad to 4-digit string
    df_population["mikrobezirk_code"] = df_population["mikrobezirk_code"].astype(str).str.zfill(4)
    
    # df_population[["mikrobezirk_code", "geometry"]]
    return df_population

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("hannover.population_shp"))):
        raise RuntimeError("German population data is not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("hannover.population_shp")))
