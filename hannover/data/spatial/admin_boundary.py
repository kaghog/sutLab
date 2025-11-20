import os

import geopandas as gpd

"""
This stages loads a file containing population data for city Hannover including the Mikrobezirk codes
"""

def configure(context):
    context.config("data_path")
    context.config("hannover.population_shp")

def execute(context):
    # Load shapes
    gdf_mikrobezirke = gpd.read_file("{}/{}".format(context.config("data_path"), context.config("hannover.population_shp")))

    # Rename
    # print(gdf_mikrobezirke.columns)
    # print(gdf_mikrobezirke.head())
    gdf_mikrobezirke = gdf_mikrobezirke[["MIKROBZ_BA", "geometry"]]
    df_population = gdf_mikrobezirke.rename(columns = { 
        "MIKROBZ_BA": "mikrobezirk_code",
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
