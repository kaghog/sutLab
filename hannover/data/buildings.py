import geopandas as gpd
import zipfile
import pyogrio
import numpy as np
import pandas as pd
import glob, os

"""
This stage loads the raw data from the Hannover building registry.
"""

def configure(context):
    context.config("data_path")
    context.config("hannover.buildings_path", "hannover/buildings/buildings_Hannover_20km.shp")
    
    context.stage("hannover.data.spatial.iris")

def execute(context):
    df_zones = context.stage("hannover.data.spatial.iris")
    df_combined = []
    
    start_index = 0
    df_buildings = gpd.read_file(os.path.join(context.config("data_path"), context.config("hannover.buildings_path")))
        
    # Weighting by area
    df_buildings["weight"] = df_buildings.area

    # Attributes
    df_buildings["building_id"] = np.arange(len(df_buildings)) + start_index
    start_index += len(df_buildings) + 1

    df_buildings["geometry"] = df_buildings.centroid

    # Filter
    df_buildings = df_buildings[
        (df_buildings["weight"] >= 40) & (df_buildings["weight"] < 400)
    ].copy()
    

    # Impute spatial identifiers
    df_buildings = gpd.sjoin(df_buildings, df_zones[["geometry", "commune_id", "iris_id"]], 
        how = "left", predicate = "within").reset_index(drop = True).drop(columns = ["index_right"])

    df_buildings = df_buildings.dropna(subset=["commune_id", "iris_id"])
    
    df_combined.append(df_buildings[[
        "building_id", "weight", "commune_id", "iris_id", "geometry"
    ]])
    
    df_combined = gpd.GeoDataFrame(pd.concat(df_combined), crs = df_combined[0].crs)

    required_zones = set(df_zones["commune_id"].unique())
    available_zones = set(df_combined["commune_id"].unique())
    missing_zones = required_zones - available_zones

    if len(missing_zones) > 0:
        print("Adding {} centroids as buildings for missing municipalities".format(len(missing_zones)))

        df_missing = df_zones[df_zones["commune_id"].isin(missing_zones)][["commune_id", "iris_id", "geometry"]].copy()
        df_missing["geometry"] = df_missing["geometry"].centroid
        df_missing["building_id"] = np.arange(len(df_missing)) + start_index
        df_missing["weight"] = 1.0

        df_combined = pd.concat([df_combined, df_missing])
    
    return df_combined

def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("hannover.buildings_path"))):
        raise RuntimeError("Hannover buildings data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("hannover.buildings_path")))
