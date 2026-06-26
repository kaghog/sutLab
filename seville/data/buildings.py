import geopandas as gpd
import zipfile
import pyogrio
import numpy as np
import pandas as pd
import glob, os

"""
This stage loads the raw data from the Seville building registry.
"""

def configure(context):
    context.config("data_path")
    context.config("seville.buildings_path", "catastre_buildings/all_buildings.gpkg")
    
    context.stage("seville.data.spatial.iris")

def execute(context):
    df_zones = context.stage("seville.data.spatial.iris")
    df_combined = []
    
    start_index = 0
    df_buildings = gpd.read_file("{}/{}".format(context.config("data_path"), context.config("seville.buildings_path")))
    df_zones = df_zones.to_crs(crs="EPSG:25830")
    df_buildings = df_buildings.to_crs(crs="EPSG:25830")
    df_buildings = df_buildings.rename(columns={"currentUse": "type"})
    # value is grossFloorArea in m^2
    df_buildings = df_buildings.rename(columns={"value": "weight"})
    df_buildings["weight"] = df_buildings["weight"].astype("float64")

    # Attributes
    df_buildings["building_id"] = np.arange(len(df_buildings)) + start_index
    start_index += len(df_buildings) + 1

    df_buildings["geometry"] = df_buildings.centroid    



    # Impute spatial identifiers
    df_buildings = gpd.sjoin(df_buildings, df_zones[["geometry", "commune_id", "iris_id"]], 
        how = "left", predicate = "within").reset_index(drop = True).drop(columns = ["index_right"])

    df_buildings = df_buildings.dropna(subset=["commune_id", "iris_id"])
    
    df_combined.append(df_buildings[[
        "building_id", "weight", "commune_id", "iris_id", "geometry", "type"
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
        df_missing["type"] = "1_residential"

        df_combined = pd.concat([df_combined, df_missing])

    return df_combined[["building_id", "weight", "commune_id", "iris_id", "geometry", "type"]]

def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("seville.buildings_path"))):
        raise RuntimeError("Seville buildings data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("seville.buildings_path")))
