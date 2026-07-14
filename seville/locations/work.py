import numpy as np
import pandas as pd

"""
Yield work location candidates for Germany.
"""

def configure(context):
    context.stage("seville.data.buildings")
    context.stage("data.spatial.municipalities")

def execute(context):
    # Load data
    df = context.stage("seville.data.buildings").copy()
    non_residential = (df["type"] != "1_residential")
    residential_with_extra_space = (df["type"] == "1_residential") & (df["units"] > df["dwellings"])
    df = df[non_residential | residential_with_extra_space]

    # Decrease weight for work locations that are partially resident buildings (residential_with_extra_space)
    df['work_units'] = (df["units"] - df["dwellings"])
    df['work_ratio'] = df['work_units'] / df['units']
    df['weight'] = df['weight'] * df['work_ratio']


    df["employees"] = df["weight"] # weight = area
    df["fake"] = False

    # Fill missing municipalities
    df_fake = context.stage("data.spatial.municipalities")
    df_fake = df_fake[~df_fake["commune_id"].isin(df["commune_id"])].copy()

    df_fake["geometry"] = df_fake["geometry"].centroid

    df_fake["iris_id"] = df_fake["commune_id"].astype(str) + "0000"
    df_fake["iris_id"] = df_fake["iris_id"].astype("category")

    df_fake["employees"] = 1
    df_fake["fake"] = True
    
    # Merge
    df = pd.concat([
        df[["employees", "fake", "commune_id", "iris_id", "geometry"]], 
        df_fake[["employees", "fake", "commune_id", "iris_id", "geometry"]]
    ])

    # Identifiers
    df["location_id"] = np.arange(len(df))
    df["location_id"] = "work_" + df["location_id"].astype(str)

    return df