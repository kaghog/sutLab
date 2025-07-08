import pandas as pd
import numpy as np
import geopandas as gpd
import zipfile
import os

"""
This stage loads the raw census data for Bavaria.

TODO: This could be replaced with a Germany-wide extract from GENESIS
"""

def configure(context):
    context.stage("hannover.data.spatial.codes")
    context.stage("hannover.data.hts.entd.cleaned")

def execute(context):
    # Extract commune_id column as a list of strings
    df_codes = context.stage("hannover.data.spatial.codes")
    commune_ids = df_codes["commune_id"].astype(str).tolist()
    
    # Define age classes
    age_classes = [6, 15, 18, 24, 30, 45, 65, 80]
    
    _, df_persons, _ = context.stage("hannover.data.hts.entd.cleaned")

    # Assign age_class to each person in df_persons
    df_persons = df_persons.copy()
    df_persons["age_class"] = pd.cut(
        df_persons["age"],
        bins=age_classes + [np.inf],  # ensure ages beyond last bin are included
        labels=age_classes,
        right=False  # left inclusive, right exclusive
    )

    # Group by sex and age_class to get counts (weights)
    df_dist = df_persons.groupby(["sex", "age_class"]).size().reset_index(name="weight")

    # Ensure age_class is integer for consistency
    df_dist["age_class"] = df_dist["age_class"].astype(int)

    # Create a dataframe of commune_ids
    commune_df = pd.DataFrame({"commune_id": commune_ids})

    # Cross join commune_df with df_dist to replicate distribution
    df_dist["key"] = 1
    commune_df["key"] = 1

    df_result = pd.merge(commune_df, df_dist, on="key").drop(columns="key")
 
    return df_result[["commune_id", "sex", "age_class", "weight"]]

