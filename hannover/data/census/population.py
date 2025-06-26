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

    context.config("data_path")
    context.config("hannover.microbezirke_excel", "hannover/Age_gender_MBZ.xlsx")

def execute(context):
    # Load the census data for Hannover
    # "Mikrobezirk" =  "commune_id"
    df_census = pd.read_excel("{}/{}".format(
        context.config("data_path"), context.config("hannover.microbezirke_excel")
    ), sheet_name = "Altersgruppen MBZ Geschlecht", skiprows = 6, names = [
        "commune_id", "age_0_male", "age_6_male", "age_15_male","age_18_male", "age_24_male", "age_30_male", "age_45_male", "age_65_male", "age_80_male", 
        "age_0_female", "age_6_female", "age_15_female","age_18_female", "age_24_female", "age_30_female", "age_45_female", "age_65_female", "age_80_female", 
        "total",
    ])

    # Only keep rows where we have a value
    df_census = df_census[~df_census["total"].isna()].copy()
    # Only rows where 'mikrobezirk' contains only digits
    df_census = df_census[df_census["commune_id"].astype(str).str.isdigit()].copy()

    # Define age classes
    age_classes = [0, 6, 15, 18, 24, 30, 45, 65, 80]
    male_cols = [f"age_{age}_male" for age in age_classes]
    female_cols = [f"age_{age}_female" for age in age_classes]

    # Melt male and female separately
    df_male = df_census[["commune_id"] + male_cols].melt(
        id_vars="commune_id", var_name="age_class", value_name="weight")
    df_male["sex"] = "male"
    df_male["age_class"] = df_male["age_class"].str.extract(r"age_(\d+)_male").astype(int)

    df_female = df_census[["commune_id"] + female_cols].melt(
        id_vars="commune_id", var_name="age_class", value_name="weight")
    df_female["sex"] = "female"
    df_female["age_class"] = df_female["age_class"].str.extract(r"age_(\d+)_female").astype(int)

    # Combine and clean
    df_final = pd.concat([df_male, df_female], ignore_index=True)
    df_final["sex"] = df_final["sex"].astype("category")
    df_final["weight"] = df_final["weight"].astype(int)
    df_final["commune_id"] = "03241" + df_final["commune_id"].astype(str) # Commune IDs = 03241 + mikrobezirk
    
    # Filter out invalid commune_ids
    df_zones = context.stage("hannover.data.spatial.codes")
    valid_communes = df_zones["commune_id"].unique()
    df_final = df_final[df_final["commune_id"].isin(valid_communes)].copy()
 
    return df_final[["commune_id", "sex", "age_class", "weight"]]


def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("hannover.microbezirke_excel"))):
        raise RuntimeError("Hannover census data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("hannover.microbezirke_excel")))
