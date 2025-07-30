import pandas as pd
import numpy as np
import geopandas as gpd


"""
This stage loads the raw census data for Hannover provided at Bezirk level.

"""

def configure(context):
    context.stage("hannover.data.spatial.codes")
    context.config("data_path")
    context.stage("hannover.data.census.raw")
    context.config("check_spatial_codes", False)

def execute(context):
    
    df = context.stage("hannover.data.census.raw")

    #define commune_id - "03241" is used to identify Hannover region
    df["commune_id"] = "03241" + df["mikrobezirk_code"].astype(str)
    df["commune_id"] = df["commune_id"].astype("category")

    # Check if all the commune_ids are in the spatial.codes
    df_spatial_codes = context.stage("hannover.data.spatial.codes")[["commune_id"]]
    missing_communes = set(df["commune_id"]) - set(df_spatial_codes["commune_id"])
    if missing_communes and not context.config("check_spatial_codes"):
        raise ValueError(f"Missing commune_ids in spatial codes: {missing_communes}")
    else:
        #remove mising communes from df
        print(f"Removing {len(missing_communes)} communes not in spatial codes from the provided administrative units.")
        df = df[df["commune_id"].isin(df_spatial_codes["commune_id"])]

    # Melt to long format for both sexes
    df_long = pd.wide_to_long(
        df,
        stubnames=["male", "female"],
        i="commune_id",
        j="age_class",
        sep="_",
        suffix='\\d+'
    )
    df_long = df_long.reset_index()

    

    df_male = df_long[["commune_id", "age_class", "male"]].rename(columns={"male": "weight"})
    df_male["sex"] = "male"
    df_female = df_long[["commune_id", "age_class", "female"]].rename(columns={"female": "weight"})
    df_female["sex"] = "female"

    df_result = pd.concat([df_male, df_female], ignore_index=True)
    df_result = df_result[df_result["weight"].notna()]

    #cleaning
    df_result["weight"] = df_result["weight"].astype(str).str.replace(",", "")
    df_result["weight"] = df_result["weight"].replace("-", 0)
    
    df_result["weight"] = pd.to_numeric(df_result["weight"], errors="coerce").fillna(-99).astype(int)
    if (len(df_result[df_result["weight"] == -99]) > 0):
        raise ValueError("There are still -99 values in the weight column, indicating conversion issues.")
    
    df_result["sex"] = df_result["sex"].astype("category")
    df_result["age_class"] = df_result["age_class"].astype(int)
    df_result["commune_id"] = df_result["commune_id"].astype("category")
    
    return df_result[["commune_id", "sex", "age_class", "weight"]]

