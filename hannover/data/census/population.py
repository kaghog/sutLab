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

def execute(context):
    
    df = context.stage("hannover.data.census.raw")
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

    #define commune_id - "03241" is used to identify Hannover region
    df_long["commune_id"] = "03241" + df_long["mikrobezirk_code"].astype(str)
    df_long["commune_id"] = df_long["commune_id"].astype("category")

    df_male = df_long[["commune_id", "age_class", "male"]].rename(columns={"male": "weight"})
    df_male["sex"] = "male"
    df_female = df_long[["commune_id", "age_class", "female"]].rename(columns={"female": "weight"})
    df_female["sex"] = "female"

    df_result = pd.concat([df_male, df_female], ignore_index=True)
    df_result = df_result[df_result["weight"].notna()]

    #cleaning
    df_result["weight"] = df_result["weight"].astype(str).str.replace(",", "")
    df_result["weight"] = df_result["weight"].replace("-", 0)
    df_result["weight"] = df_result["weight"].astype(int)
    df_result["sex"] = df_result["sex"].astype("category")
    df_result["age_class"] = df_result["age_class"].astype(int)

    print(df_result[["commune_id", "sex", "age_class", "weight"]])
    
    return df_result[["commune_id", "sex", "age_class", "weight"]]

