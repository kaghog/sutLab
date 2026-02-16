from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the national HTS.
"""


def configure(context):
    context.stage("bogota.data.hts.raw")

def execute(context):

    df_households, df_persons, df_trips = context.stage("bogota.data.hts.raw")
       #External cleaning has been done 

    if "origin_departement_id" not in df_trips:
        df_trips["origin_departement_id"] = "11001"
        df_trips["origin_departement_id"] = df_trips["origin_departement_id"].astype("category")
    
    if "destination_departement_id" not in df_trips:
        df_trips["destination_departement_id"] = "11001"
        df_trips["destination_departement_id"] = df_trips["destination_departement_id"].astype("category")

    if "departement_id" not in df_persons:
        df_persons["departement_id"] = "11001"
        df_persons["departement_id"] = df_persons["departement_id"].astype("category")
    
    return df_households, df_persons, df_trips

INCOME_CLASS_BOUNDS_SEVILLE = [1000, 1500, 2000, 3000, 4000, 5000, 1e6]
def calculate_income_class(df):
    assert "household_income" in df
    assert "consumption_units" in df

    return np.digitize(df["household_income"] / df["consumption_units"], INCOME_CLASS_BOUNDS_SEVILLE, right = True)

    
