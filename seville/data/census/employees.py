import pandas as pd
import os
import numpy as np

"""
This stage output the number of employed people in each Mikrobezirk (commune_id) in seville.
"""

def configure(context):
    context.config("data_path")
    
    context.stage("seville.data.census.employment")

def execute(context):

    # Load employment total ( will get single row for seville) 
    df_employment = context.stage("seville.data.census.employment")

    df_employment = df_employment.groupby("census_section", as_index=False)["weight"].sum()
    df_employment = df_employment.rename(
        columns={
            "weight": "weight", 
            "census_section": "commune_id"
            })
    
    df_result = df_employment[["commune_id", "weight"]]

    assert not df_result.isnull().values.any(), "df_employees contains NaNs!"

    return df_result
