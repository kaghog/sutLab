import pandas as pd
import os
import numpy as np

"""
This stage output the number of employed people in each Mikrobezirk (commune_id) in Hannover.
"""

def configure(context):
    context.config("data_path")
    
    context.stage("hannover.data.spatial.codes")
    context.stage("hannover.data.census.employment")
    context.stage("hannover.data.census.population")

def execute(context):

    # Load employment total ( will get single row for Hannover) 
    df_employment = context.stage("hannover.data.census.employment")
    total_employment = df_employment["weight"].sum()

    # Load population by Mikrobezirk (commune_id)
    df_population = context.stage("hannover.data.census.population")

    # Sum population by Mikrobezirk (commune_id) and calculate total population
    df_pop_mikro = df_population.groupby("commune_id", as_index=False)["weight"].sum()
    df_pop_mikro = df_pop_mikro.rename(columns={"weight": "population_weight"})
    total_population = df_pop_mikro["population_weight"].sum()

    # Compute proportional employment weight
    df_pop_mikro["weight"] = (df_pop_mikro["population_weight"] / total_population) * total_employment
    df_pop_mikro["weight"] = df_pop_mikro["weight"].round().astype(int)

    # Final output
    df_result = df_pop_mikro[["commune_id", "weight"]]

    return df_result
