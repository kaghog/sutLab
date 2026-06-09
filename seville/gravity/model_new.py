
import pandas as pd
import os
import numpy as np

"""
Apply gravity model to generate a distance matrix covering Seville Metropolitan Area.
"""

DEFAULT_K = -0.2
DEFAULT_A = -2.4
DEFAULT_B = 1.0
DEFAULT_G = 1.0

def configure(context):
    context.stage("seville.gravity.distance_matrix_new")
    context.stage("seville.gravity.od_zones")

    context.config('analysis_path')


def execute(context):
    # Load data
    df_distances = context.stage("seville.gravity.distance_matrix_new")

    od_zones, df_population, df_employees = context.stage("seville.gravity.od_zones")

    # Manage identifiers
    df_population = df_population.rename(columns = {
        "id": "origin_id",
    })[["origin_id", "population"]]

    df_employees = df_employees.rename(columns = {
        "id": "destination_id",
    })[["destination_id", "employees"]]
    
    # Find the set of used zones (also taking into account zero flows)
    zones = set(df_population["origin_id"])
    zones |= set(df_employees["destination_id"])
    zones |= set(df_distances["origin_id"])
    zones |= set(df_distances["destination_id"])
    zones = sorted(list(zones))

    df_matrix = pd.DataFrame([(o, d) for o in zones for d in zones], columns=['origin_id', 'destination_id'])
    df_matrix = df_matrix.merge(df_population[['origin_id', 'population']], on='origin_id')
    df_matrix = df_matrix.merge(df_employees[['destination_id', 'employees']], on='destination_id')
    df_matrix = df_matrix.merge(df_distances, on=['origin_id', 'destination_id'])
    df_matrix['distance_km'] = df_matrix['distance_km'].replace(0, 0.1)
    
    k = DEFAULT_K
    a = DEFAULT_A
    b = DEFAULT_B
    g = DEFAULT_G

    df_matrix['weight'] = k * (df_matrix['population']**a * df_matrix['employees']**b * df_matrix['distance_km']**g)
    #df_matrix['predicted_flow'] = df_matrix['predicted_flow'].round(0).astype(int)

    # Calculate totals
    df_total = df_matrix[["origin_id", "weight"]].groupby("origin_id").sum().reset_index().rename({ "weight" : "total" }, axis = 1)
    df_matrix = pd.merge(df_matrix, df_total, on = "origin_id")

    # Fix missing flows
    f_missing_total = df_matrix["total"] == 0.0
    df_matrix.loc[f_missing_total & (df_matrix["origin_id"] == df_matrix["destination_id"]), "weight"] = 1.0
    df_matrix.loc[f_missing_total, "total"] = 1.0

    # Convert to probability
    df_matrix["weight"] = df_matrix["weight"] / df_matrix["total"]
    df_matrix = df_matrix[["origin_id", "destination_id", "weight"]]

    # One representing work, one representing education
    return df_matrix, df_matrix
