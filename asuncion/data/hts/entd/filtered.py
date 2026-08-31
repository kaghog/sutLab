from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the national HTS.
"""


def configure(context):
    context.stage("asuncion.data.hts.entd.cleaned")

def execute(context):
    df_households, df_persons, df_trips = context.stage("asuncion.data.hts.entd.cleaned")
    
    # Finish up
    df_households = df_households[hts.HOUSEHOLD_COLUMNS + ["urban_type", "income_class", "household_category"]]
    df_persons = df_persons[hts.PERSON_COLUMNS]
    df_trips = df_trips[hts.TRIP_COLUMNS + ["euclidean_distance", "routed_distance", "origin_district_id", "destination_district_id"]]

    df_trips = df_trips.sort_values(by = ["person_id", "trip_id"])

    hts.check(df_households, df_persons, df_trips)
    return df_households, df_persons, df_trips
