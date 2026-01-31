from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the national HTS.
"""


def configure(context):
    context.stage("seville.data.hts.entd.add_trips")

def execute(context):
    df_households, df_persons, df_trips = context.stage("seville.data.hts.entd.add_trips")
    
    # Finish up
    df_households = df_households[hts.HOUSEHOLD_COLUMNS + ["urban_type", "income_class"]]
    df_persons = df_persons[hts.PERSON_COLUMNS]
    df_trips = df_trips[hts.TRIP_COLUMNS + ["euclidean_distance"]]

    hts.check(df_households, df_persons, df_trips)
    return df_households, df_persons, df_trips
