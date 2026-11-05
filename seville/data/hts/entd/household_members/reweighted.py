import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

"""
This stages calibrates HTS weights to help match attribute distribution of IPU population with census distribution.
"""

def configure(context):
    context.stage("seville.data.hts.entd.household_members.add_trips")
    context.stage("seville.data.census.population")
    context.stage("seville.data.census.employment")
    context.stage("seville.data.census.licenses")
    context.config("random_seed")
    context.config("weighted_trips")
    


def execute(context):
    df_households, df_persons, df_trips = context.stage("seville.data.hts.entd.household_members.add_trips")

    df_persons["person_weight"] = 1.0
    weighted_trips = context.config("weighted_trips")
    if not weighted_trips: 
        df_persons["trip_weight"] = 1.0
        df_trips["person_weight"] = 1.0
        df_trips["trip_weight"] = 1.0

    return df_households, df_persons, df_trips
