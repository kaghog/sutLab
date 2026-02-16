from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the national HTS.
"""


def configure(context):
    context.config("data_path")

def execute(context):
    df_persons = pd.read_csv(
        "%s/person_hts.csv" % context.config("data_path"), 
        encoding = "latin1"
    )

    df_households = pd.read_csv(
        "%s/household_hts.csv" % context.config("data_path"), 
        encoding = "latin1"
    )

    df_trips = pd.read_csv(
        "%s/trip_hts.csv" % context.config("data_path"), 
        encoding = "latin1"
    )
    
    return df_households, df_persons, df_trips