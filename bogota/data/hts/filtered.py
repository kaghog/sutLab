import data.hts.hts as hts
import numpy as np
import os

"""
This stage filters out ENTD observations which live or work outside
"""

def configure(context):
    context.stage("data.hts.entd.cleaned")

def execute(context):

    df_households, df_persons, df_trips = context.stage("data.hts.entd.cleaned")
    
    return df_households, df_persons, df_trips
