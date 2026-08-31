
import pandas as pd
import os
import numpy as np

def configure(context):
    context.stage("asuncion.gravity.od_zones")
    context.stage("asuncion.data.hts.entd.filtered")

def get_purpose_location(df_trips, level):
    preceding_trips = df_trips[(df_trips['preceding_purpose'] == 'home') & (df_trips['following_purpose'] == 'work')]
    preceding_trips = preceding_trips.rename(
        columns={f"origin_{level}":"origin_id", f"destination_{level}": "destination_id",})
    preceding_trips = preceding_trips[['person_id', 'origin_id', 'destination_id', 'trip_weight']]

    following_trips = df_trips[(df_trips['preceding_purpose'] == 'work') & (df_trips['following_purpose'] == 'home')]
    following_trips = following_trips.rename(
        columns={f"origin_{level}":"origin_id", f"destination_{level}": "destination_id"})
    following_trips = following_trips[['person_id', 'origin_id', 'destination_id', 'trip_weight']]

    locations = pd.concat([preceding_trips, following_trips])
    locations = locations[['person_id', 'origin_id', 'destination_id', 'trip_weight']]

    return locations


def execute(context):
    # Load data
    df_trips = context.stage("asuncion.data.hts.entd.filtered")[2]
 

    work_commute_trips = get_purpose_location(df_trips, 'district_id')


    assert len(work_commute_trips) != 0

    work_commute_trips['weight'] = work_commute_trips['trip_weight']
    # Aggregate commuter counts
    od_matrix = (
        work_commute_trips
        .groupby(['origin_id', 'destination_id'])
        ['weight'].sum()
        #.size()
        .reset_index(name='weight')
    )

    od_matrix['trips'] = od_matrix['weight']



    return od_matrix
