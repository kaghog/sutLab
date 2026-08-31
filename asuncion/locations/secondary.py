import numpy as np
import pandas as pd

"""
Yield education location candidates.
"""

def configure(context):
    context.stage("asuncion.data.work")

SECONDARY_BUILDING_TYPES = ['COMERCIO', 'SERVICIO']


def execute(context):
    # Load data
    df = context.stage("asuncion.data.work")

    df = df[df['type_category'].isin(SECONDARY_BUILDING_TYPES)].copy()
    df = df.rename(columns = { "building_id": "location_id" })

    df['offers_shop'] = df['type_category'] == 'COMERCIO'
    df['offers_leisure'] = df['type_category'] == 'SERVICIO'
    df['offers_other'] = True

    # Identifiers
    df["location_id"] = np.arange(len(df))
    df["location_id"] = "sec_" + df["location_id"].astype(str)

    df["weight"] = df["employees"]



    return df[[
        "location_id", "weight", "commune_id", "iris_id", "geometry", "offers_shop", "offers_leisure", "offers_other"
    ]]

