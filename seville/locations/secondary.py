import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

"""
Yield home location candidates for Germany.
"""

def configure(context):
    context.stage("seville.data.buildings")

SECONDARY_BUILDING_TYPES = ['4_2_retail', '4_3_publicServices', '4_1_office']


def execute(context):
    # Load data
    df = context.stage("seville.data.buildings")
    df = df[df['type'].isin(SECONDARY_BUILDING_TYPES)].copy()
    df = df.rename(columns = { "building_id": "location_id" })

    df['offers_shop'] = df['type'] == '4_2_retail'
    df['offers_leisure'] = df['type'] == '4_3_publicServices'
    df['offers_other'] = True

    # Identifiers
    df["location_id"] = np.arange(len(df))
    df["location_id"] = "sec_" + df["location_id"].astype(str)



    return df[[
        "location_id", "weight", "commune_id", "iris_id", "geometry", "offers_shop", "offers_leisure", "offers_other"
    ]]

