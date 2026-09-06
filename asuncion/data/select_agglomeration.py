import numpy as np
import pandas as pd
import os

"""
This stage loads the raw data from the Asuncion metropolitan area registry.
"""

def configure(context):
    context.config("data_path")
    context.config("asuncion.agglomeration_path", "asuncion_metropolitan_area.csv")

def execute(context):

    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("asuncion.agglomeration_path"))
    print(f"Loading list of districts belonging to the Asuncion agglomeration from {FILE_PATH}")
    df_districts = pd.read_csv(FILE_PATH, sep=";",dtype=str)



    return df_districts

def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("asuncion.agglomeration_path"))):
        raise RuntimeError("Asuncion agglomeration districts data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("asuncion.agglomeration_path")))
