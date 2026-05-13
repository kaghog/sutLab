import numpy as np
import pandas as pd
import os

"""
This stage loads the raw data from the Seville building registry.
"""

def configure(context):
    context.config("data_path")
    context.config("seville.agglomeration_path", "aglomeracion_urban_de_sevilla.csv")

def execute(context):

    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("seville.agglomeration_path"))
    print(f"Loading list of municipalities belonging to the Seville agglomeration from {FILE_PATH}")
    df_municipalities = pd.read_csv(FILE_PATH, sep=";",dtype=str)


    return df_municipalities

def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("seville.agglomeration_path"))):
        raise RuntimeError("Seville agglomeration municipalities data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("seville.agglomeration_path")))
