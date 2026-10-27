import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

"""
This stage provides a zero household income for all households as it is needed in 
downstream stages if income is missing.
"""

def configure(context):
    context.stage("synthesis.population.sampled")

def execute(context):
    # Load data
    df = context.stage("synthesis.population.sampled").copy()
    df = df.drop_duplicates("household_id")

    # Format

    if "household_income" not in df.columns:
        df["household_income"] = 0.0

    
    return df
