import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

"""
This stage provides a zero household income for all households as it is needed in 
downstream stages for Spain.
"""

def configure(context):
    context.stage("synthesis.population.sampled")

def execute(context):
    # Load data
    df = context.stage("synthesis.population.sampled")[["household_id"]]
    df = df.drop_duplicates("household_id")
    # Format
    if "income_class" in df.columns and "household_income" not in df.columns:
        df["household_income"] = df["income_class"]
    elif "household_income" not in df.columns:        
        df["household_income"] = 0.0
    
    return df
