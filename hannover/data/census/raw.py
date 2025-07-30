import pandas as pd
import numpy as np
import geopandas as gpd


"""
This stage loads the raw census data for Hannover provided at Bezirk level.

TODO: This could be replaced with a Germany-wide extract from GENESIS
"""

def configure(context):
    context.config("data_path")
    context.config("hannover.population_path", "census/Age_gender_MBZ.xlsx")

def execute(context):
    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("hannover.population_path"))
    print(f"Loading census data from {EXCEL_PATH}")
    SHEET_NAME = "Altersgruppen MBZ Geschlecht"
    AGE_GROUPS = [
        (0, "0 bis 5 Jahre"),
        (6, "6 bis 14 Jahre"),
        (15, "15 bis 17 Jahre"),
        (18, "18 bis 23 Jahre"),
        (24, "24 bis 29 Jahre"),
        (30, "30 bis 44 Jahre"),
        (45, "45 bis 64 Jahre"),
        (65, "65 bis 79 Jahre"),
        (80, "80+"),
    ]
    AGE_BOUNDS = [x[0] for x in AGE_GROUPS]

    # Read the sheet, skipping the descriptive header rows/columns
    # Assuming microbezirke is in the first column, then: 9 (male), 9 (female), 'gesamt'
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=7, header=None)
    df = df[~df[0].astype(str).str.contains("Gesamt", na=False)]
    df = df[df[0].astype(str).str.match(r"^\d+$")]

    colnames = (
    ["mikrobezirk_code"]
    + [f"male_{age[0]}" for age in AGE_GROUPS]
    + [f"female_{age[0]}" for age in AGE_GROUPS]
    + ["total_population"]
    )
    df = df.iloc[:, :len(colnames)]  # drop empty columns if present
    df.columns = colnames

    return df

