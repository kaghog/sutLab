import numpy as np
import pandas as pd
import geopandas as gpd

"""
Yield home zones for Spain based on synthetic population data.
"""

def configure(context):
    context.config("asuncion.administration_codes", "Codigos de barrios y localidades INE 2022.xlsx")
    context.config("data_path")


def normalize_codes(df, geo_codes):
    cols = [c for c in ["departement", "district", "borrough"] if c in df.columns]
    lookup = geo_codes[cols + [f"{c}_id" for c in cols]].drop_duplicates(cols)

    result = df.merge(lookup, on=cols, how="inner")
    assert len(result) == len(df), f"Normalization changed the number of rows {len(result)} == {len(df)}"

    return result
    
def execute(context):
    # Load data
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.administration_codes')}"
    df = pd.read_excel(FILE_URL, sheet_name="BARRIOS Y LOCALIDADES", header=2)

    df = df.rename(columns={
        "COD_DEP": "departement_id",
        "DEPARTAMENTO": "departement",
        "COD_DIS": "district_code",
        "DISTRITO": "district",
        "COD_BARLOC": "borrough_code",
        "BARRIOS Y LOCALIDADES": "borrough",
    })

    for c in ["departement_id", "district_code", "borrough_code"]:
        df[c] = df[c].astype("string").str.strip()
        max_length = df["departement_id"].str.len().max()
        assert max_length <= 3
        df[c] = df[c].str.rjust(3, '0')
        print(f"{c} code has length of {max_length}")

    df["district_id"] = df["departement_id"] + df["district_code"]
    df["borrough_id"] = df["district_id"] + df["borrough_code"]

    return df