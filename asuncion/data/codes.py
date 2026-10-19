import numpy as np
import pandas as pd
import geopandas as gpd
import unicodedata

"""
Yield home zones for Spain based on synthetic population data.
"""

def configure(context):
    context.config("asuncion.administration_codes", "Codigos de barrios y localidades INE 2022.xlsx")
    context.config("data_path")


NAME_MAPPING = {
        "MADAME ELISA ALICIA LINCH": "MADAME ELISA ALICIA LYNCH",
        "ZEBALLOS KUE": "ZEBALLOS CUE"
    }


def remove_diacritics(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(c)
    )

def normalize_codes(df, geo_codes):
    cols = [c for c in ["departement", "district", "borough"] if c in df.columns]
    id_cols = [f"{c}_id" for c in cols]

    df[cols] = df[cols].applymap(remove_diacritics)
    geo_codes[cols] = geo_codes[cols].applymap(remove_diacritics)

    if "borough" in cols:
        df["borough"] = df["borough"].map(NAME_MAPPING).fillna(df["borough"])


    lookup = geo_codes[cols + id_cols].drop_duplicates(cols)

    # Handle cases where borough == district
    if "district" in cols and "borough" in cols:
        district_cols = [c for c in cols if c != "borough"]
        district_ids = [f"{c}_id" for c in district_cols]

        district_lookup = (
            geo_codes[district_cols + district_ids]
            .drop_duplicates(district_cols)
            .assign(
                borough=lambda x: x["district"],
                borough_id=lambda x: x["district_id"]
            )
        )

        # Prioritize district-level mapping for matching names
        lookup = pd.concat(
            [district_lookup[cols + id_cols], lookup],
            ignore_index=True
        ).drop_duplicates(cols, keep="first")

    result = df.merge(lookup, on=cols, how="inner", validate="many_to_one")

    assert len(result) == len(df), (
        f"Normalization changed the number of rows "
        f"{len(result)} == {len(df)}"
        f"Missing boroughs: {set(df['borough']) - set(result['borough'])}"
    )

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
        "COD_BARLOC": "borough_code",
        "BARRIOS Y LOCALIDADES": "borough",
    })

    for c, max_len in zip(["departement_id", "district_code", "borough_code"], [2, 2, 3]):
        df[c] = df[c].astype("string").str.strip()
        max_length = df[c].str.len().max()
        assert max_length <= max_len, c
        df[c] = df[c].str.rjust(max_len, '0')

    df["district_id"] = df["departement_id"] + df["district_code"]
    df["borough_id"] = df["district_id"] + df["borough_code"]

    # Come names are different in the code excel and other datasets
    # This I need to map them to the same names, so I can merge them later

    df["borough"] = df["borough"].map(NAME_MAPPING).fillna(df["borough"])

    return df