import pandas as pd

"""
This stage extracts household-level census data for Hannover from raw census.
Provides household size distribution at mikrobezirk (commune) level for IPU constraints.
"""


def configure(context):
    context.stage("hannover.data.spatial.codes")
    context.stage("hannover.data.census.raw")


def execute(context):
    df_raw = context.stage("hannover.data.census.raw")

    # Prepare mikrobezirk_code for merge
    df_raw["mikrobezirk_code"] = df_raw["mikrobezirk_code"].astype(str).str.zfill(4)

    # Get spatial codes
    df_spatial_codes = context.stage("hannover.data.spatial.codes")[
        ["commune_id", "departement_id", "kreis_code"]
    ].copy()
    df_spatial_codes = df_spatial_codes.rename(
        columns={"commune_id": "mikrobezirk_code"}
    )

    # Extract household columns
    household_cols = [
        "mikrobezirk_code",
        "total_households",
        "households_1_person",
        "households_2_persons",
        "households_3_persons",
        "households_4_persons",
        "households_5_persons",
        "households_6plus_persons",
    ]

    df_households = df_raw[household_cols].copy()

    # Merge with spatial codes
    df_households = pd.merge(
        df_households, df_spatial_codes, on="mikrobezirk_code", how="left"
    )

    df_households = df_households.rename(columns={"mikrobezirk_code": "commune_id"})

    # drop missing
    df_households = df_households[df_households["departement_id"].notna()]

    df_households["commune_id"] = df_households["commune_id"].astype("category")
    df_households["departement_id"] = df_households["departement_id"].astype("category")
    df_households["kreis_code"] = df_households["kreis_code"].astype("category")

    for col in household_cols[1:]:
        df_households[col] = df_households[col].fillna(0).astype(int)

    return df_households[
        [
            "commune_id",
            "departement_id",
            "kreis_code",
            "total_households",
            "households_1_person",
            "households_2_persons",
            "households_3_persons",
            "households_4_persons",
            "households_5_persons",
            "households_6plus_persons",
        ]
    ]
