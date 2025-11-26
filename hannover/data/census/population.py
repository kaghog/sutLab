import pandas as pd

"""
This stage loads the raw census data for Hannover provided at Bezirk level.

"""


def configure(context):
    context.stage("hannover.data.spatial.codes")
    context.config("data_path")
    context.stage("hannover.data.census.raw")
    context.config("check_spatial_codes", False)


def execute(context):
    df = context.stage("hannover.data.census.raw")

    df["mikrobezirk_code"] = df["mikrobezirk_code"].astype(str).str.zfill(4)

    df_spatial_codes = context.stage("hannover.data.spatial.codes")[
        ["commune_id", "departement_id", "kreis_code"]
    ].copy()

    # Rename commune_id to mikrobezirk_code for merge
    df_spatial_codes = df_spatial_codes.rename(
        columns={"commune_id": "mikrobezirk_code"}
    )

    # Melt to long format for both sexes
    df_long = pd.wide_to_long(
        df,
        stubnames=["male", "female"],
        i="mikrobezirk_code",
        j="age_class",
        sep="_",
        suffix="\\d+",
    )
    df_long = df_long.reset_index()

    df_male = df_long[["mikrobezirk_code", "age_class", "male"]].rename(
        columns={"male": "weight"}
    )
    df_male["sex"] = "male"
    df_female = df_long[["mikrobezirk_code", "age_class", "female"]].rename(
        columns={"female": "weight"}
    )
    df_female["sex"] = "female"

    df_result = pd.concat([df_male, df_female], ignore_index=True)
    df_result = df_result[df_result["weight"].notna()]

    # cleaning
    df_result["weight"] = df_result["weight"].astype(str).str.replace(",", "")
    df_result["weight"] = df_result["weight"].replace("-", 0)

    df_result["weight"] = (
        pd.to_numeric(df_result["weight"], errors="coerce").fillna(-99).astype(int)
    )
    if len(df_result[df_result["weight"] == -99]) > 0:
        raise ValueError(
            "There are still -99 values in the weight column, indicating conversion issues."
        )

    # Merge with spatial codes to get all IDs
    df_result = pd.merge(df_result, df_spatial_codes, on="mikrobezirk_code", how="left")

    df_result = df_result.rename(columns={"mikrobezirk_code": "commune_id"})

    # Check for missing codes
    missing_codes = df_result[df_result["departement_id"].isna()]
    if len(missing_codes) > 0:
        print(
            f"Warning: {len(missing_codes)} rows have missing departement_id. Dropping them."
        )
        df_result = df_result[df_result["departement_id"].notna()]

    df_result["sex"] = df_result["sex"].astype("category")
    df_result["age_class"] = df_result["age_class"].astype(int)
    df_result["commune_id"] = df_result["commune_id"].astype("category")
    df_result["departement_id"] = df_result["departement_id"].astype("category")
    df_result["kreis_code"] = df_result["kreis_code"].astype("category")

    return df_result[
        ["commune_id", "departement_id", "kreis_code", "sex", "age_class", "weight"]
    ]
