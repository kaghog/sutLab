import numpy as np

"""
This stage adds additional attributes to the generated synthetic population from IPU.
"""


def configure(context):
    context.stage("seville.ipu.population")
    context.stage("seville.data.spatial.iris")
    context.config("ignore_age_19_and_below")

    context.config("random_seed")


def execute(context):
    df = context.stage("seville.ipu.population").copy()

    df_iris = context.stage("seville.data.spatial.iris")[
        ["departement_id", "commune_id", "iris_id"]
    ].drop_duplicates()

    print(f"Adding attributes to {len(df):,} persons from IPU...")

    # Spatial identifiers: departement -> commune -> iris
    dept_to_commune = df_iris.groupby("departement_id")["commune_id"].first().to_dict()
    commune_to_iris = dict(zip(df_iris["commune_id"], df_iris["iris_id"]))

    if "commune_id" not in df.columns:
        df["commune_id"] = df["departement_id"].map(dept_to_commune)

    if "iris_id" not in df.columns or df["iris_id"].isna().any():
        df["iris_id"] = df["commune_id"].map(commune_to_iris)

    df["commune_id"] = df["commune_id"].astype(str)
    df["iris_id"] = df["iris_id"].astype("category")

    # Household attributes
    if "household_size_capped" in df.columns and "household_size" not in df.columns:
        df["household_size"] = df["household_size_capped"]

    if "consumption_units" not in df.columns:
        df["consumption_units"] = 1.0
    if "couple" not in df.columns:
        df["couple"] = False

    # Person attributes
    if "studies" not in df.columns:
        df["studies"] = False
    if "socioprofessional_class" not in df.columns:
        df["socioprofessional_class"] = 0
    if "work_outside_region" not in df.columns:
        df["work_outside_region"] = False
    if "education_outside_region" not in df.columns:
        df["education_outside_region"] = False

    # Vehicle availability
    if "number_of_cars" not in df.columns:
        df["number_of_cars"] = 1
    if "number_of_bikes" not in df.columns:
        df["number_of_bikes"] = 1

    if "commute_mode" not in df.columns:
        df["commute_mode"] = np.nan

    # Assign unique person and household IDs
    if "new_hh_id" in df.columns:
        unique_hh_ids = df["new_hh_id"].unique()
        hh_id_mapping = dict(zip(unique_hh_ids, range(len(unique_hh_ids))))
        df["household_id"] = df["new_hh_id"].map(hh_id_mapping)
        df = df.drop(columns=["new_hh_id"])
    elif "household_id" in df.columns and df["household_id"].duplicated().any():
        print("WARNING: Recreating household_id due to duplicates")
        df["household_id"] = df.groupby(
            df["household_id"].astype(str) + "_" + df.index.astype(str)
        ).ngroup()

    df["person_id"] = np.arange(len(df))

    # Census IDs for compatibility with downstream stages
    df["census_person_id"] = df["person_id"]
    df["census_household_id"] = df["household_id"]

    print(
        f"Attributed population: {len(df):,} persons, "
        f"{df['household_id'].nunique():,} households, "
        f"{df['departement_id'].nunique()} departements"
    )

    # ===================================================================================
    # ===================================================================================
    # THIS IS REMOVING YOUNG PEOPLE FROM CENSUS, JUST THAT IT MATCHES THE HTS SAMPLES
    if context.config("ignore_age_19_and_below") == True:
        df = df[df["age_class"] >= 20]
    # ===================================================================================
    # ===================================================================================


    return df
