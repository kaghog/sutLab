
import numpy as np

"""
This stage adds additional attributes to the generated synthetic population from IPU.
"""


def configure(context):
    context.stage("seville.ipu.population")
    context.stage("seville.data.spatial.iris")
    context.stage("seville.data.census.population")
    context.config("random_seed")


def execute(context):
    df = context.stage("seville.ipu.population").copy()
    random = np.random.RandomState(context.config("random_seed"))

    df_iris = context.stage("seville.data.spatial.iris")[
        ["departement_id", "commune_id", "iris_id"]
    ].drop_duplicates()

    print(f"Adding attributes to {len(df):,} persons from IPU...")

    if "departement_id" not in df.columns and "commune_id" in df.columns:
        df["departement_id"] = df["commune_id"].str[:2]


    # Spatial identifiers: departement -> commune -> iris
    # Distribute households across communes within each departement based on population
    if "commune_id" not in df.columns:
        # Get population per commune for weighting
        df_pop = context.stage("seville.data.census.population")
        commune_pop = df_pop.groupby("commune_id")["weight"].sum().to_dict()

        # Assign commune_id to each household based on departement
        household_communes = {}
        for dept_id in df["departement_id"].unique():
            # Get all communes in this departement
            communes_in_dept = df_iris[df_iris["departement_id"] == dept_id][
                "commune_id"
            ].unique()

            # Get populations for weighting
            weights = np.array([commune_pop.get(c, 1.0) for c in communes_in_dept])
            weights = weights / weights.sum()

            # Get households in this departement
            dept_households = df[df["departement_id"] == dept_id][
                "household_id"
            ].unique()

            # Randomly assign communes based on population weights
            assigned_communes = random.choice(
                communes_in_dept, size=len(dept_households), p=weights
            )

            for hh_id, commune_id in zip(dept_households, assigned_communes):
                household_communes[hh_id] = commune_id

        df["commune_id"] = df["household_id"].map(household_communes)

    # Map commune to iris
    commune_to_iris = dict(zip(df_iris["commune_id"], df_iris["iris_id"]))
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
    # Map string household_id from IPU to sequential integers
    if "household_id" in df.columns:
        unique_hh_ids = df["household_id"].unique()
        hh_id_mapping = dict(zip(unique_hh_ids, range(len(unique_hh_ids))))
        df["household_id"] = df["household_id"].map(hh_id_mapping)

    df["person_id"] = np.arange(len(df))

    # Census IDs for compatibility with downstream stages
    df["census_person_id"] = df["person_id"]
    df["census_household_id"] = df["household_id"]

    print(
        f"Attributed population: {len(df):,} persons, "
        f"{df['household_id'].nunique():,} households, "
        f"{df['departement_id'].nunique()} departements"
    )



    return df
