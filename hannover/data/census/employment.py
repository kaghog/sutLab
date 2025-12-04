import pandas as pd

"""
This stage loads the raw employment data for Hannover
"""


def configure(context):
    context.stage("hannover.data.census.population")


# Map numeric age_class to age_group buckets
def map_age_group(age):
    if age < 25:
        return "0"
    elif 25 <= age <= 54:
        return "25"
    else:
        return "55"


def execute(context):
    # Load the population data for Hannover
    df_population = context.stage("hannover.data.census.population")

    df_population["age_group"] = df_population["age_class"].apply(map_age_group)

    # Group by commune_id, departement_id, kreis_code, age_group, and sex
    grouped_pop = df_population.groupby(
        ["commune_id", "departement_id", "kreis_code", "age_group", "sex"],
        as_index=False,
        observed=True,
    )["weight"].sum()

    # Calculates the proportion of the population that each group represents
    grouped_pop["pop_share"] = grouped_pop["weight"] / grouped_pop["weight"].sum()

    # Apply real 2023 employment distribution
    emp_age_dist = {"0": 0.09, "25": 0.709, "55": 0.201}

    # Calculate total employment based on actual population × 2023 employment rate
    employment_rate_2023 = 10761 / 23308
    total_population = df_population["weight"].sum()
    total_employment_est = total_population * employment_rate_2023

    # Compute employment per group proportionally to population share within each age group and location
    df_employment_parts = []
    for age_group in emp_age_dist:
        # Total people in this age group across all locations
        total_in_age = grouped_pop[grouped_pop["age_group"] == age_group][
            "weight"
        ].sum()

        # Real employment in this age group
        total_employed_in_age = emp_age_dist[age_group] * total_employment_est

        # Iterate over each location (commune_id, departement_id, kreis_code combination)
        for _, row in grouped_pop[grouped_pop["age_group"] == age_group].iterrows():
            commune_id = row["commune_id"]
            departement_id = row["departement_id"]
            kreis_code = row["kreis_code"]
            sex = row["sex"]

            location_share_in_age = row["weight"] / total_in_age
            employed_weight = total_employed_in_age * location_share_in_age

            df_employment_parts.append(
                {
                    "commune_id": commune_id,
                    "departement_id": departement_id,
                    "kreis_code": kreis_code,
                    "age_class": age_group,
                    "sex": sex,
                    "initial_weight": employed_weight,
                }
            )

    df_employment = pd.DataFrame(df_employment_parts)

    # Adjust final weights to match 46.6% female, 53.4% male (from 2023 data)
    female_share_target = 0.466
    male_share_target = 0.534

    total_female_init = df_employment[df_employment["sex"] == "female"][
        "initial_weight"
    ].sum()
    total_male_init = df_employment[df_employment["sex"] == "male"][
        "initial_weight"
    ].sum()

    # Compute scaling factors
    female_scaling = (female_share_target * total_employment_est) / total_female_init
    male_scaling = (male_share_target * total_employment_est) / total_male_init

    # Apply scaling
    def scale_weight(row):
        if row["sex"] == "female":
            return row["initial_weight"] * female_scaling
        else:
            return row["initial_weight"] * male_scaling

    df_employment["weight"] = df_employment.apply(scale_weight, axis=1)
    df_employment["weight"] = df_employment["weight"].round().astype(int)
    df_employment["age_class"] = df_employment["age_class"].astype(int)
    df_employment["commune_id"] = df_employment["commune_id"].astype("category")
    df_employment["departement_id"] = df_employment["departement_id"].astype("category")
    df_employment["kreis_code"] = df_employment["kreis_code"].astype("category")
    df_employment = df_employment.drop(columns="initial_weight")

    return df_employment[
        ["commune_id", "departement_id", "kreis_code", "sex", "age_class", "weight"]
    ]
