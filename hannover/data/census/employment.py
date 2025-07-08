import pandas as pd
import numpy as np
import os

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
    grouped_pop = df_population.groupby(["age_group", "sex"], as_index=False, observed=True)["weight"].sum()

    # Calculates the proportion of the population that each group (age_group + sex) represents
    grouped_pop["pop_share"] = grouped_pop["weight"] / grouped_pop["weight"].sum()

    # Apply real 2023 employment distribution
    emp_age_dist = {
        "0": 0.09,
        "25": 0.709,
        "55": 0.201
    }

    # Calculate total employment based on actual population × 2023 employment rate
    employment_rate_2023 = 10761 / 23308
    total_population = df_population["weight"].sum()
    total_employment_est = total_population * employment_rate_2023

    # Compute employment per group proportionally to population share within each age group
    df_employment_parts = []
    for age_group in emp_age_dist:
        # Total people in this age group
        total_in_age = grouped_pop[grouped_pop["age_group"] == age_group]["weight"].sum()
        
        # Real employment in this age group
        total_employed_in_age = emp_age_dist[age_group] * total_employment_est

        for sex in ["male", "female"]:
            subgroup = grouped_pop[(grouped_pop["age_group"] == age_group) & (grouped_pop["sex"] == sex)]
            
            if subgroup.empty:
                continue

            sex_weight = subgroup["weight"].values[0]
            sex_share_in_age = sex_weight / total_in_age
            employed_weight = total_employed_in_age * sex_share_in_age

            df_employment_parts.append({
                "departement_id": "03241",
                "age_class": age_group,
                "sex": sex,
                "initial_weight": int(employed_weight)
            })

    df_employment = pd.DataFrame(df_employment_parts)
    
    # Adjust final weights to match 46.6% female, 53.4% male (from 2023 data)
    female_share_target = 0.466
    male_share_target = 0.534

    total_female_init = df_employment[df_employment["sex"] == "female"]["initial_weight"].sum()
    total_male_init = df_employment[df_employment["sex"] == "male"]["initial_weight"].sum()

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
    df_employment["departement_id"] = df_employment["departement_id"].astype(str)
    df_employment = df_employment.drop(columns="initial_weight")
    
    return df_employment
