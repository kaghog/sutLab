"""
This stage prepares census targets for IPU.
"""
import pandas as pd

def configure(context):
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.data.census.households")
    context.stage("asuncion.data.census.employment")
    context.config("sampling_rate", 1.0)
    context.config("IPU_aggregation_level")


def execute(context):

    aggregation_level = context.config("IPU_aggregation_level")


    df_population = context.stage("asuncion.data.census.population")
    df_households = context.stage("asuncion.data.census.households")
    df_employment = context.stage("asuncion.data.census.employment")

    sampling_rate = context.config("sampling_rate")




    aggregation_areas = sorted(df_population[aggregation_level].unique())

    print(f"Preparing IPU targets for {len(aggregation_areas)} {aggregation_level[:-3]}s...")

    # ------------------------------------------------------------------
    # Population → employment age-bin mapping (lower bound representation)
    # ------------------------------------------------------------------
    # it is actually matching 1:1 except people with age<15
    
    AGE_BIN_MAPPING = {
        0: 0,
        5: 0,
        10: 0,
        15: 15,
        20: 20,
        25: 25,
        30: 30,
        35: 35,
        40: 40,
        45: 45,
        50: 50,
        55: 55,
        60: 60,
        65: 65,
    }
    # add empty rows of employment for persons younger than 15
    df_employment_young = df_employment[df_employment['age_class'] == 15].copy()
    df_employment_young['weight'] = 0
    df_employment_young['age_class'] = 0

    df_employment = pd.concat([df_employment, df_employment_young])

    # ------------------------------------------------------------------------

    targets_by_aggregation_area = {}

    for aggregation_area_id in aggregation_areas:
        targets_by_aggregation_area[aggregation_area_id] = {}

        df_pop = df_population[df_population[aggregation_level] == aggregation_area_id]
        df_hh = df_households[df_households[aggregation_level] == aggregation_area_id]
        df_emp = df_employment[df_employment[aggregation_level] == aggregation_area_id]

        # --------------------------------------------------------------
        # Person targets: age × sex
        # --------------------------------------------------------------
        age_sex_targets = (
            df_pop.groupby(["sex", "age_class"])["weight"]
            .sum()
            .mul(sampling_rate)
            .to_dict()
        )

        targets_by_aggregation_area[aggregation_area_id]["age_sex"] = age_sex_targets

        # --------------------------------------------------------------
        # Household category targets
        # --------------------------------------------------------------
        household_category_targets = {
            1: df_hh["weight"].sum(),
        }

        household_category_targets = {
            k: v * sampling_rate for k, v in household_category_targets.items()
        }

        targets_by_aggregation_area[aggregation_area_id]["household_category"] = household_category_targets

        # --------------------------------------------------------------
        # Employment targets
        # --------------------------------------------------------------
        # Coarse employment totals (sex × employment age bin)
        emp_coarse = (
            df_emp.groupby(["sex", "age_class"])["weight"]
            .sum()
            .mul(sampling_rate)
        )

        # Map population ages to employment bins
        df_pop_emp = df_pop.copy()
        df_pop_emp["emp_age_bin"] = df_pop_emp["age_class"].map(AGE_BIN_MAPPING)

        # Population per coarse + fine bin
        pop_by_coarse_and_fine = (
            df_pop_emp
            .groupby(["sex", "emp_age_bin", "age_class"])["weight"]
            .sum()
            .reset_index()
        )

        # Total population per coarse employment bin
        pop_totals = (
            df_pop_emp
            .groupby(["sex", "emp_age_bin"])["weight"]
            .sum()
            .reset_index()
            .rename(columns={"weight": "total_pop"})
        )

        pop_by_coarse_and_fine = pop_by_coarse_and_fine.merge(
            pop_totals, on=["sex", "emp_age_bin"]
        )

        pop_by_coarse_and_fine["proportion"] = (
            pop_by_coarse_and_fine["weight"] /
            pop_by_coarse_and_fine["total_pop"]
        )

        # Disaggregate employment to fine age bins
        employment_targets = {}

        for _, row in pop_by_coarse_and_fine.iterrows():
            sex = row["sex"]
            emp_age_bin = row["emp_age_bin"]
            age_class = row["age_class"]
            proportion = row["proportion"]

            if (sex, emp_age_bin) in emp_coarse.index:
                emp_total = emp_coarse[(sex, emp_age_bin)]
                employed = emp_total * proportion

                employment_targets[(sex, age_class, True)] = int(round(employed))

        # Unemployed as residual
        for (sex, age_class), pop_total in age_sex_targets.items():
            employed = employment_targets.get((sex, age_class, True), 0)
            unemployed = pop_total - employed
            employment_targets[(sex, age_class, False)] = max(0, unemployed)

        targets_by_aggregation_area[aggregation_area_id]["employment"] = employment_targets

    print(f"✓ Prepared targets for {len(targets_by_aggregation_area)} {aggregation_level[:-3]}s")

    print(targets_by_aggregation_area)

    return targets_by_aggregation_area
