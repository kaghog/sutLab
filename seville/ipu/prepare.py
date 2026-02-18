"""
This stage prepares census targets at configured aggregation level for IPU (Sevilla).

Aggregates census data to aggregate level:
- Person-level: age_class x sex
- Household-level: household_size
- Employment: age_class x sex x employed

Handles mismatched population vs employment age bins by
proportional disaggregation and structural zeros.

NOTE: available aggregation levels: 
    - departement
    - commune
"""
import pandas as pd

def configure(context):
    context.stage("seville.data.census.population")
    context.stage("seville.data.census.households")
    context.stage("seville.data.census.employment")
    context.config("sampling_rate", 1.0)
    context.config("IPU_aggregation_level")


def execute(context):

    aggregation_level = context.config("IPU_aggregation_level")


    df_population = context.stage("seville.data.census.population")
    df_households = context.stage("seville.data.census.households")
    df_employment = context.stage("seville.data.census.employment")

    sampling_rate = context.config("sampling_rate")


    MAP_COLUMNS = {
        "province": "province_id",
        "municipality": "municipality_id",
        "census_section": "census_section_id",
        "age": "age_class",
        "province_id": "departement_id",
        "census_section_id": "commune_id"
    }
    for df in [df_population, df_employment, df_households]:
        df.rename(MAP_COLUMNS, axis=1, inplace=True)


    aggregation_areas = sorted(df_population[aggregation_level].unique())

    print(f"Preparing IPU targets for {len(aggregation_areas)} {aggregation_level[:-3]}s...")

    # ------------------------------------------------------------------
    # Population → employment age-bin mapping (lower bound representation)
    # ------------------------------------------------------------------
    AGE_BIN_MAPPING = {
        0: 0,
        5: 0,
        10: 0,
        15: 16,     # 15–19 → 16–19
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
        70: 70,     # 70–74 → 70+
        75: 70,
        80: 70,
        85: 70,
        90: 70,
        95: 70,
        100: 70
    }
    # add empty rows of employment for persons younger than 15
    df_employment_young = df_employment[df_employment['age_class'] == 16].copy()
    df_employment_young['weight'] = 0
    df_employment_young['age_class'] = 0

    print(df_employment_young)
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
        # Household size targets
        # --------------------------------------------------------------
        household_size_targets = {
            1: df_hh["households_1_person"].sum(),
            2: df_hh["households_2_persons"].sum(),
            3: df_hh["households_3_persons"].sum(),
            4: df_hh["households_4_persons"].sum(),
            5: df_hh["households_5plus_persons"].sum(),
        }

        household_size_targets = {
            k: v * sampling_rate for k, v in household_size_targets.items()
        }

        targets_by_aggregation_area[aggregation_area_id]["household_size"] = household_size_targets

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


    return targets_by_aggregation_area
