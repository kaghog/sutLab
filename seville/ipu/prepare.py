"""
This stage prepares census targets at departement level for IPU (Sevilla).

Aggregates census data to departement level:
- Person-level: age_class x sex
- Household-level: household_size
- Employment: age_class x sex x employed

Handles mismatched population vs employment age bins by
proportional disaggregation and structural zeros.
"""


def configure(context):
    context.stage("seville.data.census.population")
    context.stage("seville.data.census.households")
    context.stage("seville.data.census.employment")
    context.config("sampling_rate", 1.0)
    context.config("ignore_age_19_and_below")


def execute(context):
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
    for df in [df_population, df_employment]:
        df.rename(MAP_COLUMNS, axis=1, inplace=True)

    # ===================================================================================
    # ===================================================================================
    # THIS IS REMOVING YOUNG PEOPLE FROM CENSUS, JUST THAT IT MATCHES THE HTS SAMPLES
    if context.config("ignore_age_19_and_below") == True:
        for df in [df_population, df_employment]:
            df = df[df["age_class"] >= 20]
    # ===================================================================================
    # ===================================================================================



    departements = sorted(df_population["departement_id"].unique())

    print(f"Preparing IPU targets for {len(departements)} departements...")

    # ------------------------------------------------------------------
    # Population → employment age-bin mapping (lower bound representation)
    # ------------------------------------------------------------------
    AGE_BIN_MAPPING = {
        0: None,
        5: None,
        10: None,
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

    targets_by_departement = {}

    for departement_id in departements:
        targets_by_departement[departement_id] = {}

        df_pop = df_population[df_population["departement_id"] == departement_id]
        df_hh = df_households[df_households["departement_id"] == departement_id]
        df_emp = df_employment[df_employment["departement_id"] == departement_id]

        # --------------------------------------------------------------
        # Person targets: age × sex
        # --------------------------------------------------------------
        age_sex_targets = (
            df_pop.groupby(["sex", "age_class"])["weight"]
            .sum()
            .mul(sampling_rate)
            .to_dict()
        )

        targets_by_departement[departement_id]["age_sex"] = age_sex_targets

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

        targets_by_departement[departement_id]["household_size"] = household_size_targets

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

        # Keep only employable population (>=16)
        df_pop_emp = df_pop_emp[df_pop_emp["emp_age_bin"].notnull()]

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

        targets_by_departement[departement_id]["employment"] = employment_targets

    print(f"✓ Prepared targets for {len(targets_by_departement)} departements")


    return targets_by_departement
