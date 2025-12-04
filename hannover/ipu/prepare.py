"""
This stage prepares census targets at departement level for IPU.

Aggregates from commune (mikrobezirk) to departement (neighborhood):
- Person-level: age_class x sex
- Household-level: household_size
- Employment: age_class x sex x employed
"""


def configure(context):
    context.stage("hannover.data.census.population")
    context.stage("hannover.data.census.households")
    context.stage("hannover.data.census.employment")
    context.config("sampling_rate", 1.0)


def execute(context):
    df_population = context.stage("hannover.data.census.population")
    df_households = context.stage("hannover.data.census.households")
    df_employment = context.stage("hannover.data.census.employment")

    sampling_rate = context.config("sampling_rate")
    departements = sorted(df_population["departement_id"].unique())

    print(f"Preparing IPU targets for {len(departements)} departements...")

    targets_by_dept = {}

    for dept_id in departements:
        targets_by_dept[dept_id] = {}

        df_pop_dept = df_population[df_population["departement_id"] == dept_id]
        df_hh_dept = df_households[df_households["departement_id"] == dept_id]
        df_emp_dept = df_employment[df_employment["departement_id"] == dept_id]

        # Age × Sex targets
        age_sex_targets = (
            df_pop_dept.groupby(["sex", "age_class"])["weight"].sum().to_dict()
        )
        age_sex_targets = {k: v * sampling_rate for k, v in age_sex_targets.items()}
        targets_by_dept[dept_id]["age_sex"] = age_sex_targets

        # Household size targets
        household_size_targets = {
            1: df_hh_dept["households_1_person"].sum(),
            2: df_hh_dept["households_2_persons"].sum(),
            3: df_hh_dept["households_3_persons"].sum(),
            4: df_hh_dept["households_4_persons"].sum(),
            5: df_hh_dept["households_5_persons"].sum(),
            6: df_hh_dept["households_6plus_persons"].sum(),
        }
        household_size_targets = {
            k: v * sampling_rate for k, v in household_size_targets.items()
        }
        targets_by_dept[dept_id]["household_size"] = household_size_targets

        # Employment targets: disaggregate from coarse census bins to fine population bins
        # Census has 3 age bins [0, 25, 55] = [0-24, 25-54, 55+]
        # HTS Population has 9 age bins [0, 6, 15, 18, 24, 30, 45, 65, 80]

        emp_coarse = (
            df_emp_dept.groupby(["sex", "age_class"])["weight"].sum() * sampling_rate
        )

        age_bin_mapping = {
            0: 0,
            6: 0,
            15: 0,
            18: 0,  # 0-24 → census bin 0
            24: 25,
            30: 25,
            45: 25,  # 25-64 → census bin 25 (note: 45-64 includes some 55+)
            65: 55,
            80: 55,  # 65+ → census bin 55
        }

        df_pop_dept_copy = df_pop_dept.copy()
        df_pop_dept_copy["emp_age_bin"] = df_pop_dept_copy["age_class"].map(
            age_bin_mapping
        )

        # Calculate population proportions within each census bin
        pop_by_coarse_and_fine = (
            df_pop_dept_copy.groupby(["sex", "emp_age_bin", "age_class"])["weight"]
            .sum()
            .reset_index()
        )

        pop_totals_coarse = (
            df_pop_dept_copy.groupby(["sex", "emp_age_bin"])["weight"]
            .sum()
            .reset_index()
            .rename(columns={"weight": "total_pop"})
        )

        pop_by_coarse_and_fine = pop_by_coarse_and_fine.merge(
            pop_totals_coarse, on=["sex", "emp_age_bin"]
        )
        pop_by_coarse_and_fine["proportion"] = (
            pop_by_coarse_and_fine["weight"] / pop_by_coarse_and_fine["total_pop"]
        )

        # Disaggregate employment to fine bins
        employment_targets = {}

        for _, row in pop_by_coarse_and_fine.iterrows():
            sex = row["sex"]
            emp_age_bin = row["emp_age_bin"]
            fine_age_class = row["age_class"]
            proportion = row["proportion"]

            if (sex, emp_age_bin) in emp_coarse.index:
                emp_total = emp_coarse[(sex, emp_age_bin)]
                emp_disaggregated = emp_total * proportion

                employment_targets[(sex, fine_age_class, True)] = int(
                    round(emp_disaggregated)
                )

        # Calculate unemployed as residual
        for (sex, age_class), pop_total in age_sex_targets.items():
            emp_count = employment_targets.get((sex, age_class, True), 0)
            unemployed_count = pop_total - emp_count
            employment_targets[(sex, age_class, False)] = max(0, unemployed_count)

        targets_by_dept[dept_id]["employment"] = employment_targets

    print(f"✓ Prepared targets for {len(targets_by_dept)} departements")
    print(
        f"  Sample (dept {departements[0]}): {sum(targets_by_dept[departements[0]]['household_size'].values())} households"
    )

    return targets_by_dept
