"""
This stage prepares census targets at departement level for IPU.

Aggregates from commune (mikrobezirk) to departement (neighborhood):
- Person-level: age_class x sex
- Household-level: household_size
# - Employment: age_class x sex x employed
"""


def configure(context):
    context.stage("hannover.data.census.population")
    context.stage("hannover.data.census.households")
    # context.stage("hannover.data.census.employment")
    context.config("sampling_rate", 1.0)


def execute(context):
    df_population = context.stage("hannover.data.census.population")
    df_households = context.stage("hannover.data.census.households")
    # df_employment = context.stage("hannover.data.census.employment")

    sampling_rate = context.config("sampling_rate")
    departements = sorted(df_population["departement_id"].unique())

    print(f"Preparing IPU targets for {len(departements)} departements...")
    targets_by_dept = {}

    for dept_id in departements:
        targets_by_dept[dept_id] = {}

        df_pop_dept = df_population[df_population["departement_id"] == dept_id]
        df_hh_dept = df_households[df_households["departement_id"] == dept_id]
        # df_emp_dept = df_employment[df_employment["departement_id"] == dept_id]

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
        # Total household constraint
        total_households = sum(household_size_targets.values())
        targets_by_dept[dept_id]["total_households"] = {None: total_households}

        # Total population constraint
        total_population = sum(age_sex_targets.values())
        targets_by_dept[dept_id]["total_population"] = {None: total_population}

    print(f"✓ Prepared targets for {len(targets_by_dept)} departements")
    print(
        f"Sample (dept {departements[0]}): {sum(targets_by_dept[departements[0]]['household_size'].values())} households"
    )

    return targets_by_dept
