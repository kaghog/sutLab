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
    context.stage("seville.data.census.licenses")
    context.config("sampling_rate", 1.0)
    context.config("IPU_aggregation_level")


def execute(context):

    aggregation_level = context.config("IPU_aggregation_level")


    df_population = context.stage("seville.data.census.population").copy()
    df_households = context.stage("seville.data.census.households").copy()
    df_employment = context.stage("seville.data.census.employment").copy()
    df_licenses = context.stage("seville.data.census.licenses").copy()

    sampling_rate = context.config("sampling_rate")


    MAP_COLUMNS = {
        "province": "province_id",
        "municipality": "municipality_id",
        "census_section": "census_section_id",
        "age": "age_class",
        "province_id": "departement_id",
        "census_section_id": "commune_id"
    }
    for df in [df_population, df_employment, df_households, df_licenses]:
        df.rename(MAP_COLUMNS, axis=1, inplace=True)


    aggregation_areas = sorted(df_population[aggregation_level].unique())

    # Cap population to 80+ age class
    # clip 70+ population to 70+ age class
    df_population["age_class"] = df_population["age_class"].clip(upper=80)

    df_population = (
        df_population
        .groupby(list(set(df_population.columns) - {"weight"}))["weight"]
        .sum()
        .reset_index()
    )


    df_employment["employed"] = True
    df_licenses["has_license"] = True

    print(f"Preparing IPU targets for {len(aggregation_areas)} {aggregation_level[:-3]}s...")

    targets_by_aggregation_area = {}

    for aggregation_area_id in aggregation_areas:
        targets_by_aggregation_area[aggregation_area_id] = {}

        df_pop = df_population[df_population[aggregation_level] == aggregation_area_id]
        df_hh = df_households[df_households[aggregation_level] == aggregation_area_id]
        df_emp = df_employment[df_employment[aggregation_level] == aggregation_area_id]
        df_lic = df_licenses[df_licenses[aggregation_level] == aggregation_area_id]

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

        targets_by_aggregation_area[aggregation_area_id]["household_size_capped"] = household_size_targets

        # --------------------------------------------------------------
        # Employment targets
        # --------------------------------------------------------------

        employment_targets = (
            df_emp.groupby(["sex", "age_class", "employed"])["weight"]
            .sum()
            .mul(sampling_rate)
            .to_dict()
        )
        targets_by_aggregation_area[aggregation_area_id]["employment"] = employment_targets

        #--------------------------------------------------------------
        # License targets
        #--------------------------------------------------------------
        
        license_targets = (
            df_lic.groupby(["sex", "age_class", "has_license"])["weight"]
            .sum()
            .mul(sampling_rate)
            .to_dict()
        )
        targets_by_aggregation_area[aggregation_area_id]["license"] = license_targets

    print(f"✓ Prepared targets for {len(targets_by_aggregation_area)} {aggregation_level[:-3]}s")


    return targets_by_aggregation_area
