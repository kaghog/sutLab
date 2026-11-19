
import pandas as pd
import numpy as np
import os
from .population_growth_rate import (
    add_scaling_factor_between_years,
    population_to_license_age_classes
)

"""
This stage loads the driving license data for Seville provided at municipality level.

"""

def configure(context):
    context.config("data_path")
    context.config("seville.licenses_path1", "licenses_2024.xlsx")
    context.config("seville.licenses_path2", "conductores_censo_2023_censo_prov_sexo_clase_edad_2023.txt")
    context.stage("seville.data.census.population_growth_rate")

    context.stage("seville.data.census.population")

    if context.config("seville_census_area_selection") == 'agglomeration':
        context.stage("seville.data.select_agglomeration")


def scale_to_census_section(context, df_municipality_expanded):
    df_population = context.stage("seville.data.census.population").copy()

    total_before = df_municipality_expanded["weight"].sum()


    df_municipality_expanded = df_municipality_expanded.drop(columns=['proportion'])


    # ========== Rescale driving licenses to census section level by age and sex ==========

    # Ensure municipality IDs have the same format
    df_population["municipality_id"] = (
        df_population["municipality_id"].astype("string").str.zfill(5)
    )
    df_municipality_expanded["municipality_id"] = (
        df_municipality_expanded["municipality_id"].astype("string").str.zfill(5)
    )

    # Expand 5-year population bins into individual ages
    # Assume uniform population distribution within each 5-year bin

    # Map population age classes to driving license age classes
    df_pop = population_to_license_age_classes(df_population)

    # Aggregate population by municipality, census section, age and sex
    df_population_grouped = (
        df_pop
        .groupby(
            ["municipality_id", "census_section_id", "sex", "age_class"],
            as_index=False
        )["weight"]
        .sum()
    )

    # Compute population proportions within each municipality
    # separately for each age class and sex
    df_population_grouped["proportion"] = (
        df_population_grouped["weight"] /
        df_population_grouped
            .groupby(["municipality_id", "sex", "age_class"])["weight"]
            .transform("sum")
    )

    # Merge population proportions with driving license data
    df_commune_expanded = df_municipality_expanded.merge(
        df_population_grouped[
            ["municipality_id", "census_section_id",
             "sex", "age_class", "proportion"]
        ],
        on=["municipality_id", "sex", "age_class"],
        how="left"
    )

    # Check that all license groups have matching population data
    assert df_commune_expanded["proportion"].notna().all(), \
        "Some age-sex groups could not be matched to census population"

    # Rescale driving license counts to census section level
    df_commune_expanded["weight"] = (
        df_commune_expanded["weight"] *
        df_commune_expanded["proportion"]
    )

    # Remove temporary proportion column
    df_commune_expanded = df_commune_expanded.drop(
        columns=["proportion"]
    )

    # Verify that the total population count did not change after scaling
    total_after = df_commune_expanded["weight"].sum()
    assert np.isclose(total_before, total_after), \
        f"Total weight mismatch after scaling: before={total_before}, after={total_after}"

    return df_commune_expanded



def execute(context):
    # ========== Load municipality-level data (by sex) ==========

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("seville.licenses_path1"))
    print(f"Loading licenses data from {EXCEL_PATH}")
    SHEET_NAME = "DatosMunicipalesGeneral_2024"

    df_municipality = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME,
                                    dtype={"Código INE":"string"} # Codigo INE is municipality code
                                    )
    
    df_municipality = df_municipality.iloc[:,[0,4,5,6,7,8,9]]
    colnames = [
        "municipality_id", 
        "population_total", 
        "male_total", 
        "female_total", 
        "drivers_male", 
        "drivers_female", 
        "drivers_total"
    ]
    df_municipality.columns = colnames
    df_municipality["municipality_id"] = df_municipality["municipality_id"].astype(str)

    # Remove "municipio sin especificar" ("unspecified municipality") row, that is not significance and is present for each province    
    df_municipality = df_municipality[~df_municipality["municipality_id"].str.endswith("000")]
    # Filter only Seville municipalities
    df_municipality = df_municipality[df_municipality["municipality_id"].str.startswith("41")]

    df_municipality = df_municipality.astype(
        {
        "municipality_id": "string", 
        "population_total": "int64", 
        "male_total": "int64", 
        "female_total": "int64", 
        "drivers_male": "int64", 
        "drivers_female": "int64", 
        "drivers_total": "int64"
        }
    )


    df_municipality = df_municipality[["municipality_id", "drivers_male", "drivers_female"]]
    df_municipality = df_municipality.melt(id_vars=["municipality_id"], value_vars=['drivers_male', 'drivers_female'],
                  var_name='sex', value_name='weight')
    df_municipality['sex'] = df_municipality['sex'].map(
        {'drivers_male': 'male', 'drivers_female': 'female'}
        ).astype('str')

    # ========== Load province-level data (by age and sex) ==========

    FILE_PATH = "{}/{}".format(context.config("data_path"),context.config("seville.licenses_path2"))
    print(f"Loading licenses data from {FILE_PATH}")

    df_province = pd.read_csv(FILE_PATH, sep="|", 
                              usecols=['COD_PROVINCIA', 'IND_SEXO', 'EDAD', 'NUM_LICENCIAS_PERMISOS'], 
                              dtype = {"COD_PROVINCIA":str},
                              encoding="latin-1"
                              )
    colnames = [
        "province_id", 
        "sex", 
        "age_class", 
        "weight",
    ]
    df_province.columns = colnames
    
    df_province = df_province[df_province["province_id"] == "41"]
    df_province["province_id"] = df_province["province_id"].astype('str')

    df_province['sex'] = df_province['sex'].map({'M': 'female', 'V': 'male'})
    df_province['sex'] = df_province['sex'].astype("category")

    df_province = df_province[df_province["age_class"]!= "Se desconoce"] # remove rows with unknown age, these are insignificant
    condition = df_province["age_class"] == "Más de 74 años"
    df_province.loc[condition, "age_class"] = 75 # extracts upper bound from "More than 74 years"
    df_province.loc[~condition, "age_class"] = df_province.loc[~condition, "age_class"].str[0:2] 
    df_province["age_class"] = df_province["age_class"].astype("int64")

    # ========== Merge both datasets ==========
    group_cols = ['sex']
    df_province['total'] = df_province.groupby(group_cols)['weight'].transform('sum')
    df_province['proportion'] = df_province['weight'] / df_province['total']
    df_province['proportion'] = df_province['proportion'].replace(np.nan, 0)

    df_municipality_expanded = pd.merge(df_municipality, df_province[group_cols + ["age_class" , 'proportion', 'province_id']], on=group_cols, how='left')
    df_municipality_expanded['weight'] = df_municipality_expanded['weight'] * df_municipality_expanded['proportion']

    result_df = df_municipality_expanded

    selected_area = context.config("seville_census_area_selection")
    if selected_area == 'province':
        # no changes
        result_df = result_df
    elif selected_area == 'agglomeration':
        # use data of municipalities inside agglomeration
        agglomeration_mun = context.stage("seville.data.select_agglomeration")
        result_df = result_df[result_df["municipality_id"].isin(agglomeration_mun['municipality_id'])]
    elif selected_area == 'municipality':
        # use data of Seville municipality only
        result_df = result_df[result_df["municipality_id"] == "41091"]
    else:
        raise NotImplementedError

    result_df = scale_to_census_section(context, result_df)

    # ========== Rescale licenses from 2024 to 2022 ==========

    df_population_growth = context.stage("seville.data.census.population_growth_rate").copy()

    # Harmonize population age bins with license age bins
    df_population_growth = population_to_license_age_classes(df_population_growth)

    # Calculate scaling factors by municipality, age and sex
    result_df = add_scaling_factor_between_years(
        result_df,
        df_population_growth,
        year_from=2024,
        year_to=2022,
        matching_attributes=["municipality_id", "sex", "age_class"]
    )

    # Check for invalid scaling factors
    assert np.isfinite(result_df["scaling_factor"]).all(), \
        "Missing or invalid age-sex population scaling factors"

    # Apply temporal scaling
    result_df["weight"] = (result_df["weight"] * result_df["scaling_factor"])
    result_df = result_df.drop(columns=["scaling_factor"])

    return result_df


def validate(context):
    filenames = [
        "seville.licenses_path1",
        "seville.licenses_path2",
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Census driving license data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
