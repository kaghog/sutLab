
import pandas as pd
import numpy as np
import os

"""
This stage loads the driving license data for Seville provided at municipality level.

"""

def configure(context):
    context.config("data_path")
    context.config("seville.licenses_path1", "licenses_2024.xlsx")
    context.config("seville.licenses_path2", "censo_conductores202510.txt")

    context.stage("seville.data.census.population")

    context.config("seville_city_data_only")


def execute(context):

    df_population = context.stage("seville.data.census.population").copy()

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

    df_province['sex'] = df_province['sex'].map({'M': 'male', 'V': 'female'})
    df_province['sex'] = df_province['sex'].astype("category")

    df_province = df_province[df_province["age_class"]!= "Se desconoce"] # remove rows with unknown age, these are insignificant
    condition = df_province["age_class"].str.startswith("Mas")
    df_province.loc[condition, "age_class"] = 74 # extracts upper bound from "More than 74 years"
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
    
    if context.config("seville_city_data_only") == True:
        result_df = result_df[result_df["municipality_id"] == "41091"]

    return result_df

def validate(context):

    CSV_FILE1 = f"{context.config('data_path')}/{context.config('seville.licenses_path1')}"
    if not os.path.exists(CSV_FILE1):
        raise RuntimeError(f"Driving license data is not available at location {CSV_FILE1}")

    CSV_FILE2 = f"{context.config('data_path')}/{context.config('seville.licenses_path2')}"
    if not os.path.exists(CSV_FILE2):
        raise RuntimeError(f"Driving license data is not available at location {CSV_FILE2}")


    return os.path.getsize(CSV_FILE1), os.path.getsize(CSV_FILE2)
