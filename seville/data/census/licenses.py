
import pandas as pd
import numpy as np

"""
This stage loads the driving license data for Seville provided at municipality level.

"""

def configure(context):
    context.config("data_path")
    context.config("seville.licenses_path1", "licenses_2024.xlsx")
    context.config("seville.licenses_path2", "censo_conductores202510.txt")

    context.stage("seville.data.census.population")


def execute(context):

    df_population = context.stage("seville.data.census.population").copy()

    # ---------------------------------------------------------------------------------
    # Importing license data per municipality by sex

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("seville.licenses_path1"))
    print(f"Loading licenses data from {EXCEL_PATH}")
    SHEET_NAME = "DatosMunicipalesGeneral_2024"

    df_municipality = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME,)
    
    df_municipality = df_municipality.iloc[:,[0,4,5,6,7,8,9]]
    colnames = [
        "municipality", 
        "population_total", 
        "male_total", 
        "female_total", 
        "drivers_male", 
        "drivers_female", 
        "drivers_total"
    ]
    df_municipality.columns = colnames

    # Remove "municipio sin especificar" ("unspecified municipality") row, that is not significance and is present for each province    
    df_municipality["municipality"] = df_municipality["municipality"].astype("string")
    df_municipality = df_municipality[~df_municipality["municipality"].str.endswith("000")]
    # Filter only Seville municipalities
    df_municipality = df_municipality[df_municipality["municipality"].str.startswith("41")]

    df_municipality = df_municipality.astype(
        {
        "municipality": "string", 
        "population_total": "int64", 
        "male_total": "int64", 
        "female_total": "int64", 
        "drivers_male": "int64", 
        "drivers_female": "int64", 
        "drivers_total": "int64"
        }
    )


    df_municipality = df_municipality[["municipality", "drivers_male", "drivers_female"]]
    df_municipality = df_municipality.melt(id_vars=['municipality'], value_vars=['drivers_male', 'drivers_female'],
                  var_name='sex', value_name='weight')
    df_municipality['sex'] = df_municipality['sex'].map(
        {'drivers_male': 'male', 'drivers_female': 'female'}
        ).astype('category')
    # ---------------------------------------------------------------------------------
    # Importing license data per province by age and sex
    FILE_PATH = "{}/{}".format(context.config("data_path"),context.config("seville.licenses_path2"))
    print(f"Loading licenses data from {FILE_PATH}")

    df_province = pd.read_csv(FILE_PATH, sep="|", 
                              usecols=['COD_PROVINCIA', 'IND_SEXO', 'EDAD', 'NUM_LICENCIAS_PERMISOS'], 
                              dtype = {"COD_PROVINCIA":str},
                              )
    colnames = [
        "province", 
        "sex", 
        "age", 
        "relative_weight",
    ]
    df_province.columns = colnames
    
    df_province = df_province[df_province["province"] == "41"]
    df_province['province'] = df_province['province'].astype('category')

    df_province['sex'] = df_province['sex'].map({'M': 'male', 'V': 'female'})
    df_province['sex'] = df_province['sex'].astype("category")

    df_province = df_province[df_province["age"]!= "Se desconoce"] # remove rows with unknown age, these are insignificant
    condition = df_province["age"].str.startswith("Mas")
    df_province.loc[condition, "age"] = 74 # extracts upper bound from "More than 74 years"
    df_province.loc[~condition, "age"] = df_province.loc[~condition, "age"].str[0:2] 
    df_province["age"] = df_province["age"].astype("int64")

    # Weight
    df_province["relative_weight"] = df_province["relative_weight"] / df_province["relative_weight"].sum()


    return df_province, df_municipality
