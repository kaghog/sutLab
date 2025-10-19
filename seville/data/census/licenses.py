
import pandas as pd

"""
This stage loads the driving license data for Seville provided at municipality level.

"""

def configure(context):
    context.config("data_path")
    context.config("seville.licenses_path", "licenses_2024.xlsx")

def execute(context):
    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("seville.licenses_path"))
    print(f"Loading licenses data from {EXCEL_PATH}")
    SHEET_NAME = "DatosMunicipalesGeneral_2024"

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME,)
    
    df = df.iloc[:,[0,4,5,6,7,8,9]]
    colnames = [
        "municipality_code", 
        "population_total", 
        "male_total", 
        "female_total", 
        "drivers_male", 
        "drivers_female", 
        "drivers_total"
    ]

    df.columns = colnames

    # Filter only Seville municipalities
    # df = df[df["municipality_code"].str.startswith("41")]
    df = df[df["municipality_code"] // 1000 == 41]

    # Remove "municipio sin especificar" ("nonspecified municipality") row, that is not significance and is present for each province    
    # TODO: alternatively we can equally distribute it
    # df = df[df["municipality_code"].str.endswith("000")]
    df = df[df["municipality_code"] % 1000 != 0]

    df = df.astype(
        {
        "municipality_code": "int64", 
        "population_total": "int64", 
        "male_total": "int64", 
        "female_total": "int64", 
        "drivers_male": "int64", 
        "drivers_female": "int64", 
        "drivers_total": "int64"
        }
    )


    print(df.head())
    print(df.info())
    print(df.describe())

    return df
