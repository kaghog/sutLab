import pandas as pd

"""
This stage extracts household-level census data for Seville.
Provides household size distribution at census section (commune) level for IPU constraints.
"""


def configure(context):
    context.stage("seville.data.spatial.codes")
    context.config("data_path")
    context.config("seville.household_data", "households.xlsx")

    context.config("seville_city_data_only")


def execute(context):

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("seville.household_data"))
    SHEET_NAME = "tabla-59543"

    print(f"Loading licenses data from {EXCEL_PATH}")
    df_households = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=8)
    household_cols = [
        "municipality_id",
        "total_households",
        "households_1_person",
        "households_2_persons",
        "households_3_persons",
        "households_4_persons",
        "households_5plus_persons",
    ]
    df_households.columns = household_cols
    
    # Clean
    df_households["municipality_id"] = df_households["municipality_id"].astype(str)
    df_households["municipality_id"] = df_households["municipality_id"].str[:5]
    df_households["departement_id"] = df_households["municipality_id"].str[:2]

    # Filter
    df_households = df_households[df_households["departement_id"] == "41"]

    assert len(df_households) != 0

    print(df_households.head())

    if context.config("seville_city_data_only") == True:
        df_households = df_households[df_households["municipality_id"] == "41091"]
    
    return df_households

