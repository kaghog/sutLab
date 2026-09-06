import pandas as pd
import os

"""
Load household census data
"""

def configure(context):
    context.config("data_path")
    context.config("asuncion.households-auncion", "census/households/Cuadro 2.3 Asunción. Jefatura de hogar según sexo, 2022.xlsx")
    context.config("asuncion.households-central", "census/households/Cuadro 13.3 Dpto. Central.Jefatura de hogar por sexo, según distrito, 2022..xlsx")
    context.config("commune_equivalent")
    context.stage("asuncion.data.codes")
    context.stage("asuncion.data.select_agglomeration")

def execute(context):

    # ========== Load Asuncion population data (by sex, borough) ==========
    # see "asuncion.households-auncion"
    df_households_asuncion = pd.DataFrame(data={"departement":["ASUNCIÓN"], "district":["ASUNCIÓN"], "weight":[133_620]})

    # ========== Load Asuncion population data (by sex, borough) ==========

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.households-central"))
    SHEET_NAME = "Central 13.3"
    SKIP_ROWS = 7
    SKIP_FOOTER = 1
    COLUMNS = [1,2,3,4]
    COLUMN_NAMES = [
        "district", "weight", "male", "female"
    
    ]
    
    print(f"Loading population data from {EXCEL_PATH}")
    df_households_central = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS, skipfooter=SKIP_FOOTER, dtype=str)
    df_households_central = df_households_central.iloc[:,COLUMNS]
    df_households_central.columns = COLUMN_NAMES

    df_households_central = df_households_central[df_households_central['district']!='Total']

    df_households_central["departement"] = 'CENTRAL'

    df_households_central['weight'] = df_households_central['weight'].astype(int)

    # ========== Merge ==========

    df = pd.concat([df_households_asuncion, df_households_central[["departement", "district", "weight"]]])

    assert not df.isna().any().any()

    df["departement"] = df["departement"].str.upper()
    df["district"] = df["district"].str.upper()
    from asuncion.data.codes import normalize_codes
    df = normalize_codes(df,context.stage("asuncion.data.codes"))


    agglomeration_mun = context.stage("asuncion.data.select_agglomeration")
    df = df[df["district_id"].isin(agglomeration_mun['id'])]


    if context.config("commune_equivalent") == "district":
        df = df.rename(columns={"district_id":"commune_id"})




    assert len(df) != 0

    return df[["departement_id", "commune_id", "weight"]]

def validate(context):
    filenames = [
        "asuncion.households-auncion",
        "asuncion.households-central"
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
