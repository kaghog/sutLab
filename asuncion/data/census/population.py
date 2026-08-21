import pandas as pd
import os

"""
Load population data
"""

def configure(context):
    context.config("data_path")
    context.config("asuncion.population-ad-sex-bario", "census/population/Cuadro 1.1 Asunción. Población por sexo y edad mediana, según barrio, 2022.xlsx")
    context.config("asuncion.population-ad-sex-age", "census/population/Cuadro 3. Asunción. Población total por área urbana-rural y sexo, según grupos de edad, 2022..xlsx")
    context.config("asuncion.population-cd-sex-age-district", "census/population/Cuadro 14. Departamento Central. Población total por área urbana-rural y sexo, según distrito y grupos de edad, 2022..xlsx")
    context.config("commune_equivalent")
    context.stage("asuncion.data.codes")
def execute(context):

    # ========== Load ASUNCIÓN population data (by sex, borough) ==========

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.population-ad-sex-bario"))
    SHEET_NAME = "C1.1_ASUNCIÓN"
    SKIP_ROWS = 7
    SKIP_FOOTER = 1
    COLUMNS = [1,2,3,4,5]
    COLUMN_NAMES = [
        "borough", "total", "male", "female", "age median"
    
    ]
    
    print(f"Loading population data from {EXCEL_PATH}")
    df_asuncion_barrios_sex = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS, skipfooter=SKIP_FOOTER, dtype=str)
    df_asuncion_barrios_sex = df_asuncion_barrios_sex.iloc[:,COLUMNS]
    df_asuncion_barrios_sex.columns = COLUMN_NAMES

    df_asuncion_barrios_sex = df_asuncion_barrios_sex[df_asuncion_barrios_sex['borough']!='Total']

    df_asuncion_barrios_sex = df_asuncion_barrios_sex.melt(
        id_vars=["borough"],                    # columns to keep
        value_vars=["male", "female"],
        var_name="sex",
        value_name="weight"
    )
    df_asuncion_barrios_sex["departement"] = 'ASUNCIÓN'
    df_asuncion_barrios_sex['district'] = 'ASUNCIÓN'

    df_asuncion_barrios_sex['weight'] = df_asuncion_barrios_sex['weight'].astype(int)

    # ========== Load ASUNCIÓN population data (by sex, age) ==========

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.population-ad-sex-age"))
    SHEET_NAME = "Asunción"
    SKIP_ROWS = 7
    SKIP_FOOTER = 1
    COLUMNS = [1,2,3,4]
    COLUMN_NAMES = [
        "age_class", "total", "male", "female"
    
    ]
    print(f"Loading population data from {EXCEL_PATH}")
    df_asuncion_age_sex = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS, skipfooter=SKIP_FOOTER, dtype=str)
    df_asuncion_age_sex = df_asuncion_age_sex.iloc[:,COLUMNS]
    df_asuncion_age_sex.columns = COLUMN_NAMES

    df_asuncion_age_sex = df_asuncion_age_sex[df_asuncion_barrios_sex['borough']!='Total']

    df_asuncion_age_sex = df_asuncion_age_sex.melt(
        id_vars=["age_class"],                    # columns to keep
        value_vars=["male", "female"],
        var_name="sex",
        value_name="weight"
    )
    df_asuncion_age_sex["departement"] = 'ASUNCIÓN'
    df_asuncion_age_sex['district'] = 'ASUNCIÓN'

    df_asuncion_age_sex['age_class'] = (
        df_asuncion_age_sex['age_class']
        .str.strip()
        .str.split()
        .str[0]
    ).astype(int)
    df_asuncion_age_sex['weight'] = df_asuncion_age_sex['weight'].astype(int)
    
    # ========== Load Central departement population data (by sex, age, district) ==========


    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.population-cd-sex-age-district"))
    SHEET_NAME = "Central"
    SKIP_ROWS = 6
    SKIP_FOOTER = 1
    COLUMNS = [1,2,3,4]
    COLUMN_NAMES = [
        "district+age_class", "total", "male", "female"
    
    ]
    print(f"Loading population data from {EXCEL_PATH}")
    df_central = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS, skipfooter=SKIP_FOOTER, dtype=str)
    df_central = df_central.iloc[:,COLUMNS]
    df_central.columns = COLUMN_NAMES

    df_central["district+age_class"] = df_central["district+age_class"].str.strip()

    # Age groups start with a number (0-4, 5-9, ...) district names are everything else
    is_age = df_central["district+age_class"].str.match(r"^\d")
    df_central.loc[~is_age, "district"] = df_central.loc[~is_age, "district+age_class"]
    # Fill district down and keep only age rows
    df_central["district"] = df_central["district"].ffill()
    df_central = df_central[is_age].copy()
    # Rename the original column
    df_central = df_central.rename(columns={"district+age_class": "age_class"})

    df_central = df_central[df_central['district'] != "Total"]


    # clean age class format
    df_central['age_class'] = (
        df_central['age_class']
        .str.strip()
        .str.split()
        .str[0]
    ).astype(int)

    # clean sex
    df_central = df_central.melt(
        id_vars=["age_class", "district"],                    # columns to keep
        value_vars=["male", "female"],
        var_name="sex",
        value_name="weight"
    )
    df_central['weight'] = df_central['weight'].astype(int)


    df_central["departement"] = 'Central'

    # ========== Merge all together ========== 

    df_asuncion_age_sex["share"] = (
        df_asuncion_age_sex["weight"]
        / df_asuncion_age_sex.groupby("sex")["weight"].transform("sum")
    )
    df = df_asuncion_barrios_sex.merge(df_asuncion_age_sex[['age_class', 'sex', 'share']], on="sex")
    df["weight"] = (
        df["share"] * df["weight"]
    )
    df = df[["sex", "age_class","weight", "departement", "district", "borough"]]


    df_central['borough'] = df_central['district']
    df = pd.concat([df, df_central])

    df["departement"] = df["departement"]

    assert not df.isna().any().any()



    # ======== rename for export ==========

    df["departement"] = df["departement"].str.upper()
    df["district"] = df["district"].str.upper()
    from asuncion.data.codes import normalize_codes
    df = normalize_codes(df,context.stage("asuncion.data.codes"))

    if context.config("commune_equivalent") == "district":
        df["commune_id"] = df["district_id"]
    else:
        raise NotImplementedError
        df["commune_id"] = df["borough"]




    return df[["sex", "age_class","weight", "departement_id", "district_id", "commune_id"]]

def validate(context):
    filenames = [
        "asuncion.population-ad-sex-bario",
        "asuncion.population-ad-sex-age",
        "asuncion.population-cd-sex-age-district"
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
