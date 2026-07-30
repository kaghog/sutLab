import pandas as pd
import os

"""
Load household census data
"""

def configure(context):
    context.config("data_path")
    
    context.config("asuncion.employment-sex", "census/employment/1_Empleo_ocupación informal según área y sexo_py_EPHC_2022-2025.xls")
    context.config("asuncion.employment-age", "census/employment/2_Empleo_ocupación informal según grupos de edad_py_EPHC_2022-2025.xls")
    context.config("asuncion.employment-departements", "census/employment/11_Empleo_ocupación informal según departamento_dpto_EPHC 2022-2025.xls")

    context.stage("asuncion.data.census.population")

    context.config("commune_equivalent")

def execute(context):


    # ========== Load National employment data (1) (by sex) ==========

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.employment-sex"))
    SHEET_NAME = "Área de residencia y sexo"
    SKIP_ROWS = 6
    SKIP_FOOTER = 8
    COLUMNS = [1, 11, 12]
    COLUMN_NAMES = [
        "sex", "weight_total", "weight_informal"
    ]
    
    print(f"Loading population data from {EXCEL_PATH}")
    df_sex = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS, skipfooter=SKIP_FOOTER, dtype=str)
    df_sex = df_sex.iloc[:,COLUMNS]
    df_sex.columns = COLUMN_NAMES

    df_sex['weight_total'] = df_sex['weight_total'].astype(int)
    df_sex['weight_informal'] = df_sex['weight_informal'].astype(int)
    df_sex['weight_formal'] = df_sex['weight_total'] - df_sex['weight_informal']

    SEX_MAP = {"Hombres": "male", "Mujeres": "female"}
    df_sex['sex'] = df_sex['sex'].map(SEX_MAP)

    # ========== Load National employment data (2) (by age) ==========

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.employment-age"))
    SHEET_NAME = "Grupos de edad"    
    SKIP_ROWS = 6
    SKIP_FOOTER = 2
    COLUMNS = [1, 11, 12]
    COLUMN_NAMES = [
        "age_class", "weight_total", "weight_informal"
    ]
    
    print(f"Loading population data from {EXCEL_PATH}")
    df_age = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS, skipfooter=SKIP_FOOTER, dtype=str)
    df_age = df_age.iloc[:,COLUMNS]
    df_age.columns = COLUMN_NAMES

    df_age['weight_total'] = df_age['weight_total'].astype(int)
    df_age['weight_informal'] = df_age['weight_informal'].astype(int)
    df_age['weight_formal'] = df_age['weight_total'] - df_age['weight_informal']

    df_age['age_class'] = df_age['age_class'].str[:2]

    # ========== Load departement data (by departement) ==========

    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.employment-departements"))
    SHEET_NAME = "Ocupados informales"    
    SKIP_ROWS = 6
    SKIP_FOOTER = 2
    COLUMNS = [1, 11, 12]
    COLUMN_NAMES = [
        "departement_id", "weight_total", "weight_informal"
    ]
    
    print(f"Loading population data from {EXCEL_PATH}")
    df_departements = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS, skipfooter=SKIP_FOOTER, dtype=str)
    df_departements = df_departements.iloc[:,COLUMNS]
    df_departements.columns = COLUMN_NAMES

    df_departements['weight_total'] = df_departements['weight_total'].astype(int)
    df_departements['weight_informal'] = df_departements['weight_informal'].astype(int)
    df_departements['weight_formal'] = df_departements['weight_total'] - df_departements['weight_informal']


    # ========== Merge ==========

    # we choose only Asuncion and Central departements
    df_departements = df_departements[df_departements["departement_id"].isin(('Asunción', 'Central'))] 


    # datasets contains columns for 1) formal employment 2) informal employment 
    formal = []

    for is_formal in [True, False]:

        sex = (
            df_sex
            .assign(weight=lambda x: x["weight_formal"] if is_formal else x["weight_informal"])
            [["sex", "weight"]]
        )
        sex["p"] = sex.weight / sex.weight.sum()

        age = (
            df_age
            .assign(weight=lambda x: x["weight_formal"] if is_formal else x["weight_informal"])
            [["age_class", "weight"]]
        )
        age["p"] = age.weight / age.weight.sum()

        dept = (
            df_departements
            .assign(weight=lambda x: x["weight_formal"] if is_formal else x["weight_informal"])
            [["departement_id", "weight"]]
        )
        dept["p"] = dept.weight / dept.weight.sum()

        tmp = (
            dept.assign(key=1)
                .merge(sex.assign(key=1), on="key")
                .merge(age.assign(key=1), on="key")
                .drop(columns="key")
        )

        total = dept.weight.sum()

        tmp["weight"] = total * tmp.p_x * tmp.p_y * tmp.p

        tmp["is_formal"] = is_formal

        formal.append(tmp[["departement_id","sex","age_class","is_formal","weight"]])

    df = pd.concat(formal, ignore_index=True)




    df = df.groupby(["departement_id","sex","age_class"])["weight"].sum().reset_index()
    df["age_class"] = df["age_class"].astype(int)
    df["departement_id"] = df["departement_id"].str[0]

    assert not df.isna().any().any()


    # ============= Recalculate employment to district level ==============



    pop = context.stage("asuncion.data.census.population").copy()
    pop = pop.groupby(["departement_id", "district", "sex", "age_class"])["weight"].sum().reset_index()

    pop["pop_share"] = (
        pop["weight"]
            / pop.groupby(["departement_id", "sex", "age_class"])["weight"]
            .transform("sum")
    )

    # Allocate employment to districts
    df_district = (
        df.merge(
            pop[["departement_id", "district", "sex", "age_class", "pop_share"]],
            on=["departement_id", "sex", "age_class"],
            how="left",
        )
    )

    df_district["weight"] *= df_district["pop_share"]

    df_district = df_district[
        ["departement_id", "district", "sex", "age_class", "weight"]
    ]

    if context.config("commune_equivalent") == "district":
        df_district = df_district.rename(columns={"district":"commune_id"})

    return df_district[["departement_id", "commune_id", "sex", "age_class", "weight"]]

def validate(context):
    filenames = [
        "asuncion.employment-sex",
        "asuncion.employment-age",
        "asuncion.employment-departements"
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
