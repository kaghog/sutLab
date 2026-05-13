import pandas as pd
import os

"""
Load population data
"""

def configure(context):
    context.config("data_path")
    context.config("seville.population", "population.csv")

    if context.config("seville_census_area_selection") == 'agglomeration':
        context.stage("seville.data.select_agglomeration")

def execute(context):


    # Load data
    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("seville.population"))
    print(f"Loading population data from {FILE_PATH}")
    population_df = pd.read_csv(FILE_PATH, sep="\t", dtype={"Total": str})
    colnames = ["province_id", "municipality_id", "census_section_id", "sex", "age_class", "year", "weight"]
    population_df.columns = colnames

    # Filter
    population_df = population_df[population_df["year"] == 2022]
    population_df = population_df[population_df["province_id"].str.startswith("41")]
    population_df = population_df[population_df["census_section_id"] != ""]
    population_df = population_df[population_df["municipality_id"] != ""]
    population_df = population_df[population_df["sex"] != "Total"]
    population_df = population_df[population_df["age_class"] != "All ages"]
    population_df = population_df.dropna()

    # Clean
    population_df["municipality_id"] = population_df["municipality_id"].str[:5].astype('str')
    population_df["census_section_id"] = population_df["census_section_id"].str[:10].astype('str')
    population_df.dropna(inplace=True)
    population_df["province_id"] = population_df["province_id"].str[:2].astype('str')
    population_df["sex"] = population_df["sex"].replace({ "Males": "male", "Females": "female" }).astype('str')
    population_df["weight"] = population_df["weight"].str.replace('.', '', regex=False)
    population_df = population_df[~population_df["weight"].isna()]
    population_df['weight'] = pd.to_numeric(population_df['weight'], errors='coerce')


    # Normalize age groups
    population_df = population_df[~(population_df["age_class"].str.startswith("16"))] # remove "16 and more years" age range
    condition = population_df["age_class"].str.startswith("From")
    population_df.loc[condition, "age_class"] = population_df.loc[condition, "age_class"].str[5:7] # extracts lower bound from "From 16 to 19 years" like string
    condition = population_df["age_class"].str.startswith("100") 
    population_df.loc[condition, "age_class"] = 100 # sets category 100 and more years
    population_df["age_class"] = population_df["age_class"].astype("int64")

    result_df = population_df

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

    return result_df

def validate(context):
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.population')}"
    if not os.path.exists(CSV_FILE):
        raise RuntimeError(f"Census population data is not available at location {CSV_FILE}")

    return os.path.getsize(CSV_FILE)
