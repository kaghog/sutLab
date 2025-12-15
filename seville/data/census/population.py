import pandas as pd

"""
Load population data
"""

def configure(context):
    context.config("data_path")
    context.config("seville.population", "population.csv")

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


    # age-group column cleanup
    population_df = population_df[~(population_df["age_class"].str.startswith("16"))] # remove "16 and more years" age range
    condition = population_df["age_class"].str.startswith("From")
    population_df.loc[condition, "age_class"] = population_df.loc[condition, "age_class"].str[5:7] # extracts lower bound from "From 16 to 19 years" like string
    condition = population_df["age_class"].str.startswith("100") 
    population_df.loc[condition, "age_class"] = 100 # sets category 100 and more years
    population_df["age_class"] = population_df["age_class"].astype("int64")

    return population_df