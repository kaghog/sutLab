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
    population_df = pd.read_csv(FILE_PATH, sep="\t")
    colnames = ["province", "municipality", "census_section", "sex", "age", "year", "weight"]
    population_df.columns = colnames

    # Filter
    population_df = population_df[population_df["year"] == 2022]
    population_df = population_df[population_df["province"].str.startswith("41")]
    population_df = population_df[population_df["census_section"] != ""]
    population_df = population_df[population_df["sex"] != "Total"]
    population_df = population_df[population_df["age"] != "All ages"]

    # Clean
    population_df["municipality"] = population_df["municipality"].str[:5].astype('category')
    population_df["census_section"] = population_df["census_section"].str[:10].astype('category')
    population_df.dropna(inplace=True)
    population_df["weight"] = population_df["weight"].astype("int64")
    population_df["province"] = population_df["province"].str[:2].astype('category')
    population_df["sex"] = population_df["sex"].astype('category')


    # age-group column cleanup
    population_df = population_df[~(population_df["age"].str.startswith("16"))] # remove "16 and more years" age range
    condition = population_df["age"].str.startswith("From")
    population_df.loc[condition, "age"] = population_df.loc[condition, "age"].str[5:7] # extracts lower bound from "From 16 to 19 years" like string
    condition = population_df["age"].str.startswith("100") 
    population_df.loc[condition, "age"] = 100 # sets category 70 and more years
    population_df["age"] = population_df["age"].astype("int64")

    return population_df