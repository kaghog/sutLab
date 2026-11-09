import pandas as pd
import os
import numpy as np

"""
Load population data
"""

def configure(context):
    context.config("data_path")
    context.config("seville.population", "population.csv")

    if context.config("seville_census_area_selection") == 'agglomeration':
        context.stage("seville.data.select_agglomeration")

def parse_population_data(context):
    # Load data
    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("seville.population"))
    print(f"Loading population data from {FILE_PATH} for population growth")
    population_df = pd.read_csv(FILE_PATH, sep="\t", dtype={"Total": str})
    colnames = ["province_id", "municipality_id", "census_section_id", "sex", "age_class", "year", "weight"]
    population_df.columns = colnames


    # Filter
    #population_df = population_df[population_df["year"] == year]
    population_df = population_df[population_df["province_id"].str.startswith("41")]
    population_df = population_df[population_df["census_section_id"] != ""]
    population_df = population_df[population_df["municipality_id"] != ""]
    population_df = population_df[population_df["sex"] != "Total"]
    population_df = population_df[population_df["age_class"] != "All ages"]
    population_df = population_df.drop(columns=["census_section_id"])
    population_df = population_df.dropna()

    # Clean
    population_df["municipality_id"] = population_df["municipality_id"].str[:5].astype('str')
    #population_df["census_section_id"] = population_df["census_section_id"].str[:10].astype('str')
    population_df.dropna(inplace=True)
    population_df["province_id"] = population_df["province_id"].str[:2].astype('str')
    population_df["sex"] = population_df["sex"].replace({ "Males": "male", "Females": "female" }).astype('str')
    population_df["weight"] = population_df["weight"].str.replace('.', '', regex=False)
    population_df = population_df[~population_df["weight"].isna()]
    population_df['weight'] = pd.to_numeric(population_df['weight'], errors='coerce')

    # Clean age groups
    condition = population_df["age_class"].str.startswith("From")
    population_df.loc[condition, "age_class"] = population_df.loc[condition, "age_class"].str[5:7] # extracts lower bound from "From 15 to 19 years" like string
    condition = population_df["age_class"].str.startswith("100") 
    population_df.loc[condition, "age_class"] = 100 # sets category 100 and more years
    population_df["age_class"] = population_df["age_class"].astype("int64")

    return population_df


def population_to_license_age_classes(df):
    """
    Convert 5-year population age bins into driving license age bins.

    Population: 0-4, 5-9, 10-14, ..., 95-99, 100+
    Licenses:   0-14, 15-17, 18-20, 21-24, 25-29, ..., 70-73, 74+

    Assumes uniform population within each 5-year bin.
    """

    df_pop = df.copy()
    df_pop["age_class"] = df_pop["age_class"].astype(int)

    valid_ages = set(range(0, 101, 5))
    invalid = sorted(set(df_pop["age_class"]) - valid_ages)

    if invalid:
        raise ValueError(f"Unexpected population age classes: {invalid}")

    license_age_classes = [
        0, 15, 18, 21, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 75
    ]

    start = df_pop["age_class"].to_numpy()
    parts = []

    for lower, upper in zip(
        license_age_classes,
        license_age_classes[1:] + [np.inf]
    ):
        # Fraction of the 5-year population bin in this license bin
        fraction = np.clip(
            np.minimum(start + 5, upper) - np.maximum(start, lower),
            0, 5
        ) / 5

        # The 100+ category belongs entirely to the last license class
        fraction[start == 100] = 1.0 if lower == 75 else 0.0

        selected = fraction > 0

        if selected.any():
            part = df_pop.loc[selected].copy()
            part["age_class"] = lower
            part["weight"] = (
                part["weight"].to_numpy() * fraction[selected]
            )
            parts.append(part)

    return pd.concat(parts, ignore_index=True)

def add_scaling_factor_between_years(df, df_population_growth, year_from, year_to, matching_attributes=['municipality_id']):
    """
    Scale a dataset from year_from to year_to using population growth rates.
    """
    df = df.copy()
    df_population_growth = df_population_growth.copy()
    df_population_growth = df_population_growth[
        (df_population_growth['year'] == year_to)
        | (df_population_growth['year'] == year_from)
    ]
    df_population_growth = df_population_growth.groupby(matching_attributes+['year'], as_index=False)['weight'].sum()
    
    df_population_growth = pd.pivot(df_population_growth, 
                                    index=matching_attributes, 
                                    columns='year', 
                                    values='weight').reset_index()

    df_population_growth["scaling_factor"] = df_population_growth[year_to] / df_population_growth[year_from]
    df_population_growth = df_population_growth[matching_attributes+["scaling_factor"]] 

    # Now apply scaling
    df = df.merge(df_population_growth, on=matching_attributes, how="left")
    assert df["scaling_factor"].notnull().all(), "Scaling factor is null for some rows, check matching attributes"

    return df




def execute(context):

    result_df = parse_population_data(context)

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
