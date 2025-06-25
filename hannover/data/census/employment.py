import pandas as pd
import numpy as np
import os

"""
This stage loads the raw employment data for Bavaria

TODO: Can this be replaced by a Germany-wide extract from GENESIS?
"""

def configure(context):
    context.stage("hannover.data.spatial.codes")
    context.stage("hannover.data.census.population")

    context.config("data_path")
    context.config("hannover.employment_path", "hannover/13111-004r.xlsx")

def execute(context):
    # Load the Bavarian employment data from bavarian script -----------------------------------
    # Load data
    df_employment = pd.read_excel("{}/{}".format(
        context.config("data_path"), context.config("hannover.employment_path")
    ), skiprows = 6, names = [
        "departement_id", "department_name", "age_class", 
        "all_total", "all_male", "all_female", 
        "national_all", "national_male", "national_female",
        "foreign_all", "foreign_male", "foreign_female"
    ])

    # Remove text at the end
    index = np.argmax(df_employment["departement_id"] == "______________")
    df_employment = df_employment.iloc[:index]

    # Filter for full Kreis entries
    df_employment["departement_id"] = df_employment["departement_id"].ffill()

    df_employment = df_employment[
        df_employment["departement_id"].str.len() == 5
    ].copy()

    # Remove totals
    df_employment = df_employment[
        df_employment["age_class"] != "Insgesamt"
    ].copy()

    # Format age class
    df_employment.loc[df_employment["age_class"] == "unter 20", "age_class"] = "0"
    df_employment["age_class"] = df_employment["age_class"].str[:2]
    df_employment = df_employment[df_employment["age_class"].astype(str).str.match(r"^\d+")].copy()
    df_employment["age_class"] = df_employment["age_class"].astype(int)

    # Format data frame
    df_employment = df_employment[[
        "departement_id", "age_class", "all_male", "all_female"
    ]]

    # Bring into long format
    df_employment = pd.melt(df_employment, 
        ["departement_id", "age_class"], ["all_male", "all_female"], 
        var_name = "sex", value_name = "weight")
    
    # Format sex
    df_employment["sex"] = df_employment["sex"].str[4:].astype("category")
    
    # ---------------------------------------------------------- -----------------------------------

    # Filter Munich employment
    df_munich = df_employment[df_employment["departement_id"] == "09162"].copy()

    # Define age ranges
    hannover_bins = {
        0: (0, 5),
        6: (6, 14),
        15: (15, 17),
        18: (18, 23),
        24: (24, 29),
        30: (30, 44),
        45: (45, 64),
        65: (65, 79),
        80: (80, 100)
    }

    munich_bins = {
        0: (0, 19),
        20: (20, 24),
        25: (25, 29),
        30: (30, 49),
        50: (50, 59),
        60: (60, 64),
        65: (65, 100)
    }

    # Function to calculate overlap
    def calculate_overlap(start1, end1, start2, end2):
        return max(0, min(end1, end2) - max(start1, start2) + 1)

    # Build redistribution matrix, redistribute data from Munich bins to Hannover bins proportionally, based on how much they overlap
    redistribution = []
    for m_age, (m_start, m_end) in munich_bins.items():
        total_munich_range = m_end - m_start + 1
        for h_age, (h_start, h_end) in hannover_bins.items():
            overlap = calculate_overlap(m_start, m_end, h_start, h_end)
            if overlap > 0:
                redistribution.append({
                    "munich_age_class": m_age,
                    "hannover_age_class": h_age,
                    "proportion": overlap / total_munich_range
                })
    df_redistribution = pd.DataFrame(redistribution)

    # Merge Munich employment data with redistribution
    df_munich = df_munich.rename(columns={"age_class": "munich_age_class"})
    df_mapped = pd.merge(df_redistribution, df_munich, on="munich_age_class", how="left")

    # Compute redistributed weight for each Hannover age class
    df_mapped["redistributed_weight"] = df_mapped["weight"] * df_mapped["proportion"]
    # Aggregate redistributed employment to Hannover age classes and sex
    df_distribution = df_mapped.groupby(["hannover_age_class", "sex"], as_index=False, observed=True)["redistributed_weight"].sum()
    df_distribution = df_distribution.rename(columns={
        "hannover_age_class": "age_class",
        "redistributed_weight": "raw_employment_weight"
    })

    # Normalize it
    total_by_sex = df_distribution.groupby("sex", observed=True)["raw_employment_weight"].transform("sum")
    df_distribution["employment_ratio"] = df_distribution["raw_employment_weight"] / total_by_sex
    df_distribution = df_distribution.drop(columns=["raw_employment_weight"])

    # Merge with Hannover population and compute estimated employment
    df_hannover = context.stage("hannover.data.census.population")
    df_hannover["departement_id"] = df_hannover["commune_id"].str[:5]
    df_hannover_agg = df_hannover.groupby(["departement_id", "age_class", "sex"], as_index=False, observed=True)["weight"].sum()
    df_merged = pd.merge(df_hannover_agg, df_distribution, on=["age_class", "sex"], how="left")
    df_merged["weight"] = (df_merged["weight"] * df_merged["employment_ratio"]).round()

    # Formatting the dataFrame
    df_merged = df_merged.dropna(subset=["weight"])
    df_merged["weight"] = df_merged["weight"].astype(int)
    df_merged["sex"] = df_merged["sex"].astype("category")
    df_employment = df_merged[["departement_id", "age_class", "sex", "weight"]]
    return df_employment


def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("hannover.employment_path"))):
        raise RuntimeError("Bavarian employment data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("hannover.employment_path")))
