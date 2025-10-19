import pandas as pd
import numpy as np

"""
This stage loads the raw census data () for Seville provided at municipality level.

"""

def configure(context):
    context.config("data_path")
    context.stage("seville.data.spatial.codes")

    context.config("seville.emplo_census_branch", "employment/emplo_census_branch.csv")
    context.config("seville.emplo_census_occupation", "employment/emplo_census_occupation.csv")
    context.config("seville.emplo_census_situation", "employment/emplo_census_situation.csv")

    context.config("seville.emplo_mun_branch", "employment/emplo_mun_branch.csv")
    context.config("seville.emplo_mun_occupation", "employment/emplo_mun_occupation.csv")
    context.config("seville.emplo_mun_situation", "employment/emplo_mun_situation.csv")

    context.config("seville.emplo_province", "employment/emplo_province.csv")


def round_population(section_df):
    # Total population to match

    section_df['rounded_estimate'] = np.round(section_df['population_estimate']).astype(int)
    section_df['fractional_part'] = section_df['population_estimate'] - np.floor(section_df['population_estimate'])

    def adjust_group(group):
        original_total = group['municipality_count'].iloc[0]
        rounded_total = group['rounded_estimate'].sum()

        discrepancy = int(round(original_total - rounded_total))

        if discrepancy != 0:
            # Sort by fractional part descending for positive discrepancy
            # or ascending for negative discrepancy
            group.sort_values(
                by='fractional_part',
                ascending=(discrepancy < 0)
            )

            # Adjust the top `abs(discrepancy)` rows
            adjust_indices = group.head(abs(discrepancy)).index
            group.loc[adjust_indices, 'rounded_estimate'] += np.sign(discrepancy)

        return group

    # Correct discrepancies for each group per group (municipality, age, sex)
    grouped = section_df.groupby(['municipality', 'age', 'sex'])
    adjusted_df = grouped.apply(adjust_group).reset_index(drop=True)

    # ======== Check if all matches =========
    rounded_total = adjusted_df['rounded_estimate'].sum()
    estimated_total = round(adjusted_df['population_estimate'].sum())
    original_total = section_df.drop_duplicates(subset=['age', 'municipality', 'sex'])['municipality_count'].sum()
    # print(f"Rounded total: {rounded_total}")
    # print(f"Estimated total (rounded): {estimated_total}")
    # print(f"Original total (rounded): {original_total}")
    assert rounded_total == original_total, f"Mismatch: {rounded_total} vs {original_total}"


    adjusted_df.drop(columns=['fractional_part'], inplace=True)
    return adjusted_df

def extrapolate_age_group(context, path1, path2, path3):
    """
    #1 load population per census section by sex
    #2 load population per commune with population above 500 people by sex, age
    #3 extrapolate age for census sections
    #4 load population per commune with population below 500 people by sex (only)
    #5 extrapolate age for communes with population below 500 people by sex
    """


    # ========== Load data per municipality + cleanup =============

    FILE_PATH = "{}/{}".format(context.config("data_path"),context.config(path2))
    print(f"Loading census data from {FILE_PATH}")
    mun_df = pd.read_csv(FILE_PATH, sep="\t", dtype={"Total": str})
    colnames = ["municipality", "age", "sex", "shared_variable", "year", "count"]
    mun_df.columns = colnames
    mun_df = mun_df[mun_df["municipality"].str.startswith("41")]
    mun_df["municipality"] = mun_df["municipality"].str[:5]
    mun_df.dropna()
    
    mun_df["count"] = mun_df["count"].str.replace('.', '', regex=False).astype("int64")
    mun_df = mun_df[mun_df["sex"] != "Total"]
    mun_df = mun_df[mun_df['year'] == 2022]

    # age-group column cleanup
    mun_df = mun_df[~(mun_df["age"].str.startswith("16"))] # remove "16 and more years" age range
    condition = mun_df["age"].str.startswith("From")
    mun_df.loc[condition, "age"] = mun_df.loc[condition, "age"].str[5:7] # extracts lower bound from "From 16 to 19 years"
    condition = mun_df["age"].str.startswith("70") 
    mun_df.loc[condition, "age"] = 70 # sets category 70 and more years
    mun_df["age"] = mun_df["age"].astype("int64")

    # filtering rows that contain "total" which is just sum of the other rows
    # using "otal" to be variable agnostic ("Total" / "CNAE total") 
    mun_df = mun_df[~(mun_df["shared_variable"].str.contains("otal"))]


    # ========== Load data per section + cleanup =============

    FILE_PATH = "{}/{}".format(context.config("data_path"),context.config(path1))
    print(f"Loading census data from {FILE_PATH}")
    census_df = pd.read_csv(FILE_PATH, sep="\t", dtype={"Total": str})
    colnames = ["province", "municipality", "census_section", "sex", "shared_variable", "year", "count"]
    census_df.columns = colnames
    census_df = census_df[census_df["province"].str.startswith("41")]
    census_df = census_df.drop("province", axis=1)
    census_df["municipality"] = census_df["municipality"].str[:5]
    census_df["census_section"] = census_df["census_section"].str[:10]
    census_df["count"] = census_df["count"].str.replace('.', '', regex=False)
    census_df['count'] = pd.to_numeric(census_df['count'], errors='coerce')
    census_df = census_df[census_df['census_section'].notna()]
    census_df = census_df[census_df["sex"]!="Total"]

    census_df = census_df[~(census_df["shared_variable"].str.contains("otal"))]
    census_df = census_df[census_df['year'] == 2022]


    # check if we have data for all the census sections and if not throw error
    codes_df = context.stage("seville.data.spatial.codes")
    missing_sections = codes_df[~codes_df['census_section_code'].isin(census_df['census_section'])]
    if not missing_sections.empty:
        print("Missing values from df1 in df2:", missing_sections.count())
        print("Missing values from df1 in df2:", codes_df.count())
        assert missing_sections.empty


    # check if all census sections have their own municipality
    # reason for missing municipalities can be that municipalities under 500 are in different dataset
    # the dataset for municipalities under 500 are only by sex (same as census section), so no need to load them as well
    condition = ~census_df['municipality'].isin(mun_df['municipality'])
    missing_sections = census_df[condition]
    census_df = census_df[~condition]

    # check if the population count matches
    assert mun_df['count'].sum() == census_df['count'].sum()

    # ================= Municipalities under 500 people =================

    # NOTE: Shared variable cannot be easily use, because the naming does not match for census_section dataset and province dataset
    #       - mapping could be implemented, but is probably not worth it for the small payoff
    
    # province total df
    FILE_PATH = "{}/{}".format(context.config("data_path"),context.config(path3))
    print(f"Loading census data from {FILE_PATH}")
    province_df = pd.read_csv(FILE_PATH, sep="\t", dtype={"Total": str})
    # National Total;	Autonomous Communities and Cities;	Provinces;	Employment;	Age;	Sex;	Periodo;	Total;
    province_df = province_df.iloc[:, [2,3,4,5,6,7]]
    colnames = ["province", "shared_variable", "age", "sex", "year", "count"]
    province_df.columns = colnames

    province_df = province_df[province_df["year"] == 2022]
    province_df = province_df.dropna()
    province_df = province_df[province_df["province"].str.startswith("41")]
    province_df = province_df[province_df["sex"]!="Total"]
    province_df["count"] = province_df["count"].str.replace('.', '', regex=False).astype("int64")

    province_df = province_df[~(province_df["shared_variable"].str.contains("otal"))]

    # age-group column cleanup
    province_df = province_df[~(province_df["age"].str.startswith("16"))] # remove "16 and more years" age range
    condition = province_df["age"].str.startswith("From")
    province_df.loc[condition, "age"] = province_df.loc[condition, "age"].str[5:7] # extracts lower bound from "From 16 to 19 years"
    condition = province_df["age"].str.startswith("70") 
    province_df.loc[condition, "age"] = 70 # sets category 70 and more years
    province_df["age"] = province_df["age"].astype("int64")


    # ========== Extrapolate age distribution for census sections in municipalities under 500 ==========

    province_df = province_df[['sex', 'age', 'count']]
    group_cols = ['sex', 'age']
    municipalities500_total_df = mun_df.groupby(group_cols)['count'].sum()
    province_df = province_df.groupby(group_cols)['count'].sum()

    municipalities50_df = (province_df - municipalities500_total_df).reset_index()
    municipalities50_df['total'] = municipalities50_df.groupby(['sex'])['count'].transform("sum")
    municipalities50_df['proportion'] = municipalities50_df['count'] / municipalities50_df['total'] # age group distribution in given sex
    municipalities50_df['proportion'] = municipalities50_df['proportion'].replace(np.nan, 0)
    municipalities50_df['municipality_count'] = municipalities50_df['count']

    missing_sections = missing_sections.groupby(['municipality', 'census_section', 'sex'])['count'].sum().reset_index()
    missing_sections = pd.merge(missing_sections, municipalities50_df[['sex', 'age', 'proportion', 'municipality_count']], on=['sex'], how='left')
    missing_sections['population_estimate'] = missing_sections['count'] * missing_sections['proportion']
    

    # ========== Extrapolate age distribution for census sections in municipalities above 500 people ==========

    # get a distribution range for the age_groups
    group_cols = ['municipality', 'sex', 'shared_variable']
    mun_df['total'] = mun_df.groupby(group_cols)['count'].transform('sum')
    mun_df['proportion'] = mun_df['count'] / mun_df['total']
    mun_df['proportion'] = mun_df['proportion'].replace(np.nan, 0)
    mun_df['municipality_count'] = mun_df['count']

    expanded = pd.merge(census_df, mun_df[group_cols + ['age' , 'proportion', 'municipality_count']], on=group_cols, how='left')
    expanded['population_estimate'] = expanded['count'] * expanded['proportion']
    group_cols = ['municipality', 'census_section', 'sex', 'age',]
    expanded = expanded.groupby(group_cols)[['municipality_count', 'population_estimate', 'proportion']].sum().reset_index()


    # ========= Merge =========

    # merge results for census sections in municipalities above 500 and below 500
    result_df =  pd.concat([expanded, missing_sections], ignore_index=True)

    return result_df[['municipality', 'census_section', 'sex', 'age', 'population_estimate', 'municipality_count']]


def execute(context):


    # Extrapolate age sex groups for census sections by seperate variables: branch, occupation, situation 
    census_age_sex_df1 = extrapolate_age_group(context, "seville.emplo_census_branch", "seville.emplo_mun_branch", "seville.emplo_province")
    census_age_sex_df2 = extrapolate_age_group(context, "seville.emplo_census_occupation", "seville.emplo_mun_occupation", "seville.emplo_province")
    census_age_sex_df3 = extrapolate_age_group(context, "seville.emplo_census_situation", "seville.emplo_mun_situation", "seville.emplo_province")

    # Average results
    averaged_df = census_age_sex_df1.copy()
    averaged_df["population_estimate"] = ( 
        census_age_sex_df1["population_estimate"] 
        + census_age_sex_df2["population_estimate"] 
        + census_age_sex_df3["population_estimate"]
        ) / 3

    # Adjust results to have population count in whole numbers without fractions of people
    rouned_df = round_population(averaged_df)


    # Simple check for data validity
    # TODO: Make the check process better:)
    FILE_PATH = "{}/{}".format(context.config("data_path"),context.config("seville.emplo_census_branch"))
    print(f"Loading census data from {FILE_PATH}")
    province_total_row = pd.read_csv(FILE_PATH, sep="\t", dtype={"Total": str}, nrows=1)
    province_total_row["Total"] = province_total_row["Total"].str.replace('.', '', regex=False).astype("int64")
    
    total = province_total_row['Total'][0]
    total_estimate = rouned_df['rounded_estimate'].sum()
    if total_estimate != total:
        raise ValueError(f"Values do not match: Expected total count: {total}; Actual total count: {total_estimate}")



    return rouned_df