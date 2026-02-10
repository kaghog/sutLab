"""
Generates the IRIS zoning system that is not used in Spain. Instead, we create one
fake IRIS for each census section in Spain. See the `codes` stage for more information.
"""

def configure(context):
    context.stage("seville.data.spatial.raw")

def execute(context):

     # Load codes
    df_codes = context.stage("seville.data.spatial.raw")

    # Clean up identifiers
    df_codes["census_section_id"] = df_codes["census_section_id"].astype(str)
    # no region id
    df_codes["region_id"] = df_codes["census_section_id"].str[:0].astype("category")
    # province -> department
    df_codes["departement_id"] = df_codes["census_section_id"].str[:2].astype("category")
    # Census_section -> commune
    df_codes["commune_id"] = df_codes["census_section_id"].astype("category")

    # Fake IRIS
    df_codes["iris_id"] = df_codes["commune_id"].astype(str) + "0000"
    df_codes["iris_id"] = df_codes["iris_id"].astype("category")


    return df_codes[["region_id", "departement_id", "commune_id", "iris_id", "geometry"]]
