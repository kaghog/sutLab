"""
Generates the IRIS zoning system that is not used in Paraguay. Instead, we create one
fake IRIS for each corresponding administrative to commune in Paraguay. See the `codes` stage for more information.
"""

def configure(context):
    context.stage("asuncion.data.spatial.raw")
    context.config("commune_equivalent")
    context.stage("asuncion.data.codes")
    context.config("pipeline_crs")
def execute(context):

     # Load codes
    df_codes = context.stage("asuncion.data.spatial.raw")

    df_codes = df_codes.to_crs(context.config("pipeline_crs"))

    # no region id
    df_codes["region_id"] = "Paraguay"
    # departement -> department
    df_codes["departement"] = df_codes["departement"].astype("category")


    df_codes["departement"] = df_codes["departement"].str.upper()
    df_codes["district"] = df_codes["district"].str.upper()
    from asuncion.data.codes import normalize_codes
    df_codes = normalize_codes(df_codes, context.stage("asuncion.data.codes"))


    if context.config("commune_equivalent") == "district":
        df_codes["commune_id"] = df_codes["district_id"].astype("category")
    else:
        df_codes["commune_id"] = df_codes["borough_id"].astype("category")

    # Fake IRIS
    df_codes["iris_id"] = df_codes["commune_id"].astype(str) + "0000"
    df_codes["iris_id"] = df_codes["iris_id"].astype("category")


    return df_codes[["region_id", "departement_id", "commune_id", "iris_id", "geometry"]]
