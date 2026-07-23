"""
Generates the IRIS zoning system that is not used in Paraguay. Instead, we create one
fake IRIS for each corresponding administrative to commune in Paraguay. See the `codes` stage for more information.
"""

def configure(context):
    context.stage("asuncion.data.spatial.raw")

def execute(context):

     # Load codes
    df_codes = context.stage("asuncion.data.spatial.raw")

    # no region id
    df_codes["region_id"] = "Paraguay"
    # departement -> department
    df_codes["departement_id"] = df_codes["departement_id"].astype("category")
    # borough -> commune
    df_codes["commune_id"] = df_codes["borough"].astype("category")

    # Fake IRIS
    df_codes["iris_id"] = df_codes["commune_id"].astype(str) + "0000"
    df_codes["iris_id"] = df_codes["iris_id"].astype("category")


    return df_codes[["region_id", "departement_id", "commune_id", "iris_id", "geometry"]]
