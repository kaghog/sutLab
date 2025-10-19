"""
Generates the IRIS zoning system that is not used in Spain. Instead, we create one
fake IRIS for each municipality in Spain. See the `codes` stage for more information.
"""

def configure(context):
    context.stage("seville.data.spatial.raw")

def execute(context):

    # TODO: IT IS CURRENTLY NOT COMPATIBLE RAW.PY !!!!!!!!!!
    raise NotImplemented("Iris not implemented")
    # CHECK FORMAT FOR IRIS AND HOW TO FAKE IT

    # Load shapes
    df = context.stage("seville.data.spatial.raw")[["census_section_code", "geometry"]]

    # Clean up identifiers
    df["commune_id"] = ("03241" + df["census_section_code"].astype(str)).astype("category")

    # Fake IRIS
    df["iris_id"] = df["commune_id"].astype(str) + "0000"
    df["iris_id"] = df["iris_id"].astype("category")

    # Departement identifiers
    df["departement_id"] = df["commune_id"].str[:5]

    # Region dummu
    df["region_id"] = 1
    df["region_id"] = df["region_id"].astype("category")


    return df[["iris_id", "commune_id", "departement_id", "geometry"]]
