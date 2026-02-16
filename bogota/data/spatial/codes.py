"""

- The French statistical unit (IRIS) does not exist
"""

def configure(context):
    context.stage("bogota.data.spatial.iris")

def execute(context):

    # Load codes
    df_codes = context.stage("bogota.data.spatial.iris")

    return df_codes[["region_id", "departement_id", "commune_id", "iris_id"]]
