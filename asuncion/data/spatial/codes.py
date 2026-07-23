
def configure(context):
    context.stage("asuncion.data.spatial.iris")

def execute(context):

    # Load codes
    df_codes = context.stage("asuncion.data.spatial.iris")

    return df_codes[["region_id", "departement_id", "commune_id", "iris_id"]]
