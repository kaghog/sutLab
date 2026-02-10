"""
The codes are hierarchichally structred as follows:

- 2 digits: Province (departement)
- 3 digits: Municipality (municipality)
- 5 digits: Census section (commune)

The correspondance to the code initially developed for France is as follows:

- '' -> région (is not used, empty string value)
- province -> département
- municipality -> (municipality)
- census section -> commune 
- The French statistical unit (IRIS) does not exist

In Seville, the official code starts with: 41
41: Province = Sevilla (Seville)

41091: Seville City code
"""

def configure(context):
    context.stage("seville.data.spatial.iris")

def execute(context):

    # Load codes
    df_codes = context.stage("seville.data.spatial.iris")

    return df_codes[["region_id", "departement_id", "commune_id", "iris_id"]]
