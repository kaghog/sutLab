"""
The codes (Amtlichen Regionalschlüssel - ARS)
are hierarchichally structred as follows:

- 2 digits: Bundesland (or city state)
- 1 digit: Regierungsbezirk / Bezirk
- 2 digits: Landkreis or Kreisfreie Stadt (city without "Landkreis / Kreis")
- 4 digits: Gemeindeverband (municipality associations)
- 3 digits: Gemeinde (municipality)

The correspondance to the code initially developed for France is as follows:

- Bundesland -> région
- Regierungsbezirk -> no correspondance
- Landkreis -> département
- Gemeindeverband -> no correspondance (theoretically communauté de communes)
- Gemeinde -> commune 
- The French statistical unit (IRIS) does not exist

In Hannover, the official AGS (ARS) starts with: 03241
03: Bundesland = Niedersachsen (Lower Saxony)
2: Regierungsbezirk = Hannover
41: Kreisfreie Stadt = Region Hannover
"""

def configure(context):
    context.stage("seville.data.spatial.iris")

def execute(context):

    # Load codes
    df_codes = context.stage("seville.data.spatial.iris")

    return df_codes[["region_id", "departement_id", "commune_id", "iris_id"]]
