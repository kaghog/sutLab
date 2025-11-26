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
    context.stage("hannover.data.spatial.admin_boundary")


def execute(context):
    # Load codes
    df_codes = context.stage("hannover.data.spatial.admin_boundary")[
        ["mikrobezirk_code", "district_code"]
    ]

    # Clean up identifiers
    # region_id: 03 (Niedersachsen)
    df_codes["region_id"] = "03"
    df_codes["region_id"] = df_codes["region_id"].astype("category")

    # kreis_code: Kreis-level code (Region Hannover) - 5 chars
    df_codes["kreis_code"] = "03241"
    df_codes["kreis_code"] = df_codes["kreis_code"].astype("category")

    # departement_id: 3-digit district code (neighborhood level, comes from shapefile)
    df_codes["departement_id"] = df_codes["district_code"].astype("category")

    # commune_id: 4-digit mikrobezirk code (smallest unit, comes from shapefile)
    df_codes["commune_id"] = df_codes["mikrobezirk_code"].astype("category")

    # ID Structure:
    # region_id = "03" (2 chars) - Niedersachsen
    # kreis_code = "03241" (5 chars) - Region Hannover (for IPF)
    # departement_id = "XXX" (3 chars) - District/neighborhood
    # commune_id = "XXXX" (4 chars) - Mikrobezirk

    return df_codes[
        [
            "region_id",
            "kreis_code",
            "departement_id",
            "commune_id",
        ]
    ]
