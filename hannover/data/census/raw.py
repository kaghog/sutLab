from pathlib import Path

import geopandas as gpd
import pandas as pd

"""
This stage loads the raw census data for Hannover provided at Bezirk level.

TODO: This could be replaced with a Germany-wide extract from GENESIS
"""


def augment(df, data_path, mikrobezirke_shp_path):
    """Augment the dataframe with additional census columns from GPKG files."""

    EPSG_CODE = 3035
    CENSUS_DIR = Path(data_path) / "census" / "Census_2022"

    DATASETS = [
        ("Hannover_family_size.gpkg", "family_size"),
        ("Hannover_household_family_type_detailed.gpkg", "household_family_detailed"),
        ("Hannover_household_family_type.gpkg", "household_family"),
        ("Hannover_household_household_type.gpkg", "household_type"),
        ("Hannover_household_seniors.gpkg", "household_seniors"),
        ("Hannover_household_size.gpkg", "household_size"),
        ("Hannover_individual_marital_status.gpkg", "marital_status"),
        ("Hannover_population_density.gpkg", "population"),
    ]

    mikrobezirke_gdf = gpd.read_file(mikrobezirke_shp_path)
    if mikrobezirke_gdf.crs is None or mikrobezirke_gdf.crs.to_epsg() != EPSG_CODE:
        mikrobezirke_gdf = mikrobezirke_gdf.to_crs(epsg=EPSG_CODE)

    # Process each dataset
    for gpkg_file, dataset_name in DATASETS:
        gpkg_path = CENSUS_DIR / gpkg_file

        if not gpkg_path.exists():
            print(f"  Warning: {gpkg_file} not found, skipping...")
            continue

        aggregated = aggregate_gpkg_to_mikrobezirke(
            gpkg_path, mikrobezirke_gdf, dataset_name, EPSG_CODE
        )

        if aggregated:
            for col_name, series in aggregated.items():
                # Convert mikrobezirk_code to int for proper mapping
                df[col_name] = df["mikrobezirk_code"].astype(int).map(series).fillna(0)

    # Rename columns to English, drop useless ones
    df = rename_columns(df)

    return df


def aggregate_gpkg_to_mikrobezirke(
    gpkg_path, mikrobezirke_gdf, dataset_name, epsg_code
):
    grid_gdf = gpd.read_file(gpkg_path)
    if grid_gdf.crs is None or grid_gdf.crs.to_epsg() != epsg_code:
        grid_gdf = grid_gdf.to_crs(epsg=epsg_code)

    # geometry and metadata columns
    exclude_cols = {
        "id",
        "x_sw",
        "y_sw",
        "x_mp",
        "y_mp",
        "x_mp_100m",
        "y_mp_100m",
        "ags",
        "geometry",
    }
    stat_cols = [col for col in grid_gdf.columns if col not in exclude_cols]

    # columns are strings instead of numbers for some reason
    for col in stat_cols:
        grid_gdf[col] = pd.to_numeric(grid_gdf[col], errors="coerce")

    valid_coords = grid_gdf["x_mp_100m"].notna() & grid_gdf["y_mp_100m"].notna()
    has_data = (grid_gdf[stat_cols] > 0).any(axis=1)
    grid_gdf = grid_gdf[valid_coords & has_data].copy()
    if len(grid_gdf) == 0:
        return None

    # Spatial join with Mikrobezirke
    grid_centroids = grid_gdf.copy()
    grid_centroids["geometry"] = grid_gdf.geometry.centroid

    mikrobezirke_with_code = mikrobezirke_gdf[["MIKROBZ_BA", "geometry"]].copy()
    mikrobezirke_with_code["MIKROBZ_BA"] = mikrobezirke_with_code["MIKROBZ_BA"].astype(
        int
    )

    joined = gpd.sjoin(
        grid_centroids, mikrobezirke_with_code, how="left", predicate="within"
    )

    agg_dict = {col: "sum" for col in stat_cols}
    aggregated = joined.groupby("MIKROBZ_BA").agg(agg_dict)

    result = {}
    for col in stat_cols:
        result[f"{dataset_name}_{col}"] = aggregated[col]

    return result


def rename_columns(df):
    rename_map = {
        "MIKROBZ_BA": "mikrobezirk_code",
        "family_size_Insgesamt_Familien": "total_families",
        "family_size_a2Personen": "families_2_persons",
        "family_size_a3Personen": "families_3_persons",
        "family_size_a4Personen": "families_4_persons",
        "family_size_a5Personen": "families_5_persons",
        "family_size_a6Pers_und_mehr": "families_6plus_persons",
        # "household_family_detailed_Insgesamt_Familie": "total_families_detailed",
        "household_family_detailed_Ehep_ohneKind": "married_couples_no_children",
        "household_family_detailed_Ehep_mind_1Kind_unter18": "married_couples_children_under18",
        "household_family_detailed_Ehep_Kinder_ab18": "married_couples_children_over18",
        "household_family_detailed_EingetrLP_ohneKind": "registered_partners_no_children",
        "household_family_detailed_EingetrLP_mind_1Kind_unter18": "registered_partners_children_under18",
        "household_family_detailed_EingetrLP_Kinder_ab18": "registered_partners_children_over18",
        "household_family_detailed_NichtehelLG_ohneKind": "unmarried_couples_no_children",
        "household_family_detailed_NichtehelLG_mind_1Kind_unter18": "unmarried_couples_children_under18",
        "household_family_detailed_NichtehelLG_Kinder_ab18": "unmarried_couples_children_over18",
        "household_family_detailed_Vater_mind_1Kind_unter18": "single_fathers_children_under18",
        "household_family_detailed_Vater_Kinder_ab18": "single_fathers_children_over18",
        "household_family_detailed_Mutter_mind_1Kind_unter18": "single_mothers_children_under18",
        "household_family_detailed_Mutter_Kinder_ab18": "single_mothers_children_over18",
        "household_family_Insgesamt_Haushalte": "total_households",
        "household_family_EinpersHH_SingleHH": "single_person_households",
        "household_family_Paare_ohneKind": "couples_no_children",
        "household_family_Paare_mitKind": "couples_with_children",
        "household_family_Alleinerziehende": "single_parents",
        "household_family_MehrpersHHohneKernfam": "multiperson_households_no_family",
        # "household_type_Insgesamt_Haushalte": "total_households_by_type",
        # "household_type_EinpersHH_SingleHH": "single_person_households_type",
        "household_type_Ehepaare": "married_couples",
        "household_type_EingetrLebensp": "registered_partnerships",
        "household_type_NichtehelLebensg": "unmarried_partnerships",
        "household_type_AlleinerzMuetter": "single_mothers",
        "household_type_AlleinerzVaeter": "single_fathers",
        # "household_type_MehrpersHHohneKernfam": "multiperson_households_no_family_type",
        # "household_seniors_Insgesamt_Haushalte": "total_households_by_seniors",
        "household_seniors_HH_nurSenioren": "households_only_seniors",
        "household_seniors_HH_mitSenioren": "households_with_seniors",
        "household_seniors_HH_ohneSenioren": "households_without_seniors",
        # "household_size_Insgesamt_Haushalte": "total_households_by_size",
        "household_size_X1_Person": "households_1_person",
        "household_size_X2_Personen": "households_2_persons",
        "household_size_X3_Personen": "households_3_persons",
        "household_size_X4_Personen": "households_4_persons",
        "household_size_X5_Personen": "households_5_persons",
        "household_size_X6_Personen_und_mehr": "households_6plus_persons",
        # "marital_status_Insgesamt_Bevoelkerung": "total_population",
        "marital_status_Ledig": "marital_single",
        "marital_status_Verheiratet": "marital_married",
        "marital_status_Verwitwet": "marital_widowed",
        "marital_status_Geschieden": "marital_divorced",
        "marital_status_EingetrLebenspartnerschaft": "marital_registered_partnership",
        "marital_status_EingetrLebenspartVerstorben": "marital_partner_deceased",
        "marital_status_EingetrLebenspartAufgehoben": "marital_partnership_dissolved",
        "marital_status_OhneAngabe": "marital_not_specified",
        # "population_Einwohner": "population",
    }

    essential_columns = ["mikrobezirk_code", "total_population"]
    columns_to_keep = [
        col
        for col in df.columns
        if col in rename_map
        or col in essential_columns
        or col.startswith("male_")
        or col.startswith("female_")
    ]
    df = df[columns_to_keep]

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def configure(context):
    context.config("data_path")
    context.config("hannover.population_path", "census/Age_gender_MBZ.xlsx")
    context.config("hannover.population_shp")


def execute(context):
    EXCEL_PATH = "{}/{}".format(
        context.config("data_path"), context.config("hannover.population_path")
    )
    print(f"Loading census data from {EXCEL_PATH}")
    SHEET_NAME = "Altersgruppen MBZ Geschlecht"
    AGE_GROUPS = [
        (0, "0 bis 5 Jahre"),
        (6, "6 bis 14 Jahre"),
        (15, "15 bis 17 Jahre"),
        (18, "18 bis 23 Jahre"),
        (24, "24 bis 29 Jahre"),
        (30, "30 bis 44 Jahre"),
        (45, "45 bis 64 Jahre"),
        (65, "65 bis 79 Jahre"),
        (80, "80+"),
    ]

    # Read the sheet, skipping the descriptive header rows/columns
    # Assuming microbezirke is in the first column, then: 9 (male), 9 (female), 'gesamt'
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=7, header=None)
    df = df[~df[0].astype(str).str.contains("Gesamt", na=False)]
    df = df[df[0].astype(str).str.match(r"^\d+$")]

    colnames = (
        ["mikrobezirk_code"]
        + [f"male_{age[0]}" for age in AGE_GROUPS]
        + [f"female_{age[0]}" for age in AGE_GROUPS]
        + ["total_population"]
    )
    df = df.iloc[:, : len(colnames)]  # drop empty columns if present
    df.columns = colnames

    # additional census data
    data_path = context.config("data_path")
    mikrobezirke_shp_path = "{}/{}".format(
        data_path, context.config("hannover.population_shp")
    )
    df = augment(df, data_path, mikrobezirke_shp_path)

    return df
