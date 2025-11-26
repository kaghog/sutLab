import os

import geopandas as gpd

"""
This stages loads a file containing population data for city Hannover including the Mikrobezirk codes
"""


def configure(context):
    context.config("data_path")
    context.config("hannover.population_shp")
    context.config(
        "hannover.districts_shp",
        "admin_units/Statistical districts/SKH20_Statistische_Bezirke.shp",
    )


def execute(context):
    # Load shapes
    gdf_mikrobezirke = gpd.read_file(
        "{}/{}".format(
            context.config("data_path"), context.config("hannover.population_shp")
        )
    )
    gdf_districts = gpd.read_file(
        "{}/{}".format(
            context.config("data_path"), context.config("hannover.districts_shp")
        )
    )

    # print(gdf_districts.head())
    # print(gdf_districts.columns)
    # print(gdf_districts.iloc[0])

    # Mikrobezirke
    # Rename
    # print(gdf_mikrobezirke.columns)
    # print(gdf_mikrobezirke.head())
    gdf_mikrobezirke = gdf_mikrobezirke[["MIKROBZ_BA", "geometry"]]
    df_population = gdf_mikrobezirke.rename(
        columns={
            "MIKROBZ_BA": "mikrobezirk_code",
        }
    )

    # Clean
    df_population = df_population[
        df_population["mikrobezirk_code"].astype(str).str.isdigit()
    ].copy()

    # Sort by code
    df_population["mikrobezirk_code"] = df_population["mikrobezirk_code"].astype(int)
    df_population = df_population.sort_values("mikrobezirk_code").reset_index(drop=True)

    # Pad to 4-digit string
    df_population["mikrobezirk_code"] = (
        df_population["mikrobezirk_code"].astype(str).str.zfill(4)
    )

    # Districts/neighborhoods
    # Rename
    gdf_districts = gdf_districts[["STATBEZNR", "geometry"]]
    gdf_districts = gdf_districts.rename(columns={"STATBEZNR": "district_code"})

    # Clean district codes
    gdf_districts["district_code"] = (
        gdf_districts["district_code"].astype(str).str.zfill(3)
    )

    # assign district_code to each mikrobezirk based on centroid
    df_population["centroid"] = df_population.geometry.centroid
    df = gpd.sjoin(
        df_population.set_geometry("centroid"),
        gdf_districts[["district_code", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right", "centroid"])

    # Restore original geometry
    df = df.set_geometry("geometry")

    return df[["mikrobezirk_code", "district_code", "geometry"]]


def validate(context):
    if not os.path.exists(
        "%s/%s"
        % (context.config("data_path"), context.config("hannover.population_shp"))
    ):
        raise RuntimeError("German population data is not available")

    return os.path.getsize(
        "%s/%s"
        % (context.config("data_path"), context.config("hannover.population_shp"))
    )
