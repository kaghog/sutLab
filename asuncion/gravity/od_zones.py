import os
import geopandas as gpd
import zipfile
import numpy as np
import pandas as pd

"""
"""

def configure(context):
    context.config("analysis_path")
    context.config("data_path")

    context.config("asuncion.districts_shp", "spatial/distritos.geojson")
    context.stage("asuncion.data.spatial.iris")
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.data.census.employees")
    context.stage("asuncion.data.homes")
    context.stage("asuncion.data.work")
    context.stage("asuncion.data.codes")



# keep only districts within area covered by iris stage
def clean_distrits(gdf_districts, gdf_iris):
    gdf_districts = gdf_districts.to_crs(gdf_iris.crs)
    gdf_iris = gdf_iris.copy()
    gdf_iris["geometry"] = gdf_iris.geometry.centroid
    joined = gpd.sjoin(
        gdf_iris,
        gdf_districts,        
        how='left',
        predicate='within'
    )
    joined = joined[['district_id', 'commune_id']]
    joined = joined[joined['district_id'].notna()]
    joined['macrozone_id'] = joined['district_id']
    commune2macrozone_map = joined.copy()
    joined = joined.drop_duplicates(subset='district_id')
    assert len(joined) != 0

    gdf_districts = gdf_districts.merge(joined, on='district_id')
    gdf_districts = gdf_districts[["district_id", "macrozone_id", "geometry"]]

    return gdf_districts, commune2macrozone_map


def calculate_weighted_centroids(
        locations_gdf,
        spatial_gdf,
        value_column,
        spatial_id_column,
        target_crs=25830,
        missing_value=-1,
        missing_replacement=1
):

    # Project CRS
    locations_gdf = locations_gdf.to_crs(target_crs)
    spatial_gdf = spatial_gdf.to_crs(target_crs)

    # Intersections
    locations_gdf = gpd.sjoin(locations_gdf, spatial_gdf, predicate="within", how="inner")

    locations_gdf["x"] = locations_gdf.centroid.x

    locations_gdf["y"] = locations_gdf.centroid.y



    # Weighted coordinates
    locations_gdf["wx"] = locations_gdf["x"] * locations_gdf[value_column]
    locations_gdf["wy"] = locations_gdf["y"] * locations_gdf[value_column]

    # Aggregate by district
    weighted = (
        locations_gdf.groupby(spatial_id_column)
        .agg({
            value_column:"sum",
            "wx":"sum",
            "wy":"sum"
        })
        .reset_index()
    )

    # Weighted centroid coordinates
    weighted["weighted_x"] = weighted["wx"] / weighted[value_column]
    weighted["weighted_y"] = weighted["wy"] / weighted[value_column]

    # Create centroid geometry
    weighted_centroids = gpd.GeoDataFrame(
        weighted,
        geometry=gpd.points_from_xy(
            weighted["weighted_x"],
            weighted["weighted_y"]
        ),
        crs=locations_gdf.crs
    )

    weighted_centroids[value_column] = weighted_centroids[value_column]

    return weighted_centroids


def execute(context):
    # Load spatial data
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.districts_shp')}"
    gdf_districts = gpd.read_file(FILE_URL)
    gdf_districts = gdf_districts.rename(columns={"DIST_DESC_": "district", "DPTO_DESC":"departement"})[['district', 'departement', 'geometry']]
    gdf_districts = gdf_districts[(gdf_districts["departement"] == "ASUNCIÓN") | (gdf_districts["departement"] == "CENTRAL")]
    gdf_districts["district"] = gdf_districts["district"].str.upper()

    from asuncion.data.codes import normalize_codes
    gdf_districts = normalize_codes(gdf_districts,context.stage("asuncion.data.codes"))

    # use iris to assign ID codes to districts
    gdf_iris = context.stage("asuncion.data.spatial.iris")[['commune_id', 'geometry']]
    od_zones, commune2macrozone_map = clean_distrits(gdf_districts, gdf_iris)


    # Load data
    print()
    df_population = context.stage("asuncion.data.census.population")
    print(commune2macrozone_map.info())
    print(df_population.info())
    print(commune2macrozone_map)
    print(df_population)
    df_population = df_population.merge(commune2macrozone_map)

    assert len(df_population) != 0
    df_population = df_population.groupby("macrozone_id")["weight"].sum().reset_index()

    df_employees = context.stage("asuncion.data.census.employees")
    df_employees = df_employees.groupby("district_id")["weight"].sum().reset_index()

    print(df_employees.info())
    df_employees = df_employees.rename(columns={"district_id":"macrozone_id"})

    df_homes = context.stage("asuncion.data.homes").copy()
    df_homes = df_homes.rename(columns={"weight":"population"})
    df_work = context.stage("asuncion.data.work")

    population_centroids = calculate_weighted_centroids(
        locations_gdf=df_homes,
        spatial_gdf=od_zones,
        value_column="population",
        spatial_id_column="macrozone_id"
    )

    employment_centroids = calculate_weighted_centroids(
        locations_gdf=df_work,
        spatial_gdf=od_zones,
        value_column="employees",
        spatial_id_column="macrozone_id"
    )


    print(df_population.info())
    print(df_employees.info())

    df_population["population"] = df_population["weight"]
    df_employees["employees"] = df_employees["weight"]
    print(df_employees)
    print(employment_centroids)
    population_centroids = population_centroids[["macrozone_id", "geometry"]].merge(df_population[["macrozone_id", "population"]])
    employment_centroids = employment_centroids[["macrozone_id", "geometry"]].merge(df_employees[["macrozone_id", "employees"]])
    print(employment_centroids)
    exit(0)
    return od_zones[['macrozone_id', 'geometry']], population_centroids, employment_centroids, commune2macrozone_map




def validate(context):
    filenames = [
        "asuncion.districts_shp",
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Census household data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
