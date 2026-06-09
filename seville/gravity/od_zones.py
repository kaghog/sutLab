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
    context.config("seville.od_districts", "shapefiles/districts_shapefile.gpkg")
    context.config("seville.municipalities_shp", "shapefiles/municipalities.gpkg")
    context.config("seville.selected_municipalities", "aglomeracion_urban_de_sevilla.csv")

    context.config("seville.population_grid", "grid/population/mep24_250.shp")
    context.config("seville.employees_grid", "grid/employees/mee24_250m.shp")
    context.config("seville.companies", "grid/companies/estab24_pt.shp")


def fix_employees_grid(gdf_companies, employees_grid):

    companies_loc = gdf_companies
    companies_loc['estrato_em'].value_counts()
    # Lower third of the interval is used here
    EMPLOYEE_COUNT_MAP = {
        "Menos de 10 asalariados":3,
        "De 10 a 19 asalariados":13,       
        "De 20 a 49 asalariados":29,      
        "De 50 a 99 asalariados":66,        
        "De 100 a 249 asalariados":150,      
        "250 o más asalariados":500,
    }

    companies_loc['predicted_employee_count'] = companies_loc['estrato_em'].map(EMPLOYEE_COUNT_MAP)
    companies_loc = companies_loc[companies_loc['provincia'] == 'Sevilla'][['predicted_employee_count', 'geometry']]


    employees_grid['tile_id'] = np.arange(len(employees_grid))
    companies_assigned = gpd.sjoin(
        companies_loc,
        employees_grid[['tile_id', 'geometry']],        
        how='left',
        predicate='within'
    )

    companies_assigned = companies_assigned.groupby('tile_id')['predicted_employee_count'].sum().reset_index(name='predicted_employee_count')
    employees_grid = employees_grid.merge(companies_assigned[['tile_id', 'predicted_employee_count']])


    # fix where values are missing
    mask = employees_grid['employees'] == -1
    print('Number of tiles with missing job count: ', mask.sum())
    employees_grid.loc[mask, 'employees'] = employees_grid.loc[mask, 'predicted_employee_count']

    return employees_grid


def calculate_weighted_centroids(
        grid_gdf,
        spatial_gdf,
        value_column,
        spatial_id_column,
        target_crs=25830,
        missing_value=-1,
        missing_replacement=1
):

    # Project CRS
    grid_gdf = grid_gdf.to_crs(target_crs)
    spatial_gdf = spatial_gdf.to_crs(target_crs)

    # Original cell area
    grid_gdf["cell_area"] = grid_gdf.geometry.area

    # Fix missing values
    grid_gdf.loc[grid_gdf[value_column] == missing_value,value_column] = missing_replacement

    # Intersections
    intersections = gpd.overlay(grid_gdf, spatial_gdf, how="intersection")

    # Area of intersected piece
    intersections["intersection_area"] = (intersections.geometry.area)

    allocated_column = f"{value_column}_allocated"

    # Allocate employees proportionally
    intersections[allocated_column] = (
        intersections[value_column]
        * intersections["intersection_area"]
        / intersections["cell_area"]
    )

    # Centroid of intersected geometry
    intersections["centroid"] = intersections.geometry.centroid

    intersections["x"] = intersections.centroid.x

    intersections["y"] = intersections.centroid.y

    # Weighted coordinates
    intersections["wx"] = intersections["x"] * intersections[allocated_column]
    intersections["wy"] = intersections["y"] * intersections[allocated_column]

    # Aggregate by district
    weighted = (
        intersections.groupby(spatial_id_column)
        .agg({
            allocated_column:"sum",
            "wx":"sum",
            "wy":"sum"
        })
        .reset_index()
    )

    # Weighted centroid coordinates
    weighted["weighted_x"] = weighted["wx"] / weighted[allocated_column]
    weighted["weighted_y"] = weighted["wy"] / weighted[allocated_column]

    # Create centroid geometry
    weighted_centroids = gpd.GeoDataFrame(
        weighted,
        geometry=gpd.points_from_xy(
            weighted["weighted_x"],
            weighted["weighted_y"]
        ),
        crs=grid_gdf.crs
    )

    weighted_centroids[value_column] = weighted_centroids[allocated_column]

    return weighted_centroids

def execute(context):
    # Load data
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.od_districts')}"
    gdf_districts = gpd.read_file(CSV_FILE)
    gdf_districts = gdf_districts.rename(columns={"Distri_11D": "id"})

    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.municipalities_shp')}"
    gdf_municipalities = gpd.read_file(CSV_FILE)
    gdf_municipalities = gdf_municipalities.rename(columns={"nombre": "id"})


    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.selected_municipalities')}"
    df_metropolitan_mun = pd.read_csv(CSV_FILE, sep=';')



    # select zones for gravity model: metropolitan municipalities + seville macrozones
    od_zones = gdf_municipalities[gdf_municipalities['id'].isin(df_metropolitan_mun['municipality'])]

    # instead of seville municipality area use its individual districts
    od_zones = od_zones[od_zones['id']!='Sevilla']

    gdf_districts = gdf_districts.to_crs(od_zones.crs)
    od_zones = pd.concat([od_zones, gdf_districts])
    


    # Add population data from 250x250m grid
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.population_grid')}"
    population_grid = gpd.read_file(CSV_FILE)
    population_grid = population_grid.rename(columns={"POB_TOT":"population"})


    # Add employees data from 250x250m grid
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.employees_grid')}"
    employees_grid = gpd.read_file(CSV_FILE)
    employees_grid = employees_grid.rename(columns={"empleo":"employees"})


    # Some tiles have unknown employee count, we estimate it using data describing located companies
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.companies')}"
    gdf_companies = gpd.read_file(CSV_FILE)

    employees_grid = fix_employees_grid(gdf_companies, employees_grid)



    # Calculate weighted centroids

    employment_centroids = calculate_weighted_centroids(
        grid_gdf=employees_grid,
        spatial_gdf=od_zones,
        value_column="employees",
        spatial_id_column="id"
    )

    population_centroids = calculate_weighted_centroids(
        grid_gdf=population_grid,
        spatial_gdf=od_zones,
        value_column="population",
        spatial_id_column="id"
    )


    #od_zones.to_file(f"{context.config('analysis_path')}/od_zones.gpkg")
    #population_centroids.to_file(f"{context.config('analysis_path')}/population_centroids.gpkg")
    #employment_centroids.to_file(f"{context.config('analysis_path')}/employment_centroids.gpkg")

    
    return od_zones, population_centroids, employment_centroids

def validate(context):
    filenames = [
        "seville.od_districts",
        "seville.municipalities_shp",
        "seville.selected_municipalities",
        "seville.population_grid",
        "seville.employees_grid",
        "seville.companies",        
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Census household data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
