import pandas as pd
import os
import numpy as np
import geopandas as gpd

"""
This stage outputs the number of employed people in each census section (commune) in Seville.
"""

def configure(context):
    context.config("data_path")

    context.config("seville.employment_position_distribution", "grid/employees/mee24_250m.shp")


    context.stage("seville.data.spatial.iris")

def execute(context):


    # Import 250m x 250m grid, each tile has number of employees working in the corresponding area
    FILE_URL = f"{context.config('data_path')}/{context.config('seville.employment_position_distribution')}"
    grid_gdf = gpd.read_file(FILE_URL)
    grid_gdf = grid_gdf.rename(columns={"empleo":"employees"})[['employees', 'geometry']]

    spatial_gdf = gpd.GeoDataFrame(context.stage("seville.data.spatial.iris"))

    grid_gdf = grid_gdf.to_crs(25830)
    spatial_gdf = spatial_gdf.to_crs(25830)
    grid_gdf["cell_area"] = grid_gdf.geometry.area

    # TODO: fix number of employees
    grid_gdf.loc[grid_gdf['employees']==-1, 'employees'] = 1


    # Calculate number of employees in each administrative unit

    intersections = gpd.overlay(
    grid_gdf,
    spatial_gdf,
    how="intersection"
    )

    intersections["intersection_area"] = (
        intersections.geometry.area
    )

    intersections["employees_allocated"] = (
        intersections["employees"]
        * intersections["intersection_area"]
        / intersections["cell_area"]
    )

    district_employees = (
        intersections
        .groupby("commune_id")["employees_allocated"]
        .sum()
        .reset_index()
    )

    spatial_gdf = spatial_gdf.merge(
        district_employees,
        on="commune_id",
        how="left"
    )


    spatial_gdf['weight'] = spatial_gdf['employees_allocated']


    return spatial_gdf

def validate(context):
    filenames = [
        "seville.employment_position_distribution",
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Census household data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
