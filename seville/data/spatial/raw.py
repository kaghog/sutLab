import os
import geopandas as gpd
import zipfile
import numpy as np

"""
This stages loads a file containing census section codes and census section shapefiles.

NOTE: This is stage is alternatively called "population.raw" in other pipelines (e.g., Hannover). 
For Seville, the stage is renamed to better reflect its nature since it is no longer related to population data.
"""

def configure(context):
    context.config("data_path")
    context.config("seville.population_shp", "shapefiles/census_district_shapefiles/SECC_CE_20220101.shp")
    if context.config("seville_locations_area_selection") == 'agglomeration':
        context.stage("seville.data.select_agglomeration")


def execute(context):
    # Load shapes
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.population_shp')}"
    gdf_census_sections = gpd.read_file(CSV_FILE)[
            ["CUSEC", "CUMUN", "CPRO", "geometry"]
        ]

    # Rename
    gdf_census_sections = gdf_census_sections.rename(columns = {
        "CUSEC": "census_section_id",
        "CUMUN": "municipality_id",
        "CPRO": "province_id",
    })
    


    # Clean
    gdf_census_sections["province_id"] = gdf_census_sections["province_id"].astype(str)
    gdf_census_sections["municipality_id"] = gdf_census_sections["municipality_id"].astype(str)
    gdf_census_sections["census_section_id"] = gdf_census_sections["census_section_id"].astype(str)

    
    # Filter
    gdf_census_sections = gdf_census_sections[gdf_census_sections["census_section_id"].str.isdigit()].copy()
    gdf_census_sections =  gdf_census_sections[gdf_census_sections["province_id"] == "41"]


    selected_area = context.config("seville_locations_area_selection")
    if selected_area == 'province':
        # no changes
        gdf_census_sections = gdf_census_sections
    elif selected_area == 'agglomeration':
        # use data of municipalities inside agglomeration
        agglomeration_mun = context.stage("seville.data.select_agglomeration")
        gdf_census_sections = gdf_census_sections[gdf_census_sections["municipality_id"].isin(agglomeration_mun['municipality_id'])]
    elif selected_area == 'municipality':
        # use data of Seville municipality only
        gdf_census_sections = gdf_census_sections[gdf_census_sections["municipality_id"] == "41091"]
    else:
        raise NotImplementedError


    return gdf_census_sections[["census_section_id", "geometry"]]

def validate(context):
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.population_shp')}"
    if not os.path.exists(CSV_FILE):
        raise RuntimeError(f"Census section geo-spatial data is not available at location {CSV_FILE}")

    return os.path.getsize(CSV_FILE)
