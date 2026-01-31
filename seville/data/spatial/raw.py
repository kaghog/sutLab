import os
import geopandas as gpd
import zipfile
import numpy as np

"""
This stages loads a file containing census section codes and census section shapefiles.
This serves as alternative to missleadingly named population.raw
"""

def configure(context):
    context.config("data_path")
    context.config("seville.population_shp", "shapefiles/census_district_shapefiles/SECC_CE_20220101.shp")
    context.config("seville_city_data_only")

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

    if context.config("seville_city_data_only") == True:
        gdf_census_sections = gdf_census_sections[gdf_census_sections["municipality_id"] == "41091"]



    return gdf_census_sections[["census_section_id", "geometry"]]

def validate(context):
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.population_shp')}"
    if not os.path.exists(CSV_FILE):
        raise RuntimeError(f"Census section geo-spatial data is not available at location {CSV_FILE}")

    return os.path.getsize(CSV_FILE)
