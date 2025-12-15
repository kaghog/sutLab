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

def execute(context):
    # Load shapes
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.population_shp')}"
    gdf_census_sections = gpd.read_file(CSV_FILE)[
            ["CUSEC", "CMUN", "CPRO", "geometry"]
        ]

    # Rename
    gdf_census_sections = gdf_census_sections.rename(columns = {
        "CUSEC": "census_section_id",
        "CMUN": "municipality_id",
        "CPRO": "province_id",
    })
    
    # Clean
    gdf_census_sections = gdf_census_sections[gdf_census_sections["census_section_id"].astype(str).str.isdigit()].copy()
    
    # Filter only Seville
    gdf_census_sections =  gdf_census_sections[gdf_census_sections["province_id"] == "41"]

    return gdf_census_sections[["census_section_id", "geometry"]]

def validate(context):
    CSV_FILE = f"{context.config('data_path')}/{context.config('seville.population_shp')}"
    if not os.path.exists(CSV_FILE):
        raise RuntimeError(f"Census section geo-spatial data is not available at location {CSV_FILE}")

    return os.path.getsize(CSV_FILE)
