import os
import geopandas as gpd
import zipfile
import numpy as np

"""
This stages loads a file containing census section codes and census section shapefiles.
"""

def configure(context):
    context.config("data_path")
    context.config("seville.population_shp", "census_district_shapefiles/SECC_CE_20220101.shp")

def execute(context):
    # Load shapes
    gdf_census_sections = gpd.read_file("{}/{}".format(context.config("data_path"), context.config("seville.population_shp")))[
            ["CUSEC", "CMUN", "CPRO", "geometry"]
        ]

    # Rename
    gdf_census_sections = gdf_census_sections.rename(columns = {
        "CUSEC": "census_section_code",
        "CMUN": "municipality_code",
        "CPRO": "province_code",
    })
    
    # Clean
    gdf_census_sections = gdf_census_sections[gdf_census_sections["census_section_code"].astype(str).str.isdigit()].copy()
    
    # Filter only Seville
    gdf_census_sections =  gdf_census_sections[gdf_census_sections["province_code"] == "41"]

    # Sort by code
    # df_population["census_section_code"] = df_population["census_section_code"].astype(int)
    # df_population = df_population.sort_values("census_section_code").reset_index(drop=True)
    
    # Pad to 4-digit string
    # df_population["census_section_code"] = df_population["census_section_code"].astype(str).str.zfill(4)
    
    # df_population[["census_section_code", "geometry"]]
    return gdf_census_sections

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("seville.population_shp"))):
        raise RuntimeError("Census section geo-spatial data is not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("seville.population_shp")))
