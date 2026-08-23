import pandas as pd
import os
import numpy as np
import geopandas as gpd

"""
This stage outputs the number of employed people in each census section (commune) in Seville.
"""

def configure(context):
    context.config("data_path")
    context.stage("seville.data.education.schools")
    context.stage("seville.data.education.universities")
    context.stage("seville.data.spatial.iris")

def execute(context):
    gdf_schools = context.stage("seville.data.education.schools").copy()
    gdf_universities = context.stage("seville.data.education.universities").copy()
    gdf_schools = gdf_schools.to_crs("EPSG:25830")
    gdf_universities = gdf_universities.to_crs("EPSG:25830")

    gdf_education = gpd.GeoDataFrame(pd.concat([gdf_schools,gdf_universities], ignore_index=True), crs=gdf_schools.crs)
    



    gdf_education["location_id"] = np.arange(len(gdf_education))
    gdf_education["location_id"] = "edu_" + gdf_education["location_id"].astype(str)

    # keep only education locations inside area of interest
    df_zones = context.stage("seville.data.spatial.iris")

    gdf_education = gpd.sjoin(gdf_education, df_zones[["geometry", "commune_id", "iris_id"]], 
        how = "left", predicate = "within").reset_index(drop = True).drop(columns = ["index_right"])
    gdf_education = gdf_education.dropna(subset=["commune_id", "iris_id"])




    return gdf_education[["location_id", "weight", "education_type", "commune_id", "iris_id", "departement_id", ]]

