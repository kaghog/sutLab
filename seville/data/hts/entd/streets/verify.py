from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts
from geopy.distance import geodesic
import geopy
import time
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point



# TODO: implement checking based on the census district shapefile / city shapefile

def configure(context):
    context.config("data_path")
    context.config("seville.province_shapefile", "seville_province/seville_province.gpkg.shp")
    context.config("seville.street_data", "street_data.csv")

    context.stage("seville.data.hts.entd.streets.manual_cleaning")

def execute(context):


    # Merge both dataframes:
    CSV_PATH = f"{context.config('data_path')}/{context.config('seville.street_data')}"
    df_streets_original = pd.read_csv(CSV_PATH, sep='\t', dtype={"location":str})
    df_streets_fixed = context.stage("seville.data.hts.entd.streets.manual_cleaning")


    df_streets = df_streets_original.merge(df_streets_fixed[['municipality', 'zone', 'street', 'location']], 
                           on=['municipality', 'zone', 'street'], 
                           how='left', 
                           suffixes=('', '_updated'))
    
    # Replace values in 'location' column from the merged DataFrame where there is an update
    df_streets['location'] = df_streets['location_updated'].fillna(df_streets['location'])



    print("Checking that all locations are in the province of Seville...")

    df_streets = df_streets[df_streets["municipality"] != "Otros"] # Filter out unknown


    def parse_location(location_str):
        # Clean and parse the string
        location_str = str(location_str)
        cleaned_str = location_str.strip("()")
        latitude, longitude  = cleaned_str.split(',')
        return Point(float(longitude), float(latitude))

    # Apply the function to the 'location' column
    df_streets['geometry'] = df_streets['location'].apply(parse_location)

    gdf_points = gpd.GeoDataFrame(df_streets, crs='EPSG:4326')

    SHP_FILE = f"{context.config('data_path')}/{context.config('seville.province_shapefile')}"
    gdf_shapefile = gpd.read_file(SHP_FILE, crs='EPSG:4326')
    gdf_points['is_inside'] = gdf_points.geometry.apply(lambda point: gdf_shapefile.contains(point).any())


    print(gdf_points[gdf_points['is_inside']==False])
    assert all(gdf_points['is_inside']==True)

    return df_streets[['municipality', 'zone', 'street', 'geometry', 'location']]
