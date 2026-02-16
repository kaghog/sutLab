import os
import geopandas as gpd
import zipfile
import numpy as np

"""
This stages loads a file containing census section codes and census section shapefiles.

"""

def configure(context):
    context.config("data_path")
    context.config("bogota.geospatial_shp")

def execute(context):
    # Load shapes
    CSV_FILE = f"{context.config('data_path')}/{context.config('bogota.geospatial_shp')}"
    gdf = gpd.read_file(CSV_FILE)[
            ['MUNCodigo', 'MUNNombre', 'LOCNombre', 'UTAM', 'UTAMNombre', 'UTAMArea',
       'geometry']
        ]

    # Rename
    gdf = gdf.rename(columns = {
        "MUNCodigo": "municipality_id",
        "MUNNombre": "municipality_name",
        "LOCNombre": "locality_name",
        "UTAMNombre": "utam_name",
    })

    #fix utam strings
    gdf['utam_code'] = gdf['UTAM'].str.replace(
    r'(\d+)',
    lambda x: x.group(0).zfill(3),
    regex=True
    )
    gdf['utam_code'] = 'UTAM' + gdf['UTAM'].str.extract(r'(\d+)')[0].str.zfill(3)

    gdf["municipality_id"] = gdf["municipality_id"].astype("int").astype(str)
    
    gdf = gdf[gdf["municipality_id"] == "11001"] #filter out Bogota

    

    return gdf[["municipality_id", "municipality_name", "locality_name", "utam_code", "utam_name", "geometry"]]

def validate(context):
    CSV_FILE = f"{context.config('data_path')}/{context.config('bogota.geospatial_shp')}"
    if not os.path.exists(CSV_FILE):
        raise RuntimeError(f"Census section geo-spatial data is not available at location {CSV_FILE}")

    return os.path.getsize(CSV_FILE)
