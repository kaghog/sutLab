import numpy as np
import pandas as pd
import geopandas as gpd

"""
Yield home zones for Spain based on synthetic population data.
"""

def configure(context):
    context.config("asuncion.homes_asuncion", "locations/viviendas_asuncion.geojson")
    context.config("asuncion.home_central", "locations/viviendas_central.geojson")

    context.config("data_path")

    context.stage("asuncion.data.spatial.iris")



def execute(context):
    # Load data
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.homes_asuncion')}"
    homes_asuncion = gpd.read_file(FILE_URL, dtype=str)
    FILE_URL = f"{context.config('data_path')}/{context.config('asuncion.home_central')}"
    homes_central = gpd.read_file(FILE_URL, dtype=str)

    homes = pd.concat([homes_asuncion, homes_central])
    homes = homes[["geometry"]].copy()
    homes["home_location_id"] = np.arange(len(homes))
    homes["weight"] = 1.0

    # add spatial identifiers
    gdf_spatial = context.stage("asuncion.data.spatial.iris")
    homes = homes.to_crs(gdf_spatial.crs)
    homes = gpd.sjoin(homes, gdf_spatial, predicate="within", how="inner")

    assert len(homes) != 0

    return homes[["home_location_id", "weight", "commune_id", "iris_id", "geometry",]]
