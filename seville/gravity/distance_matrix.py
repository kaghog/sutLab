from tqdm import tqdm
import pandas as pd
import numpy as np
import numpy.linalg as la

"""
Generates a distance matrix for the Seville's census sections.
"""

def configure(context):
    context.stage("seville.gravity.od_zones")

def execute(context):


    _, population_centroids, employment_centroids = context.stage("seville.gravity.od_zones")
        
    assert len(population_centroids) == len(employment_centroids)

    # The existing metres-to-kilometres conversion requires EPSG:25830.
    if population_centroids.crs is None or employment_centroids.crs is None:
        raise ValueError("Population and employment centroids must have a CRS")
    if (population_centroids.crs != employment_centroids.crs
            or population_centroids.crs.to_epsg() != 25830):
        raise ValueError("Both centroid tables must use EPSG:25830 (metres)")

    pop_locations = population_centroids.sort_values('macrozone_id')
    emplo_locations = employment_centroids.sort_values('macrozone_id')

    # Equal lengths do not guarantee that the same zones are aligned.
    for name, locations in (("Population", pop_locations),
                            ("Employment", emplo_locations)):
        if locations["macrozone_id"].isna().any() or locations["macrozone_id"].duplicated().any():
            raise ValueError(f"{name} macrozone IDs are missing or duplicated")
        if locations.geometry.isna().any() or locations.geometry.is_empty.any():
            raise ValueError(f"{name} centroids have missing or empty geometry")
    if not np.array_equal(pop_locations["macrozone_id"].values,
                          emplo_locations["macrozone_id"].values):
        raise ValueError("Population and employment centroid macrozone IDs do not match")

    municipalities = pop_locations["macrozone_id"].values

    # Initialize matrix to zero
    distance_matrix = np.ones((len(municipalities), len(municipalities)))
    

    # Convert locations to (N,2)-array
    pop_locations = np.array([
        pop_locations["geometry"].centroid.x,
        pop_locations["geometry"].centroid.y
    ]).T

    emplo_locations = np.array([
        emplo_locations["geometry"].centroid.x,
        emplo_locations["geometry"].centroid.y
    ]).T

    
    # Calculate Euclidean distances per row
    for k in range(len(pop_locations)):
        distance_matrix[k,:] = la.norm(pop_locations[k] - emplo_locations, axis = 1)
    
    # Convert to km
    distance_matrix *= 1e-3
    
    # Formatting into a data frame
    df_distances = pd.DataFrame({ "distance_km": distance_matrix.reshape(-1) }, index = pd.MultiIndex.from_product([
    municipalities, municipalities
    ], names = ["origin_id", "destination_id"])).reset_index()
   
    assert not df_distances.isnull().values.any(), "Df distances contains NaNs!"
    
    return df_distances
