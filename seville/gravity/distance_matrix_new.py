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
    
    pop_locations = population_centroids.sort_values('id')
    emplo_locations = employment_centroids.sort_values('id')

    municipalities = pop_locations["id"].values

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
