
import pandas as pd
import os
import numpy as np

"""
Apply gravity model to generate a distance matrix for Bogota's census sections.
"""
# -0.2
DEFAULT_SLOPE = -0.09 # -0.09 came from IDF, value -2.0 has been calibrated
DEFAULT_CONSTANT = -2.4
DEFAULT_DIAGONAL = 1.0

def configure(context):
    context.stage("bogota.gravity.distance_matrix")
    context.stage("bogota.data.census.processed")
    context.stage("bogota.data.spatial.iris")
    context.config("gravity_slope", DEFAULT_SLOPE)
    context.config("gravity_constant", DEFAULT_CONSTANT)
    context.config("gravity_diagonal", DEFAULT_DIAGONAL)

def evaluate_gravity(population, employees, friction):
    # Initizlize production, attraction, and flow
    production = np.ones((len(population),))
    attraction = np.ones((len(population),))
    flow = np.ones((len(population), len(population)))
    converged = False

    # Perform maximum 100 iterations (but convergence will hopefully happen earlier)
    for iteration in range(int(1e2)):
        # Backup to calculate change
        previous_production = np.copy(production)
        previous_attraction = np.copy(attraction)
        previous_flow = np.copy(flow)

        # Calculate production terms
        for k in range(len(population)):
            production[k] = population[k] / np.sum(attraction * friction[k,:])

        # Calculate attraction terms
        for k in range(len(population)):
            attraction[k] = employees[k] / np.sum(production * friction[:,k])

        # Initialize new flow matrix
        flow = np.copy(friction)

        # Apply production terms
        for i in range(len(population)):
            flow[i,:] *= production[i]

        # Apply attraction terms
        for j in range(len(population)):
            flow[:,j] *= attraction[j]

        # Calculate change to previous iteration
        production_delta = np.abs(production - previous_production)
        attraction_delta = np.abs(attraction - previous_attraction)
        flow_delta = np.abs(flow - previous_flow)

        print("Gravity iteration", iteration, 
            "prod. max. Δ:", np.max(production_delta),
            "attr. max. Δ:", np.max(attraction_delta),
            "flow max. Δ:", np.max(flow_delta),
        )

        # Stop if change is sufficiently small
        if np.max(production_delta) < 1e-3 and np.max(attraction_delta) < 1e-3 and np.max(flow_delta) < 1e-3:
            converged = True
            break
    
    assert converged
    return flow

def execute(context):
    # Load data
    df_distances = context.stage("bogota.gravity.distance_matrix")
    df_pop = context.stage("bogota.data.census.processed")
    
    df_spatial = context.stage("bogota.data.spatial.iris")

    df_pop["departement_id"] = 11001 #ToDo

    #Aggregate population to commune_id level
    df_population, df_employees = pop_employees_commune_id(df_spatial, df_pop)

    # Find the set of used municipalities (also taking into account zero flows)
    municipalities = set(df_population["origin_id"])
    municipalities |= set(df_employees["destination_id"])
    municipalities |= set(df_distances["origin_id"])
    municipalities |= set(df_distances["destination_id"])
    municipalities = sorted(list(municipalities))

    
    print(type(municipalities))  # Should be list
    print(set(type(m) for m in municipalities))  


    df_pop = df_distances.copy()
    nan_rows = df_pop[df_pop.isnull().any(axis=1)]
    nan_columns = nan_rows.columns[nan_rows.isnull().any(axis=0)]
    print(nan_rows[nan_columns])
    assert not df_pop.isnull().values.any(), "df_pop contains NaNs!"
    
    # Make sure we have all municipalities in all data sets
    df_population = df_population.set_index("origin_id").reindex(municipalities).fillna(0.0)
    df_employees = df_employees.set_index("destination_id").reindex(municipalities).fillna(0.0)

    full_pairs = pd.MultiIndex.from_product([municipalities, municipalities], names=["origin_id", "destination_id"])
    df_distances = df_distances.set_index(["origin_id", "destination_id"]).reindex(full_pairs).fillna(0.0).reset_index()

    assert not df_distances.isnull().values.any(), "Df distance 3 contains NaNs!"
    # NaNs are generated because there are municipalities pairs that don't match from running the below
    # df_distances = df_distances.set_index(["origin_id", "destination_id"]).reindex(pd.MultiIndex.from_product([
    #     municipalities, municipalities
    # ]))

    # Transform from a list into a matrix
    distances = df_distances["distance_km"].values.reshape((len(municipalities), len(municipalities)))

    # Check for nan in distance matrix
    
    assert not np.isnan(distances).any(), "Distance matrix contains NaNs!"

    # Run model
    population = df_population["population"] 
    employees = df_employees["employees"]

    # Balancing of the remaining population and workplaces
    observations = min(np.sum(population), np.sum(employees))
    population *= observations / np.sum(population)
    employees *= observations / np.sum(employees)

    # Model parameters estimated from Île-de-France
    slope = context.config("gravity_slope")
    constant = context.config("gravity_constant")
    diagonal = context.config("gravity_diagonal")

    friction = np.exp(slope * distances + constant) + np.eye(len(municipalities)) * diagonal
    flow = evaluate_gravity(population, employees, friction)

    # Convert to data frame
    df_matrix = pd.DataFrame({
        "weight": flow.reshape((-1,)),
    }, index = pd.MultiIndex.from_product([municipalities, municipalities], names = [
        "origin_id", "destination_id"
    ])).reset_index()

    # Calculate totals
    df_total = df_matrix[["origin_id", "weight"]].groupby("origin_id").sum().reset_index().rename({ "weight" : "total" }, axis = 1)
    df_matrix = pd.merge(df_matrix, df_total, on = "origin_id")

    # Fix missing flows
    f_missing_total = df_matrix["total"] == 0.0
    df_matrix.loc[f_missing_total & (df_matrix["origin_id"] == df_matrix["destination_id"]), "weight"] = 1.0
    df_matrix.loc[f_missing_total, "total"] = 1.0

    # Convert to probability
    df_matrix["weight"] = df_matrix["weight"] / df_matrix["total"]
    df_matrix = df_matrix[["origin_id", "destination_id", "weight"]]

    # One representing work, one representing education
    return df_matrix, df_matrix

def pop_employees_commune_id(df_zones, df_pop_big):

    gdf_communes = gpd.GeoDataFrame(df_zones, geometry='geometry').drop_duplicates('commune_id')

    gdf_communes['commune_area'] = gdf_communes.geometry.area

    municipality_totals = gdf_communes.groupby('departement_id')['commune_area'].sum().reset_index()
    municipality_totals = municipality_totals.rename(columns={'commune_area': 'total_muni_area'})

    #Create Weights
    mapping = pd.merge(gdf_communes[['commune_id', 'departement_id', 'commune_area']], 
                       municipality_totals, on='departement_id')
    
    mapping['spatial_weight'] = mapping['commune_area'] / mapping['total_muni_area']

    # Distribute the Population and apply weight
    df_disaggregated = pd.merge(mapping, df_pop_big, on='departement_id')
    
    df_disaggregated['population'] = df_disaggregated['weight'].sum() * df_disaggregated['spatial_weight']

    df_population = df_disaggregated.rename(columns={"commune_id": "origin_id"})[["origin_id", "population"]]
    df_population = df_population.drop_duplicates().reset_index()[["origin_id", "population"]]
    #employees
    df_employed_agents = df_pop_big[df_pop_big['employment_status'] == 'Employed']
    df_muni_jobs = df_employed_agents.groupby("municipality_id")["weight"].sum().reset_index()
    
    df_employees = pd.merge(mapping[['commune_id', 'municipality_id', 'spatial_weight']], 
                            df_muni_jobs, on="municipality_id")
    df_employees['employees'] = df_employees['weight'].sum() * df_employees['spatial_weight']
    df_employees = df_employees.rename(columns={"commune_id": "destination_id"})[["destination_id", "employees"]]
    df_employees = df_employees.drop_duplicates().reset_index()[["destination_id", "employees"]]
   

    return df_population, df_employees