import pandas as pd

"""
Load population data
"""

def configure(context):
    context.config("data_path")
    context.config("bogota.population", "population.csv")
    context.stage("bogota.data.spatial.iris")

def execute(context):

    df_population = pd.read_csv(
        "%s/%s" % (context.config("data_path"), context.config("bogota.population")),
        encoding = "latin1"
    )

    if "household id" in df_population.columns:
        df_population["household_id"] = df_population["household id"]

    if "person id" in df_population.columns:
        df_population["person_id"] = df_population["person id"]

    df_population["departement_id"] = "11001"
    df_population["departement_id"] = df_population["departement_id"].astype("category")


    df_spatial = context.stage("bogota.data.spatial.iris")

    return df_population

def add_commune(df_spatial, df_pop):
    gdf_communes = gpd.GeoDataFrame(df_spatial, geometry='geometry').drop_duplicates('commune_id')

    gdf_communes['commune_area'] = gdf_communes.geometry.area

    municipality_totals = gdf_communes.groupby('departement_id')['commune_area'].sum().reset_index()
    municipality_totals = municipality_totals.rename(columns={'commune_area': 'total_muni_area'})
    
    #Create Weights
    mapping = pd.merge(gdf_communes[['commune_id', 'departement_id', 'commune_area', 'iris_id']], 
                       municipality_totals, on='departement_id')
    
    mapping['spatial_weight'] = mapping['commune_area'] / mapping['total_muni_area']

    # We only want to assign each household ONCE
    df_households = df_pop[['household_id', 'departement_id']].drop_duplicates()

    # --- 3. The Assignment Logic ---
    assigned_list = []

    # Iterate through each department to perform weighted sampling
    for dept, group in df_households.groupby('departement_id'):
        # Get the communes and weights for THIS department
        dept_mapping = mapping[mapping['departement_id'] == dept]
        
        communes = dept_mapping['commune_id'].values
        probs = dept_mapping['spatial_weight'].values
        
        # Ensure probabilities sum to 1 (float precision fix)
        probs = probs / probs.sum()

        # Randomly pick a commune for every household in this department
        # This is vectorized and very fast
        assignments = np.random.choice(communes, size=len(group), p=probs)
        
        group['commune_id'] = assignments
        assigned_list.append(group[['household_id', 'commune_id']])

    # Combine all assignments
    df_assignments = pd.concat(assigned_list)

    # --- 4. Merge back to the full population ---
    # This ensures everyone in the same household gets the same commune_id
    df_final = pd.merge(df_pop, df_assignments, on='household_id', how='left')

    return df_final
