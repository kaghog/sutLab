import geopandas as gpd
import pandas as pd
import analysis.seville.ivt_style.myplottools as myplottools
import numpy as np

def configure(context):
    context.config("output_path")
    context.config("data_path")
    context.config("analysis_path")
    context.config("output_prefix")

    context.stage("synthesis.output")
    
    context.stage("data.hts.entd.reweighted")


def compute_distances_synthetic(df_syn, threshold = 25):
    # Use euclidean_distance if available, otherwise try crowfly_distance
    if "euclidean_distance" in df_syn.columns:
        df_syn["crowfly_distance"] = 0.001 * np.array(df_syn["euclidean_distance"])
    elif "crowfly_distance" in df_syn.columns:
        df_syn["crowfly_distance"] = 0.001 * np.array(df_syn["crowfly_distance"])
    else:
        print("WARNING: No distance column found in synthetic data")
        return df_syn

    # Only consider crowfly distances shorter than <threshold> km
    df_syn_dist = df_syn[df_syn["crowfly_distance"] < threshold]
    df_syn_dist = df_syn_dist[df_syn_dist["crowfly_distance"] > 0]
    return df_syn_dist


def compute_distances_actual(df_act, threshold = 25):
    # Use routed_distance from HTS data (already in meters) as approximation for crowfly distance
    df_act["crowfly_distance"] = df_act["routed_distance"] / 1000.0  # Convert meters to km
    
    df_act_dist = df_act[df_act["crowfly_distance"] < threshold]
    df_act_dist = df_act_dist[df_act_dist["crowfly_distance"] > 0]
    return df_act_dist


def import_data_actual(context, population_selector = None):
    try:
        hts_data = context.stage("data.hts.entd.reweighted")
        if hts_data is None or any(x is None for x in hts_data):
            print("WARNING: HTS data is not available - returning None")
            return None, None, None
        
        df_act_households , df_act_persons, df_act_trips = hts_data
    except Exception as e:
        print(f"WARNING: Could not load HTS data: {e}")
        return None, None, None
    
    # First ensure number_of_vehicles is numeric and convert to boolean
    df_act_households["number_of_vehicles"] = pd.to_numeric(df_act_households["number_of_vehicles"], errors="coerce").fillna(0)
    df_act_households["car_availability"] = df_act_households["number_of_vehicles"] > 0
    # Merge to persons dataframe on household_id
    df_act_persons = df_act_persons.merge(
        df_act_households[["household_id", "car_availability"]],
        on="household_id",
        how="left"
    )
    # Fill missing households with False (if person household_id not in households df)
    df_act_persons["car_availability"] = df_act_persons["car_availability"].fillna(False)

    df_act_persons.rename(columns = {"person_weight": "weight_person"}, inplace = True)
    df_px = df_act_persons[["person_id", "weight_person", "employed", "studies",
                                                "age", "sex", "car_availability", "has_license", "has_pt_subscription", "socioprofessional_class"]]
    df_act = df_act_trips.merge(df_px, on=["person_id"], how='left')

    df_act["preceding_purpose"] = df_act["preceding_purpose"].astype(str)
    df_act["following_purpose"] = df_act["following_purpose"].astype(str)
    df_act["od"] = df_act["preceding_purpose"] + "_" + df_act["following_purpose"]

    # Only keep the persons that could have been used in activity chain matching, due to 
    # using specifc days of the week
    df_act = df_act[~df_act["weight_person"].isna()]
    df_act = df_act.set_index(["person_id"])
    df_act.sort_index(inplace=True)

    t_id = df_act_trips["person_id"].values.tolist()
    # Tag active persons (with at least one trip)
    df_act_persons["is_active"] = df_act_persons["person_id"].isin(t_id).astype(bool)
    df_persons_no_trip = df_act_persons[np.logical_not(df_act_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])
    print(df_persons_no_trip.shape, "persons without trip in hts")
    # Debug: size and coverage
    total_hts_persons = len(df_act_persons)
    persons_in_trips = df_act_trips["person_id"].nunique()
    # print(f"[DEBUG] HTS persons total={total_hts_persons}, in trips={persons_in_trips} ({persons_in_trips/total_hts_persons*100:.2f}%)")

    if population_selector:
        if "age_selector" in population_selector.keys():
            age_min = population_selector["age_selector"][0]
            age_max = population_selector["age_selector"][1]
            # Filter trip-level by person age via merged attributes
            df_act = df_act[(df_act["age"] <= age_max) & (df_act["age"] >= age_min)]
            df_persons_no_trip = df_persons_no_trip[(df_persons_no_trip["age"] <= age_max) & (df_persons_no_trip["age"] >= age_min)]
            print("INFO excluding agents NOT between the age of ", age_min, " and ", age_max)
        if "gender_selector" in population_selector.keys():
            gender = population_selector["gender_selector"]
            if gender == "male":
                g = 0
            else:
                g = 1
            df_act = df_act[df_act["sex"] == g]
            df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["sex"] == g]
            print("INFO only considering ", gender, " agents.")

    # df_act contains only those that have trips
    # Legacy return was (df_act_trip_merged, df_persons_no_trip). We now return the
    # person-level dataframe (with is_active) alongside the trips and the no-trip view.
    return df_act_persons, df_act_trips, df_persons_no_trip

    

def execute(context):


    # ==============================================================================
    # LOAD INPUT DATA
    # ==============================================================================


    df_act_persons, df_act_trips, _ = import_data_actual(context)



    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")
    
    # Trip geometries corresponding to HTS trips
    filepath = "%s/%strips.csv" % (output_path, output_prefix)
    gdf_syn_trips = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    filepath = "%s/%spersons.csv" % (output_path, output_prefix)
    gdf_syn_persons = pd.read_csv(filepath, encoding = "latin1", sep = ";")
    filepath = "%s/%shomes.gpkg" % (output_path, output_prefix)
    gdf_syn_households = gpd.read_file(filepath)

    # City boundary
    #  
    gdf_city = gpd.read_file(f"{context.config('data_path')}/shapefiles/municipalities.gpkg")
    gdf_city = gdf_city[gdf_city['nombre']=="Sevilla"]



    gdf_syn_trips = gdf_syn_trips.merge(gdf_syn_persons[["person_id", "household_id"]], on="person_id")
    gdf_syn_trips = gdf_syn_trips.merge(gdf_syn_households[["household_id", "geometry"]], on="household_id")
    
    gdf_syn_trips = gpd.GeoDataFrame(gdf_syn_trips, crs=gdf_syn_households.crs)

    # ==============================================================================
    # FILTER HTS TRIPS OF PEOPLE LIVING IN THE CITY
    # ==============================================================================

    if gdf_syn_trips.crs != gdf_city.crs:
        gdf_city = gdf_city.to_crs(gdf_syn_trips.crs)

    city_geom = gdf_city.union_all()

    # Keep trips intersecting the city boundary
    gdf_trips_syn_city = gdf_syn_trips[
        gdf_syn_trips.geometry.intersects(city_geom)
    ].copy()

    # ==============================================================================
    # COMPUTE DISTANCES (same logic as notebook)
    # ==============================================================================

    df_syn_dist = compute_distances_synthetic(
        gdf_trips_syn_city.copy()
    )

    df_act_dist = compute_distances_actual(
        df_act_trips.reset_index(drop=True).copy()
    )

    # ==============================================================================
    # ATTACH HTS WEIGHTS
    # ==============================================================================

    df_hts_for_plotting = df_act_dist.merge(
        df_act_persons[
            ["person_id", "weight_person"]
        ].drop_duplicates("person_id"),
        on="person_id",
        how="left",
    )

    if (
        "purpose" not in df_hts_for_plotting.columns
        and "following_purpose" in df_hts_for_plotting.columns
    ):
        df_hts_for_plotting["purpose"] = (
            df_hts_for_plotting["following_purpose"]
        )

    # ==============================================================================
    # PLOT COMPARISON
    # ==============================================================================

    cdf_title = "Trip_Distance_CDF_(City_Trips_Only)_by_purpose"

    myplottools.plot_comparison_cdf_purpose(
        context,
        cdf_title,
        df_hts_for_plotting,
        df_syn_dist,
        dpi=300,
    )

    dmc_title = "Trip_Distance_CDF_(City_Trips_Only)_by_mode"


    myplottools.plot_comparison_cdf_mode(
        context, 
        dmc_title, 
        df_hts_for_plotting, 
        df_syn_dist,
        dpi = 300, 
    )