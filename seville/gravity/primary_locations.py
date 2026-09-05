import numpy as np
import pandas as pd
import geopandas as gpd
from .work_candidates import EDUCATION_MAPPING
from scipy.spatial import cKDTree

"""
This stage assigns primary locations (work and education) to individuals 
in the population based on their home locations and commute distances.
 
It uses a K-nearest neighbors approach for school assignments, taking 
into account both distance and school capacity. University assignments are 
made using weighted random selection based on available university locations.
"""


def configure(context):
    context.stage("synthesis.population.spatial.primary.candidates")
    context.stage("synthesis.population.spatial.commute_distance")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.locations.work")
    context.stage("synthesis.locations.education")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")

    context.config("education_graduation_age")

def assign_schools_with_capacity_weighting(people, schools, school_type, random, k_nearest=5, distance_decay_power=2.0):
    """
    Assign students to schools using K-nearest schools with distance and capacity weighting.
    
    Parameters:
    - people: DataFrame with person_id and home_location geometry
    - schools: GeoDataFrame with location_id, commune_id, geometry, and capacity columns
    - school_type: string for logging
    - random: numpy RandomState
    - k_nearest: number of nearest schools to consider for each student
    - distance_decay_power: power for distance decay (higher = stronger preference for closer schools)
    
    Returns:
    - DataFrame with assigned person_id, commune_id, location_id, geometry
    """
    
    # Build KD-tree of school locations
    school_xy = np.column_stack([
        schools.geometry.x.values,
        schools.geometry.y.values
    ])
    tree = cKDTree(school_xy)
    
    schools["capacity"] = schools["students"]

    # Initialize remaining capacity for each school
    remaining_capacity = schools["capacity"].copy().reset_index(drop=True)
    
    assignments = []
    person_xy = np.array([(p.x, p.y) for p in people["home_location"]])
    
    # Query K nearest schools for each person
    k = min(k_nearest, len(schools))
    distances, indices = tree.query(person_xy, k=k)
    
    for person_idx, (person_distances, person_indices) in enumerate(zip(distances, indices)):
        # Handle case where k=1 (distances/indices are scalars)
        if k == 1:
            person_distances = np.array([person_distances])
            person_indices = np.array([person_indices])
        
        # Filter to schools with remaining capacity
        available_mask = remaining_capacity.iloc[person_indices].values > 0
        available_indices = person_indices[available_mask]
        available_distances = person_distances[available_mask]
        
        if len(available_indices) == 0:
            # No schools with capacity - assign to closest school regardless
            print(f"[WARNING] No schools with capacity for {school_type} student {person_idx}. Assigning to nearest.")
            assigned_idx = person_indices[0]
        else:
            # Calculate weights: inverse distance (closer = higher) * remaining capacity
            # Avoid division by zero
            available_distances = np.maximum(available_distances, 1.0)
            distance_weights = 1.0 / (available_distances ** distance_decay_power)
            capacity_weights = remaining_capacity.iloc[available_indices].values
            combined_weights = distance_weights * capacity_weights
            
            # Normalize to probabilities
            combined_weights = combined_weights / combined_weights.sum()
            
            # Randomly select school based on weights
            assigned_idx = random.choice(available_indices, p=combined_weights)
        
        # Record assignment
        school_row = schools.iloc[assigned_idx]
        assignments.append({
            "person_id": people.iloc[person_idx]["person_id"],
            "commune_id": school_row["commune_id"],
            "location_id": school_row["location_id"],
            "geometry": school_row["geometry"]
        })
        
        # Decrement capacity
        remaining_capacity.iloc[assigned_idx] -= 1
    
    return pd.DataFrame(assignments)


def define_distance_ordering(df_persons, df_candidates, progress):
    indices = []

    f_available = np.ones((len(df_candidates),), dtype = bool)
    costs = np.ones((len(df_candidates),)) * np.inf

    commute_coordinates = np.vstack([
        df_candidates["geometry"].x.values,
        df_candidates["geometry"].y.values
    ]).T

    for home_coordinate, commute_distance in zip(df_persons["home_location"], df_persons["commute_distance"]):
        home_coordinate = np.array([home_coordinate.x, home_coordinate.y])
        distances = np.sqrt(np.sum((commute_coordinates[f_available] - home_coordinate)**2, axis = 1))
        costs[f_available] = np.abs(distances - commute_distance)

        selected_index = np.argmin(costs)
        indices.append(selected_index)
        f_available[selected_index] = False
        costs[selected_index] = np.inf

        progress.update()

    assert len(set(indices)) == len(df_candidates)

    return indices

def define_random_ordering(df_persons, df_candidates, progress):
    progress.update(len(df_candidates))
    return np.arange(len(df_candidates))

define_ordering = define_distance_ordering

def process_municipality(context, origin_id):
    # Load data
    df_candidates, df_persons = context.data("df_candidates"), context.data("df_persons")

    # Find relevant records
    df_persons = df_persons[df_persons["commune_id"] == origin_id][[
        "person_id", "home_location", "commute_distance"
    ]].copy()
    df_candidates = df_candidates[df_candidates["origin_id"] == origin_id]

    # From previous step, this should be equal!
    assert len(df_persons) == len(df_candidates)

    indices = define_ordering(df_persons, df_candidates, context.progress)
    df_candidates = df_candidates.iloc[indices]

    df_candidates["person_id"] = df_persons["person_id"].values
    df_candidates = df_candidates.rename(columns = dict(destination_id = "commune_id"))

    return df_candidates[["person_id", "commune_id", "location_id", "geometry"]]

def process(context, purpose, df_persons, df_candidates):
    unique_ids = df_candidates["origin_id"].unique()

    df_result = []

    with context.progress(label = "Distributing %s destinations" % purpose, total = len(df_persons)) as progress:
        with context.parallel(dict(df_persons = df_persons, df_candidates = df_candidates)) as parallel:
            for df_partial in parallel.imap_unordered(process_municipality, unique_ids):
                df_result.append(df_partial)

    return pd.concat(df_result).sort_index()



def process_work_locations(context):

    df_work_candidates, df_work = context.stage("synthesis.population.spatial.primary.candidates")

    # Attach home locations
    df_home = context.stage("synthesis.population.spatial.home.locations")

    df_work = pd.merge(df_work, df_home[["household_id", "geometry"]].rename(columns = {
        "geometry": "home_location"
    }), how = "left", on = "household_id")

    # Attach commute distances
    df_commute_distance = context.stage("synthesis.population.spatial.commute_distance")

    df_work = pd.merge(df_work, df_commute_distance["work"], how = "left", on = "person_id")

    # Attach geometry
    df_locations = context.stage("synthesis.locations.work")[["location_id", "geometry"]]
    df_work_candidates = pd.merge(df_work_candidates, df_locations, how = "left", on = "location_id")
    df_work_candidates = gpd.GeoDataFrame(df_work_candidates)

    # Assign destinations
    df_work = process(context, "work", df_work, df_work_candidates)
    
    return df_work


def process_edu_locations(context):
    random = np.random.RandomState(context.config("random_seed"))
    gdf_education = context.stage("synthesis.locations.education")

    school_weights = gdf_education[gdf_education["education_type"]!="university"].copy()
    university_weights = gdf_education[gdf_education["education_type"]=="university"].copy()

    # get people with education trips
    df_trips = context.stage("synthesis.population.trips")
    df_persons = context.stage("synthesis.population.enriched")[["person_id", "household_id", "age"]].copy()
    
    # Identify people with education trips
    education_trip_persons = df_trips[
        (df_trips["following_purpose"] == "education") | (df_trips["preceding_purpose"] == "education")
    ]["person_id"].unique()
    
    df_persons["has_education_trip"] = df_persons["person_id"].isin(education_trip_persons)
    
    # Count by education type before filtering
    df_all_persons = df_persons.copy()
    df_persons = df_persons[df_persons["has_education_trip"] == True]
    
    # Log education eligibility
    graduation_age = context.config("education_graduation_age")
    eligible_by_type = {
        "kindergarten": len(df_all_persons[df_all_persons["age"] <= graduation_age["kindergarten"]]),
        "elementary": len(df_all_persons[(df_all_persons["age"] > graduation_age["kindergarten"]) & (df_all_persons["age"] <= graduation_age["elementary"])]),
        "highschool": len(df_all_persons[(df_all_persons["age"] > graduation_age["elementary"]) & (df_all_persons["age"] <= graduation_age["highschool"])]),
        "university": len(df_all_persons[df_all_persons["age"] > graduation_age["highschool"]])
    }
    
    with_trips_by_type = {
        "kindergarten": len(df_persons[df_persons["age"] <= graduation_age["kindergarten"]]),
        "elementary": len(df_persons[(df_persons["age"] > graduation_age["kindergarten"]) & (df_persons["age"] <= graduation_age["elementary"])]),
        "highschool": len(df_persons[(df_persons["age"] > graduation_age["elementary"]) & (df_persons["age"] <= graduation_age["highschool"])]),
        "university": len(df_persons[df_persons["age"] > graduation_age["highschool"]])
    }
    
    print(f"[EDUCATION ASSIGNMENT] Eligible by type: {eligible_by_type}")
    print(f"[EDUCATION ASSIGNMENT] With education trips: {with_trips_by_type}")

    # Attach home locations
    df_home = context.stage("synthesis.population.spatial.home.locations")

    df_persons = pd.merge(df_persons, df_home[["household_id", "geometry"]].rename(columns = {
        "geometry": "home_location"
    }), how = "left", on = "household_id")


    # ====================== SCHOOLS =========================
    graduation_age = context.config("education_graduation_age")

    # Only school trips
    df_school = df_persons[df_persons["age"] <= graduation_age["highschool"]].copy()
    df_school["school_type"] = "highschool"
    df_school.loc[df_school["age"] <= graduation_age["elementary"], "school_type"] = "elementary"
    df_school.loc[df_school["age"] <= graduation_age["kindergarten"], "school_type"] = "kindergarten"

    assignments = []

    for school_type in ["kindergarten", "elementary", "highschool"]:

        people = df_school[df_school["school_type"] == school_type].copy()
        schools = school_weights[school_weights["education_type"] == school_type].copy()

        if len(people) == 0 or len(schools) == 0:
            print(f"[EDUCATION ASSIGNMENT] Skipping {school_type}: {len(people)} people, {len(schools)} schools")
            continue

        print(f"[EDUCATION ASSIGNMENT] Assigning {len(people)} {school_type} students to {len(schools)} schools using capacity-weighted K-nearest")

        # Use capacity-weighted K-nearest assignment
        df_assigned = assign_schools_with_capacity_weighting(
            people, 
            schools, 
            school_type, 
            random,
            k_nearest=5,
            distance_decay_power=2.0
        )

        assignments.append(df_assigned)

    if len(assignments) > 0:
        df_school_locations = pd.concat(assignments, ignore_index=True)
    else:
        df_school_locations = pd.DataFrame(columns=["person_id", "commune_id", "location_id", "geometry"])
        raise(Exception("[ERROR] School assignment failed! No students assigned to school locations."))

    # ======================= UNIVERSITIES ===================
    df_university_people = df_persons[df_persons["age"] > graduation_age["highschool"]].copy()
    
    if len(df_university_people) > 0:
        # Ensure unique location_ids and valid weights
        university_weights = university_weights[["location_name", "location_id", "commune_id", "geometry", "weight"]].copy()
        university_weights = university_weights.drop_duplicates(subset=["location_id"], keep="first")
        
        # Remove rows with NaN or invalid weights
        assert len(university_weights.dropna(subset=["weight", "location_id", "commune_id"])) == len(university_weights)
        
        # Validate that we have universities to assign
        if len(university_weights) == 0:
            print("[WARNING] No valid university locations found for assignment!")
            df_university_people = pd.DataFrame(columns=["person_id", "commune_id", "location_id", "geometry"])
            raise(Exception("[ERROR] University assignment failed! No valid university locations available."))
        else:
            # Normalize weights
            university_weights["weight"] = university_weights["weight"] / university_weights["weight"].sum()
            
            print(f"[EDUCATION ASSIGNMENT] University locations available: {len(university_weights)}")
            print(f"[EDUCATION ASSIGNMENT] University students to assign: {len(df_university_people)}")
            
            # Assign location_ids using weighted random choice
            df_university_people["location_id"] = random.choice(
                university_weights["location_id"].values,
                size=len(df_university_people),
                p=university_weights["weight"].values
            )

            # Merge with location details
            df_university_before_merge = len(df_university_people)
            df_university_people = df_university_people.merge(
                university_weights[
                    ["location_id", "commune_id", "geometry"]
                ],
                on="location_id",
                how="left"
            )
            df_university_after_merge = len(df_university_people)
            
            # Check for data loss
            if df_university_before_merge != df_university_after_merge:
                print(f"[ERROR] University merge lost {df_university_before_merge - df_university_after_merge} records!")
                raise(Exception("[ERROR] University assignment failed! Merge with university_weights resulted in data loss."))

            # Remove any rows with NaN geometry (merge failures)
            lost_rows = df_university_people["geometry"].isna().sum()
            if lost_rows > 0:
                print(f"[WARNING] {lost_rows} university students have invalid geometry after merge")
                df_university_people = df_university_people.dropna(subset=["geometry"])
                raise(Exception("[ERROR] University assignment failed! Some students could not be assigned to valid locations."))
            
            # Verify assignment
            if len(df_university_people) > 0:
                print(f"[EDUCATION ASSIGNMENT] University students successfully assigned: {len(df_university_people)}")
                df_university_people = df_university_people[["person_id", "commune_id", "location_id", "geometry"]]
            else:
                print("[ERROR] No university students assigned!")
                df_university_people = pd.DataFrame(columns=["person_id", "commune_id", "location_id", "geometry"])
                raise(Exception("[ERROR] University assignment failed! No students assigned to university locations."))
    else:
        df_university_people = pd.DataFrame(columns=["person_id", "commune_id", "location_id", "geometry"])
        raise(Exception("[ERROR] University assignment failed! No students eligible for university assignment."))
    # ========================== MERGE =========================

    df_education = pd.concat(
        [df_school_locations, df_university_people],
        ignore_index=True
    )


    return df_education

def execute(context):

    df_work = process_work_locations(context)
    df_education = process_edu_locations(context)

    return df_work, df_education
