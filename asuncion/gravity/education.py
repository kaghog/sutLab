import numpy as np
import pandas as pd
import geopandas as gpd
from .work_candidates import EDUCATION_MAPPING
from scipy.spatial import cKDTree

"""
DESCRIPTION:

"""


def configure(context):
    context.stage("synthesis.population.spatial.commute_distance")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.locations.education")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")

    context.config("education_graduation_age")



def process_edu_locations(context):
    random = np.random.RandomState(context.config("random_seed"))
    gdf_education = context.stage("synthesis.locations.education")

    school_weights = gdf_education[gdf_education["education_type"]!="university"].copy()
    university_weights = gdf_education[gdf_education["education_type"]=="university"].copy()

    # get people with education trips
    df_trips = context.stage("synthesis.population.trips")
    df_persons = context.stage("synthesis.population.enriched")[["person_id", "household_id", "age"]].copy()
    df_persons["has_education_trip"] = df_persons["person_id"].isin(df_trips[
        (df_trips["following_purpose"] == "education") | (df_trips["preceding_purpose"] == "education")
    ]["person_id"])
    df_persons = df_persons[df_persons["has_education_trip"] == True]


    # Attach home locations
    df_home = context.stage("synthesis.population.spatial.home.locations")

    df_persons = pd.merge(df_persons, df_home[["household_id", "geometry"]].rename(columns = {
        "geometry": "home_location"
    }), how = "left", on = "household_id")


    # ====================== SCHOOLS =========================
    graduation_age = context.config("education_graduation_age")

    assignments = []

    # Sort school levels by their maximum age, youngest -> oldest
    school_levels = sorted(
        graduation_age.items(),
        key=lambda x: x[1]
    )

    for i, (school_type, max_age) in enumerate(school_levels):

        # Lower bound is the previous age threshold + 1
        min_age = 0 if i == 0 else school_levels[i - 1][1] + 1

        people = df_persons[
            df_persons["age"].between(min_age, max_age)
        ].copy()

        schools = school_weights[
            school_weights["education_type"] == school_type
        ].copy()

        if len(people) == 0 or len(schools) == 0:
            continue

        # Build KD-tree of school locations
        school_xy = np.column_stack([
            schools.geometry.x.values,
            schools.geometry.y.values
        ])
        tree = cKDTree(school_xy)

        # Query nearest school for every person
        person_xy = np.array([
            (p.x, p.y) for p in people["home_location"]
        ])

        _, idx = tree.query(person_xy, k=1)

        people["commune_id"] = schools.iloc[idx]["commune_id"].values
        people["location_id"] = schools.iloc[idx]["location_id"].values
        people["geometry"] = schools.iloc[idx]["geometry"].values

        assignments.append(
            people[["person_id", "commune_id", "location_id", "geometry"]]
        )

    df_school_locations = pd.concat(assignments, ignore_index=True)

    # ======================= UNIVERSITIES ===================
    # university using university_od stuff
    university_weights["weight"] = university_weights["weight"] / university_weights["weight"].sum()
    
    # select university students
    max_school_age = max(graduation_age.values())
    df_university_people = df_persons[df_persons["age"] > max_school_age].copy()

    # assign university locations
    df_university_people["location_id"] = np.random.choice(
        university_weights["location_id"],
        size=len(df_university_people),
        p=university_weights["weight"]
    )

    df_university_people = df_university_people.merge(
        university_weights[
            ["location_id", "commune_id", "geometry"]
        ],
        on="location_id",
        how="left"
    )
    df_university_people = df_university_people[["person_id", "commune_id", "location_id", "geometry"]]

    # ========================== MERGE =========================

    df_education = pd.concat(
        [df_school_locations, df_university_people],
        ignore_index=True
    )


    return df_education

def execute(context):

    df_education = process_edu_locations(context)

    return df_education
