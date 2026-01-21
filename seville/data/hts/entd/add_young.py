import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import KDTree
import data.spatial.utils as spatial_utils
import matplotlib.pyplot as plt
from pyproj import Geod
from pyproj import Transformer, Geod

def configure(context):
    context.stage("data.hts.entd.cleaned")
    context.stage("seville.locations.education")

def add_young_persons(df_persons, df_trips, df_household_members):
    """since HTS does not contain any population under age of 15,
    this function aims to extrapolate population under age of 15
    based on the known HTS data
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    # GENERATE PEOPLE
    df_young_people = df_household_members[df_household_members['age']<15]
    df_young_people = pd.merge(df_young_people, df_persons[['person_id', 'household_id', 'person_weight']], on='household_id')
    
    # add home location
    # currently not used, might be in the future
    df_homes = df_trips[df_trips["preceding_purpose"] == "home"].drop_duplicates("person_id")
    df_young_people = pd.merge(df_young_people, df_homes[['person_id', 'origin_location']], on='person_id', how="left")
    df_young_people['home_geometry'] = df_young_people['origin_location']

    # generate new person_id and sex for young persons
    df_young_people = df_young_people.reset_index(drop=True) 
    df_young_people['person_id'] = df_young_people.index
    df_young_people['sex'] = df_young_people['person_id'].apply(lambda x: 'male' if x%2 == 0 else 'female') 
    # person_id negative number to mark it as mock data, avoid duplicate zero
    df_young_people['person_id'] = -df_young_people['person_id'] - 1
        
    df_young_people['age_class'] = pd.cut(
        df_young_people['age'],
        bins=[0, 5, 10, 15],
        labels=[0, 5, 10],
        right=False
    ).astype(float)

    weight_15_19 = df_persons.loc[df_persons['age']<20, 'person_weight'].sum()

    df_young_people['person_weight'] = (
        weight_15_19
        / df_young_people.groupby('age_class')['age_class'].transform('size')
    )

    print(df_young_people.head())

    # add other attributes
    df_young_people['employed'] = False
    df_young_people['studies'] = True
    df_young_people['has_license'] = False
    df_young_people['has_pt_subscription'] = False
    #df_young_people['number_of_trips'] = df_young_people['age'].apply(lambda x: 2 if x >=3 else 0)
    df_young_people['number_of_trips'] = 2
    df_young_people['departement_id'] = "41"
    df_young_people['trip_weight'] = df_young_people['person_weight']
    df_young_people['is_passenger'] = True
    df_young_people['socioprofessional_class'] = 8

    print(df_trips.info())

    return df_young_people


from scipy.spatial import cKDTree

def impute_education_locations(df_young_persons, df_edu_location):

    df = df_young_persons.copy()
    # df = df_young_persons[df_young_persons['age'] >= 3]
    df.dropna(subset=['home_geometry'], inplace=True)

    # Extract home coordinates
    df["home_x"] = df["home_geometry"].apply(lambda p: p[0])
    df["home_y"] = df["home_geometry"].apply(lambda p: p[1])


    # Extract education coordinates
    df_edu = df_edu_location.copy()
    df_edu["edu_x"] = df_edu["geometry"].apply(lambda p: p.x)
    df_edu["edu_y"] = df_edu["geometry"].apply(lambda p: p.y)

    print(df_edu.value_counts(subset='education_type'))
    print(df_edu.info())
    print(df.info())

    # Output columns
    df["edu_geometry"] = None
    df["euclidean_distance"] = np.nan

    age_bounds = [(-np.inf, 6), (7, 16),]
    education_types = [["kindergarten"], ["school"],]

    geod = Geod(ellps="WGS84")

    transformer = Transformer.from_crs(
        "EPSG:25830",
        "EPSG:4326",
        always_xy=True
    )
    df_edu["edu_x"], df_edu["edu_y"] = transformer.transform(
        df_edu["edu_x"].values,
        df_edu["edu_y"].values
    )



    for (lower, upper), types in zip(age_bounds, education_types):
        print(f"{(lower, upper, types)}")

        f_persons = (df["age"] >= lower) & (df["age"] <= upper)
        df_candidates = df_edu[df_edu["education_type"].isin(types)]

        edu_coords = np.column_stack((df_candidates["edu_x"], df_candidates["edu_y"]))
        home_coords = np.column_stack((df.loc[f_persons, "home_x"], df.loc[f_persons, "home_y"]))

        tree = cKDTree(edu_coords)
        _, indices = tree.query(home_coords, k=1)
        nearest = df_candidates.iloc[indices].reset_index(drop=True)

        df.loc[f_persons, "edu_geometry"] = nearest['geometry'].values

        print(df[["home_x","home_y"]].head())
        print(nearest[["edu_x","edu_y"]].head())
        print(df[["home_x","home_y"]].dtypes)

        # VECTORIZED geodesic distance
        lon1 = df.loc[f_persons, "home_x"].values
        lat1 = df.loc[f_persons, "home_y"].values
        edu_lon = nearest["edu_x"].values
        edu_lat = nearest["edu_y"].values

        _, _, dist_m = geod.inv(lon1, lat1, edu_lon, edu_lat)
        df.loc[f_persons, "euclidean_distance"] = dist_m



    df_young_trips = df[['person_id', 'trip_weight', 'euclidean_distance']].copy()
    df_young_trips["mode"] = 'pt'
    df_young_trips["origin_departement_id"] = "41"
    df_young_trips["destination_departement_id"] = "41"

    # 1.3 is routed distance detour factor
    # we assume speed of 15 kms^-1
    # trip duration is in seconds
    df_young_trips['trip_duration'] = df_young_trips['euclidean_distance'] * 1.3 / 15 * 3.6

    df_to_school_trips = df_young_trips.copy()
    df_to_school_trips["trip_id"] = df_to_school_trips['person_id'].astype(str) + '_1'
    df_to_school_trips['departure_time'] = 9*3600 - df_to_school_trips['trip_duration']
    df_to_school_trips['arrival_time'] = 9*3600
    df_to_school_trips['activity_duration'] = 15*3600 - 9*3600
    df_to_school_trips['following_purpose'] = "education"
    df_to_school_trips['preceding_purpose'] = "home"
    df_to_school_trips['is_first_trip'] = True
    df_to_school_trips['is_last_trip'] = False

    df_from_school_trips = df_young_trips.copy()
    df_from_school_trips["trip_id"] = df_to_school_trips['person_id'].astype(str) + '_2'
    df_from_school_trips['departure_time'] = 15*3600
    df_from_school_trips['arrival_time'] = 15*3600 + df_to_school_trips['trip_duration']
    df_from_school_trips['activity_duration'] = np.nan
    df_from_school_trips['following_purpose'] = "home"
    df_from_school_trips['preceding_purpose'] = "education"
    df_from_school_trips['is_first_trip'] = False
    df_from_school_trips['is_last_trip'] = True

    print(df_young_trips['euclidean_distance'].describe())
    print(len(df_young_trips[df_young_trips['euclidean_distance'] > 5000]))
    print(df_young_trips.loc[df_young_trips['euclidean_distance'] > 5000, 'euclidean_distance'])

    df_result = pd.concat([df_to_school_trips, df_from_school_trips])
    df_result = df_result.sort_values(by = ["person_id", "trip_id"])
    
    return df_result


def execute(context):
    df_households, df_persons, df_trips, df_household_members = context.stage("data.hts.entd.cleaned")

    df_young_persons = add_young_persons(df_persons, df_trips, df_household_members)
    df_persons = pd.concat([df_persons, df_young_persons[df_persons.columns]])

    # add education trips your young persons
    df_edu_locations = context.stage("seville.locations.education")
    df_young_trips = impute_education_locations(df_young_persons, df_edu_locations)    
    df_trips = pd.concat([df_trips, df_young_trips])

    return df_households, df_persons, df_trips
