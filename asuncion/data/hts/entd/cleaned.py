from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts
from geopy.distance import geodesic

"""
This stage cleans the national HTS.
"""

def configure(context):
    context.stage("data.hts.entd.raw")


INCOME_CLASS_BOUNDS_SEVILLE = [1000, 1500, 2000, 3000, 4000, 5000, 1e6]


PURPOSE_MAP = {
    1: "home", # Regreso al Hogar
    2: "work", # Trabajo
    3: "leisure", # Entretenimiento y Ocio - Ir a Comer
    4: "other", # 'Dejar - Recoger - Acompañar a Alguién
    5: "shop", # Compras
    6: "other", # 'Asistencia Médica o Dental - Trámites Personales
    7: "education", # Estudios
    8: "other", # 'Visitar a Alguién
    9: "other", # 'Actividad Política -Sindical - Comunitaria - Religiosa - Otro
    10: "work", # 'Trámites de Trabajo
}


MODES_MAP = {
    "Bus":"pt", #1
    "Auto":"car", #2
    "Moto":"car", #3
    "A Pie":"walk", #4
    "Bicicleta":"bike", #5
    "Taxi/Remis/Uber/MUV/BOLT":"drt", #6
    "Bus Escolar/Empresa":"pt", #7
    "???":"???", #8
    "A Pie Corto":"walk", #9
}



def execute(context):
    df_persons, df_households, df_trips, df_legs = context.stage("data.hts.entd.raw")


    # Make copies
    df_persons = df_persons.copy()
    df_households = df_households.copy()
    df_trips = df_trips.copy()
    df_legs = df_legs.copy()

    # ========== Clean Household Data ==========
    
    # Set type as int for all numerical columns
    df_households["household_id"] = df_households["household_id"].astype(int)
    df_households["household_size"] = df_households["household_size"].astype(int)
    df_households["departement_id"] = df_households["departement_id"].astype(str)
    df_households["income_class"] = df_households["income_class"].astype(int)
    df_households["number_of_bikes"] = df_households["number_of_bikes"].astype(int)

    df_households["household_weight"] = df_households["household_weight"].astype(float)
    
    df_households["number_of_vehicles"] = (
        df_households["number_of_vehicles1"].astype(int)
        + df_households["number_of_vehicles2"].astype(int)
        + df_households["number_of_vehicles3"].astype(int)
        + df_households["number_of_vehicles4"].astype(int)
        + df_households["number_of_vehicles5"].astype(int)
        + df_households["number_of_vehicles6"].astype(int)
    )


    # Clean urban type
    df_households['urban_type'] = "urban"
    df_households["urban_type"] = df_households["urban_type"].astype("category")


    # ========== Clean Persons Data ==========


    # Transform original IDs to integer
    df_persons["person_id"] = df_persons["person_id"].astype(int)
    df_persons["household_id"] = df_persons["household_id"].astype(int)
    df_persons["departement_id"] = df_persons["departement_id"].astype(str)

    df_persons["person_weight"] = df_persons["person_weight"].astype(float)
    df_persons["trip_weight"] = df_persons["trip_weight"].astype(float)

    df_persons["age"] = df_persons["age"].astype(int)
    df_persons["number_of_trips"] = df_persons["number_of_trips"].astype(int)

    PERSONS_SEX_MAP = {
        "Hombre": "male",
        "Mujer": "female"
    }
    df_persons["sex"] = df_persons["sex"].map(PERSONS_SEX_MAP).astype("category")

    def clean_employed(x):
        if x == "":
            return False
        if x == "0":
            return False
        if x == "99999999999":
            return False
        return True
    
    df_persons["employed"] = df_persons["employed"].apply(clean_employed)

    # 1 - has license
    # 2 - yes, but it is expired
    # 3 - no
    df_persons["has_license"] = df_persons["has_license"] == "1"

    # Has pt subscription
    df_persons["has_pt_subscription"] = np.nan # we have no subscription data

    # ========== Clean Trips Data ==========

    df_trips["person_id"] = df_trips["person_id"].astype(int)
    df_trips["trip_id"] = df_trips["trip_id"].astype(int)
    df_trips["trip_weight"] = df_trips["trip_weight"].astype(float)

    df_trips.loc[df_trips["origin_district_id"] == "0" ,"origin_district_id"] = "0000"
    df_trips.loc[df_trips["destination_district_id"] == "0" ,"destination_district_id"] = "0000"

    df_trips["origin_departement_id"] = df_trips["origin_district_id"].str[:2] # 11 is central departement
    df_trips["destination_departement_id"] = df_trips["destination_district_id"].str[:2]

    # Trip purpose
    df_trips["following_purpose"] = df_trips["following_purpose"].astype(int).map(PURPOSE_MAP).astype("category")
    df_trips["preceding_purpose"] = df_trips["preceding_purpose"].astype(int).map(PURPOSE_MAP).astype("category")

    # Trip mode
    df_trips['mode'] = df_trips['mode'].map(MODES_MAP)


    
    # calculate trip distances from legs
    df_trips = calculate_distance(df_trips, df_legs)

    # Fix passenger
    passenger_trip_ids = (
        df_legs[
            df_legs["mode"].str.contains("passenger", case=False, na=False)
        ]["trip_id"]
        .unique()
    )
    mask = (df_trips["mode"].eq("car") & df_trips["trip_id"].isin(passenger_trip_ids))

    df_trips.loc[mask, "mode"] = "car_passenger"
    print(df_trips['euclidean_distance'])


    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Trip times are in format hours[.min] where .min part is optional
    def convert_time(x):
        time = x.split(',')
        hours = int(time[0])
        # if there is no minute part, minutes == 0
        minutes = '00' if len(time) == 1 else time[1]
        # if there is only single digit (e.g. 10.2) those are tens of minuts 
        minutes = int(minutes if len(minutes) == 2 else minutes + '0')

        return hours * 3600 + minutes * 60

    df_trips["departure_time"] = df_trips["departure_time"].apply(convert_time).astype(float) # in seconds
    df_trips["trip_duration"] = df_trips["trip_duration"].astype(float) * 60 # minutes => seconds in seconds
    df_trips["arrival_time"] = df_trips["arrival_time"].apply(convert_time).astype(float)
    df_trips = hts.fix_trip_times(df_trips)


    # ========== Other Attributes ==========

    # Calculate consumption units    
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on = "household_id")

    # Durations
    hts.compute_activity_duration(df_trips)
      
    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(
        df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
    )
    
    # Fix activity types (because of 1 inconsistent ENTD data)
    hts.fix_activity_types(df_trips)
    

    # Get shifted departure_time as a new Series
    next_departure_time = df_trips["departure_time"].shift(-1)

    # Identify problematic rows: not last trip & arrival after next departure
    f = (~df_trips["is_last_trip"]) & (df_trips["arrival_time"] > next_departure_time)
    # Get unique person_ids with such invalid trips    
    problematic_ids = df_trips.loc[f, "person_id"].unique()
    print(f"Invalidating {len(problematic_ids)} persons with arrival_time > next departure_time for trip matching")
    # Delete from df_trips and df_persons
    df_trips['is_valid'] &= ~df_trips["person_id"].isin(problematic_ids)
    
    
    print(len(set(df_persons["person_id"].values) - set(df_trips["person_id"].values)), "raw number of persons without trips")



    # Invalidate Trips with no legs
    trips_with_legs = df_legs['trip_id'].unique()
    df_trips.loc[~(df_trips["trip_id"].isin(trips_with_legs)), "is_valid"] = False


    # set number of trips -1 for all persons that have at least 1 invalid trip
    # to exclude them from trip matching

    # Trips that are invalid OR contain any NaN
    invalid_trips = (~df_trips["is_valid"])
    # Person IDs with at least one invalid trip
    invalid_persons = df_persons["person_id"].isin(df_trips.loc[invalid_trips, "person_id"].unique())
    
    # Set number_of_trips = -1 for those persons
    df_persons.loc[invalid_persons, "number_of_trips"] = -1


    print(f"[INFO] in total invalid {len(df_trips[invalid_trips])} trips from total of {len(df_trips)}")
    print("[INFO] we remove all the trips of person having even single invalid trips to avoid")

    print(f"[INFO] in total we remove trips for {len(df_persons[df_persons['number_of_trips'] == -1])} persons")
    df_trips = df_trips[df_trips['person_id'].isin(df_persons.loc[df_persons['number_of_trips'] != -1, 'person_id' ])]



    return df_households, df_persons, df_trips



def calculate_distance(df_trips, df_legs):
    df_legs["trip_id"] = df_legs["trip_id"].astype(int)
    df_legs["routed_distance"] = df_legs["routed_distance"].str.strip()
    df_legs["euclidean_distance"] = df_legs["euclidean_distance"].str.strip()

    df_legs.loc[df_legs["routed_distance"] == "", "routed_distance"] = np.nan
    df_legs.loc[df_legs["euclidean_distance"] == "", "euclidean_distance"] = np.nan


    df_legs["routed_distance"] = (
        df_legs["routed_distance"].str.replace(",", ".").astype(float) * 1000
    )
    df_legs["euclidean_distance"] = (
        df_legs["euclidean_distance"].str.replace(",", ".").astype(float) * 1000
    )

    # Trips with at least one leg missing either distance
    invalid_trip_ids = df_legs.loc[
        df_legs[["routed_distance", "euclidean_distance"]].isna().any(axis=1),
        "trip_id",
    ].unique()

    df_trips.loc[df_trips["trip_id"].isin(invalid_trip_ids), "is_valid"] = False


    # Aggregate distances
    trip_distances = (
        df_legs.groupby("trip_id", as_index=False)
               .agg(
                   routed_distance=("routed_distance", "sum"),
                   euclidean_distance=("euclidean_distance", "sum"),
               )
    )

    df_trips = df_trips.merge(
        trip_distances[["trip_id", "routed_distance", "euclidean_distance"]],
        on="trip_id",
        how="left",
    )
    return df_trips