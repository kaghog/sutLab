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
    context.stage("seville.data.hts.entd.streets.verify")



INCOME_CLASS_BOUNDS_SEVILLE = [1000, 1500, 2000, 3000, 4000, 5000, 1e6]

    # Casa	1	Casa
    # Trabajo	2	Trabajo
    # Gestiones trabajo	3	Gestiones trabajo
    # Estudios	4	Estudios
    # Médico	5	Médico
    # Compra diaria	6	Compra diaria
    # Compra no diaria	7	Compra no diaria
    # Asuntos personales	8	Asuntos personales
    # Ocio	9	Ocio
    # Llevar/recoger a un acompañante	10	Llevar/recoger a un acompañante
    # Visita a familiares o amigos	11	Visita a familiares o amigos
    # Otros:	12	Otros:


PURPOSE_MAP = {
    1: "home", # Casa
    2: "work", # Trabajo
    3: "work", # Gestiones trabajo
    4: "education", # Estudios
    5: "other", # Médico
    6: "shop", # Compra diaria
    7: "shop", # Compra no diaria
    8: "other", # Asuntos personales
    9: "leisure", # Ocio
    10: "other", # Llevar/recoger a un acompañante - pickup friend??? TODO:
    11: "other", # Visita a familiares o amigos
    12: "other", # unknown
}

# Spanish travel mode:
#    1 A pie (Al menos 5 minutos caminando)	1
#    2 Coche Conductor	2
#    3 Coche Pasajero	3
#    4 Bus urbano TUSSAM	4
#    5 Bus interurbano Consorcio Tte.Área Sevilla	5
#    6 Bus discrecional (empresa, escolar)	6
#    7 Metro	7
#    8 Metrocentro. Tranvía Sevilla	8
#    9 Moto	9
#    10 Bicicleta privada	10
#    11 Bicicleta pública. SEVICI	11
#    12 Cercanías	12
#    13 Tren Regional	13
#    14 Tren Largo Recorrido	14
#    15 Taxi	15
#    16 Other 16

MODES_BASIC_MAP = {
    "-": 0, # "-" marks that the trip does not have phase of corresponding number
    1: 1,
    2: 4,
    3: 5,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 4,
    10: 3,
    11: 3,
    12: 2,
    13: 2,
    14: 2,
    15: 4,
    16: 6, 
}

MODES_MAP = {
    1: "walk",
    2: "pt",
    3: "bike",
    4: "car",
    5: "car_passenger",
    6: "pt" # Other assume pt
}


def convert_time(x):
    return float(x.hour * 3600 + x.minute * 60 + x.second)

def execute(context):
    df_persons, df_households, df_trips, df_household_members = context.stage("data.hts.entd.raw")


    # Make copies
    df_persons = pd.DataFrame(df_persons, copy = True)
    df_households = pd.DataFrame(df_households, copy = True)
    df_trips = pd.DataFrame(df_trips, copy = True)
    df_household_members = pd.DataFrame(df_household_members, copy = True)

    # Transform original IDs to integer (they are hierarchichal)
    df_persons["person_id"] = df_persons["person_id"].astype(int)
    df_persons["household_id"] = df_persons["household_id"].astype(int)
    df_households["household_id"] = df_households["household_id"].astype(int)
    df_trips["person_id"] = df_trips["person_id"].astype(int)

    # Set the weight as float
    df_persons["person_weight"] = df_persons["person_weight"].astype(float)
    df_households["household_weight"] = df_households["household_weight"].astype(float)
    df_trips["trip_weight"] = df_trips["trip_weight"].astype(float)

    # Clean household member count
    df_households["household_size"] = (
        df_households["household_size"]
        .astype(str)
        .str.split(" ", n=1)
        .str[0]
    ).astype(int)

    # Clean sex
    df_persons["sex"] = df_persons["sex"].astype("str")
    df_persons.loc[df_persons["sex"] == "1", "sex"] = "male"
    df_persons.loc[df_persons["sex"] == "2", "sex"] = "female"
    df_persons["sex"] = df_persons["sex"].astype("category")

    # Clean departement
    df_households["departement_id"] = df_households["departement_id"].astype("category")
    df_persons["departement_id"] = df_persons["departement_id"].astype("category")

    df_trips["origin_departement_id"] = df_trips["origin_departement_id"].astype("category")
    df_trips["destination_departement_id"] = df_trips["destination_departement_id"].astype("category")

    # Clean urban type
    df_households['urban_type'] = "central_city"
    df_households["urban_type"] = df_households["urban_type"].astype("category")

    # -------------------------------------------------------------------------------------
    # Spanish SITLAB values:
        # 1 Employed
        # 2 Employed and student
        # 3 Retired / pensionista / invalid / jubilado
        # 4 Unemployed
        # 5 Unemployed
        # 6 Student
        # 7 Househusband, Housewife
        # 8 Other
        # 99 No data

    # Map education
    df_persons["studies"] = df_persons["employed"].isin([2, 6])

    # Map work situation
    df_persons["employed" ] = df_persons["employed"].isin([1, 2])

    # License
    df_persons["has_license"] = (df_persons["has_license"] == 1)

    # Has subscription
    df_persons["has_pt_subscription"] = np.nan # we have no subscription data

    # Household income
    # Spanish income groups:
    #   1 Menos de 12.000€ brutos/año	1       |     0 - 1.000
    #   2 Entre 12.000 y 18.000€ brutos/año	2   | 1.000 - 1.500
    #   3 Entre 18.000 y 24.000€ brutos/año	3   | 1.500 - 2.000
    #   4 Entre 24.000 y 36.000€ brutos/año	4   | 2.000 - 3.000
    #   5 Entre 36.000 y 48.000€ brutos/año	    | 3.000 - 4.000
    #   6 Entre 48.000 y 60.000 € brutos/año    | 4.000 - 5.000
    #   7 Más de 60.000 €brutos/año             | 5.000 or more per month
    #   99 No contesta

    #  The mapping of the bins must be changed even further in the pipeline
    df_households.loc[df_households["income_class"] == 99, "income_class"] = -1 # no values
    df_households.loc[df_households["income_class"] == ' ', "income_class"] = -1 # no values
    df_households["income_class"] = df_households["income_class"].astype(int)


    # Trip purpose
    df_trips["following_purpose"] = df_trips["following_purpose"].map(PURPOSE_MAP).astype("category")
    df_trips["preceding_purpose"] = df_trips["preceding_purpose"].map(PURPOSE_MAP).astype("category")

    # Trip mode
    df_trips = aggregate_transport_mode(df_trips)

    # Trip distance
    df_trips = calculate_trip_distance(context, df_trips)

    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Trip times
    df_trips["departure_time"] = df_trips["departure_time"].apply(convert_time).astype(float) # in seconds
    df_trips["trip_duration"] = df_trips["trip_duration"].astype(float) * 60 # minutes => seconds in seconds
    df_trips["arrival_time"] = df_trips["departure_time"] + df_trips["trip_duration"]
    df_trips = hts.fix_trip_times(df_trips)

    # Durations
    hts.compute_activity_duration(df_trips)

    # Number of trips
    df_persons = pd.merge(
        df_persons, df_trips[["person_id", "number_of_trips"]].drop_duplicates("person_id"),
        on = "person_id", how = "left"
    )
    df_persons["number_of_trips"] = df_persons["number_of_trips"].fillna(-1).astype(int)
    df_persons.loc[(df_persons["number_of_trips"] == -1), "number_of_trips"] = 0
      
    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(
        df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
    )
    

    # Clean household members data   
    df_household_members = pd.wide_to_long(
        df_household_members,
        stubnames=["age", "sex"],
        i="household_id",
        j="person",
        sep="",
        suffix="\d+"
    ).reset_index()
    # Drop rows with NaN values (empty ages)
    df_household_members = df_household_members.dropna(subset=["age"])
    df_household_members = df_household_members[df_household_members['age']!='-']
    df_household_members.reset_index(drop=True, inplace=True)

    df_household_members["sex"] = df_household_members["sex"].astype("str")
    df_household_members.loc[df_household_members["sex"] == "1", "sex"] = "male"
    df_household_members.loc[df_household_members["sex"] == "2", "sex"] = "female"
    df_household_members["sex"] = df_household_members["sex"].astype("category")


    # Calculate consumption units    
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_household_members), on = "household_id")

    # Socioprofessional class
    df_persons["socioprofessional_class"] = df_persons["socioprofessional_class"].fillna(80).astype(int) // 10

    # Fix activity types (because of 1 inconsistent ENTD data)
    hts.fix_activity_types(df_trips)
    
    df_persons["person_id"] = df_persons["person_id"].astype(int)
    df_trips["person_id"] = df_trips["person_id"].astype(int)

    # Get shifted departure_time as a new Series
    next_departure_time = df_trips["departure_time"].shift(-1)
    # Identify problematic rows: not last trip & arrival after next departure
    f = (~df_trips["is_last_trip"]) & (df_trips["arrival_time"] > next_departure_time)
    # Get unique person_ids with such invalid trips
    problematic_ids = df_trips.loc[f, "person_id"].unique()
    print(f"Deleting {len(problematic_ids)} persons with arrival_time > next departure_time")
    # Delete from df_trips and df_persons
    df_trips = df_trips[~df_trips["person_id"].isin(problematic_ids)].copy()
    df_persons = df_persons[~df_persons["person_id"].isin(problematic_ids)].copy()

    # Filter out persons for which we do not have sufficient information
    unknown_ids = set(df_trips[
        (df_trips["mode"] == "unknown") | (df_trips["preceding_purpose"] == "unknown")
        | (df_trips["following_purpose"] == "unknown")
    ]["person_id"])

    print("  Removed %d persons with trips with unknown mode or unknown purpose" % len(unknown_ids))
    df_trips = df_trips[~df_trips["person_id"].isin(unknown_ids)]
    df_persons = df_persons[~df_persons["person_id"].isin(unknown_ids)].copy()

    print(len(set(df_persons["person_id"].values) - set(df_trips["person_id"].values)), "raw number of persons without trips")

    return df_households, df_persons, df_trips, df_household_members




# =====================================================
# Aggregation logic, same as in the excel but adjusted:
#   if (all == 1) => 1                                 
#   else if (any == 2) => 2                            
#   else if (any == 3) => 3                            
#   else if (any == 4) => 4                            
#   else if (any == 5) => 5                            
#   else => 6                                          
#                                                      
#   1 = walk                                           
#   2 = public transport                               
#   3 = bike                                           
#   4 = car                                            
#   5 = car_passenger                                  
#   6 = other                                          
# =====================================================

def aggregate_transport_mode(df_trips):
    MODE_COLUMNS = ['mode_part1', 'mode_part2', 'mode_part3', 'mode_part4']

    # Map to basic modes
    for mode_column in MODE_COLUMNS:
        df_trips[mode_column] = df_trips[mode_column].map(MODES_BASIC_MAP)


    # Aggregate mode
    def mode_aggregator(row):
        if(row['mode_part1'] + row['mode_part2'] + row['mode_part3'] + row['mode_part4'] == 1):
            return 1 # walk
        for mode_column in MODE_COLUMNS:
            if row[mode_column] == 2:
                return 2 # pt
        for mode_column in MODE_COLUMNS:
            if row[mode_column] == 3:
                return 3 # bike
        for mode_column in MODE_COLUMNS:
            if row[mode_column] == 4:
                return 4 # car
        for mode_column in MODE_COLUMNS:
            if row[mode_column] == 5:
                return 5 # car_passenger
        return 6 # other

    df_trips['mode'] = df_trips.apply(mode_aggregator, axis=1)

    # Map final mode name
    df_trips["mode"] = df_trips["mode"].map(MODES_MAP)
    df_trips["mode"] = df_trips["mode"].astype("category")
    return df_trips


MAP_ZONES_COLUMNS = {
    "ZONA17": "zone_code", # code representation of the zone
    "BARRIO": "zone", #  name of the zone
}

MAP_MUNICIPALITY_COLUMNS = {
    "Cod": "municipality_code", # code representation of the municipality
    "MUNICIPIO": "municipality" # name of the municipality
}

def calculate_trip_distance(context, df_trips):
    EXCEL_PATH = f"{context.config('data_path')}/{context.config('seville.hts')}"
    # List of all zones with name-code mapping
    df_zones = pd.read_excel(
        EXCEL_PATH,
        dtype = {},
        sheet_name="Zon_",
        usecols="A:B",
        nrows=137
    )
    df_zones = df_zones.rename(MAP_ZONES_COLUMNS, axis=1)

    # List of all municipalities with name-code mapping
    df_municipalities = pd.read_excel(
        EXCEL_PATH,
        dtype = {},
        sheet_name="Zon_",
        usecols="A:B",
        skiprows=137
    )
    df_municipalities = df_municipalities.rename(MAP_MUNICIPALITY_COLUMNS, axis=1)
    # Seville is not represented, therefore appears only as "-" or "" value
    seville_row = pd.DataFrame({"municipality_code": ["-", ""], "municipality": ["Sevilla", "Sevilla"]})
    df_municipalities = pd.concat([df_municipalities, seville_row], ignore_index=True)

    column_names = ["street", "zone_code", "municipality_code"]

    # Get all origins and destinations of all trips, origin/destination has street, zone and municipality
    df_ori = df_trips[["street_ori", "zone_code_ori", "municipality_code_ori"]]    
    df_ori.columns = column_names

    df_des = df_trips[["street_des", "zone_code_des", "municipality_code_des"]]
    df_des.columns = column_names

    # assign to origin/destination (df_streets) a coordinates
    # we only know coordinates based on the names of the locations
    # so we first assign zone and municipality names
    def assign_coords(df_streets, df_coords):
        df = df_streets.copy()
        # save indices to align with df_trips
        df["__idx__"] = df.index
        
        # add zone and municipality name based on the codes
        df = df.merge(df_zones, on='zone_code', how='left', validate='m:1')
        df = df.merge(df_municipalities, on='municipality_code', how='left', validate='m:1')
        # add coordinates based on the street, municipality and zone names
        df = df.merge(df_coords, on=['municipality','zone','street'], how='left', validate='m:1')
        # reset indices to align with df_trips
        df.index = df["__idx__"]
        return df

    df_coords :pd.DataFrame = context.stage("seville.data.hts.entd.streets.verify").copy()
    print(f"calculate_distance: Dropping {df_coords[['municipality','zone','street']].duplicated().sum()} duplicated coordinates")
    df_coords.drop_duplicates(inplace=True, subset=['municipality','zone','street'])

    df_ori = assign_coords(df_ori, df_coords)
    df_des = assign_coords(df_des, df_coords)
    assert len(df_des) == len(df_ori)


    assert len(df_ori) == len(df_des) == len(df_trips), f"df_ori:{len(df_ori)} == df_des:{len(df_des)} == df_trips:{len(df_trips)}"
    known_geometry = df_ori['geometry'].notna() & df_des['geometry'].notna()
    #delete_condition = delete_condition.reindex(df_trips.index, fill_value=False)

    print("Calculating euclidean distance:")
    print(f"[INFO] Cannot calculate distance for {(~known_geometry).sum()} trips.")

    df_trips["ori"] = np.nan
    df_trips["des"] = np.nan
    df_trips.loc[known_geometry, "ori"] = df_ori.loc[known_geometry, 'geometry']
    df_trips.loc[known_geometry, "des"] = df_des.loc[known_geometry, 'geometry']


    df_trips['euclidean_distance'] = np.nan
    df_trips.loc[known_geometry, 'euclidean_distance'] = df_trips[known_geometry].apply(
        lambda x: geodesic((x.ori.y, x.ori.x), (x.des.y, x.des.x)).meters, 
        axis=1
    ) # result in meters
    
    df_trips['origin_location'] = df_trips["ori"]
    df_trips['destination_location'] = df_trips["des"]

    return df_trips


def calculate_income_class(df):
    assert "household_income" in df
    assert "consumption_units" in df

    return np.digitize(df["household_income"] / df["consumption_units"], INCOME_CLASS_BOUNDS_SEVILLE, right = True)

    
