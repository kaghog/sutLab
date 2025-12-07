from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import geopandas as gpd
import geopy

"""
This stage loads the raw data of the MiD german ENTD (Mobilität in Deutschland) 2017 survey and converts it to the format used by the HTS.
"""


MAP_HOUSEHOLDS_COLUMNS = {
    'ID': 'household_id', 
    'Coef': 'household_weight', 
    'Tamaño_hogar': 'household_size',
    'Total_vehiculos': 'number_of_vehicles', 
    'N_VEH_4': 'number_of_bikes', 
#    '': 'departement_id', # Seville - 41
#    '': 'consumption_units', # calculated in cleaned.py using df_households_members

    'NOCAPITAL': 'urban_type', # 

    'INGRESOS': 'income_class' # income class
}

MAP_PERSONS_COLUMNS = {
    'ID': 'person_id',
    # 'ID': 'household_id', - is set later on, same as person_id
    'Coef': 'person_weight',
    'EDAD': 'age',
    'SEXO': 'sex',
    'SITLAB': 'employed', # this is main activity, includes work and study
    # '': 'studies',  # is mapped from 'employed'
    'CARNET': 'has_license',
    # '': 'has_pt_subscription', # no data - we idicate as NaN
    # '': 'number_of_trips', # calculated from df_trips
    # '': 'departement_id', # Seville - 41
    # 'Coef': 'trip_weight',
    # '': 'is_passenger', # we use 'mode' to determine if passenger
    # '': 'socioprofessional_class' # we use nan
}

MAP_TRIPS_COLUMNS = {
    'ID': 'person_id', 
    '#ID-VIAJE': 'trip_id',
    'COEF': 'trip_weight',
    'HINI_D1': 'departure_time',
    # '': 'arrival_time', # calculated late using 'duration + departure time'
    'DURAC_D1': 'trip_duration',
    # '': 'activity_duration', # calculated using hts.compute_activity_duration
    'MOTDES_D1': 'following_purpose',
    'MOTORI_D1': 'preceding_purpose',
    # '': 'is_last_trip', # calculated using hts.compute_first_last
    # '': 'is_first_trip', # calculated using hts.compute_first_last
    # '': 'mode', # is inffered from MODO_D1_EX values


    # '': 'origin_departement_id', # Seville - 41
    # '': 'destination_departement_id', # Seville - 41

    # we could probably calculate this by using average speed of the type of transport or something
    # '': 'routed_distance' # TODO: NOT FOUND
    '#Den_CALLESEV_ORI': 'source_address',
    '#Den_CALLESEVDES': 'destination_address',


    'MODO_D1_E1': 'mode_part1',
    'MODO_D1_E2': 'mode_part2',
    'MODO_D1_E3': 'mode_part3',
    'MODO_D1_E4': 'mode_part4',

    
    'Numero de Viajes': 'number_of_trips', # to get number of trips in df_persons

    'PROVI_ORI_D1': 'origin_departement_id',
    'PROVI_DES_D1': 'destination_departement_id',

    "#Den_CALLESEV_ORI": "street_ori",
    "#Zona_CALLESEV_ORI": "zone_code_ori",
    "MUNI_ORI_SEV_D1": "municipality_code_ori",
    "#Den_CALLESEVDES": "street_des",
    "#Zona_CALLESEV_DES": "zone_code_des",
    "MUNI_DES_SEV_D1": "municipality_code_des",
}

MAP_HOUSEHOLD_MEMBERS_COLUMNS = {
    'ID': 'household_id',
    'SIND_ERG_1': 'age1',
    'SIND_ERG_2': 'age2',
    'SIND_ERG_3': 'age3',
    'SIND_ERG_4': 'age4',
    'SIND_ERG_5': 'age5',
    'SIND_ERG_6': 'age6',
    'SIND_ERG_7': 'age7',
    'SIND_ERG_8': 'age8',
    'SIND_ERG_9': 'age9',
    'SIND_ERG_10': 'age10',
}


# --------------------------------------------------------------------------

def configure(context):
    context.config("data_path")
    context.config("seville.hts", "Household Travel Survey 2017/BD entrevistas telefonicas Completa_Final_v2.xlsb.xlsx")

def execute(context):

    EXCEL_PATH = f"{context.config('data_path')}/{context.config('seville.hts')}"
    households_sheet = pd.read_excel(
        EXCEL_PATH,
        dtype = {},
        sheet_name="Hogares"
    )
    trips_sheet = pd.read_excel(
        EXCEL_PATH,
        dtype = {},
        sheet_name="Viajes_Dep"
    )

    # Filter columns
    df_persons = households_sheet[MAP_PERSONS_COLUMNS.keys()]

    df_households = households_sheet[MAP_HOUSEHOLDS_COLUMNS.keys()]
    df_trips = trips_sheet[MAP_TRIPS_COLUMNS.keys()]

    # We usually have travel details only about single member of the household.
    # For others we have only sex and age. These data is used only for the
    # for the calculation of the consumption units in 'cleaned.py', For the clarity
    # of the code, it is separete from the df_households.
    df_household_members = households_sheet[MAP_HOUSEHOLD_MEMBERS_COLUMNS.keys()]

    # Map columns names
    df_persons = df_persons.rename(MAP_PERSONS_COLUMNS, axis=1)
    df_households = df_households.rename(MAP_HOUSEHOLDS_COLUMNS, axis=1)
    df_trips = df_trips.rename(MAP_TRIPS_COLUMNS, axis=1)
    df_household_members = df_household_members.rename(MAP_HOUSEHOLD_MEMBERS_COLUMNS, axis=1)

# --------------------------------------------------------------------------

    # Set duplicated columns
    df_persons['household_id'] = df_persons['person_id']
    df_persons['trip_weight'] = df_persons['person_weight']

    # Filter out trips outside province of Seville
    df_trips = df_trips[df_trips["origin_departement_id"] == "-"] # "-" indicates province of Seville
    df_trips = df_trips[df_trips["destination_departement_id"] == "-"] # "-" indicates province of Seville

    # Mock departnemt-related columns
    DEPARTMENT_VALUE = "41"
    df_households["departement_id"] = DEPARTMENT_VALUE
    df_persons["departement_id"] = DEPARTMENT_VALUE 
    df_trips["origin_departement_id"] = DEPARTMENT_VALUE
    df_trips["destination_departement_id"] = DEPARTMENT_VALUE
                                 
    
    # Social progessional class
    #   - we have no data
    df_persons["socioprofessional_class"] = np.nan
    # Public transport subscription 
    #   - we have no data
    df_persons["has_pt_subscription"] = np.nan 

    # -------------------------------------------------------------------------------------

    # Check for whitespace-only or truly empty strings
    empty_trip_duration = df_trips["trip_duration"].isna()
    empty_departure_time = df_trips["departure_time"].isna() | (df_trips["departure_time"].str.strip() == "")
    # Delete those rows which have empty start or end time
    df_trips = df_trips[~(empty_departure_time | empty_trip_duration)].copy()
  
    return df_persons, df_households, df_trips, df_household_members

def validate(context):
    EXCEL_PATH = f"{context.config('data_path')}/{context.config('seville.hts')}"
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File missing from ENTD: {EXCEL_PATH}")

    return [
        os.path.getsize(EXCEL_PATH),
    ]
