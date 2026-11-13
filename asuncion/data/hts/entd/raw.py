from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import geopandas as gpd
import geopy

"""
This stage loads the raw data of the Household Travel Survey 2017 from Seville municipality 
and converts it to the format used by the HTS.
"""


MAP_HOUSEHOLDS_COLUMNS = {
    'Id': 'household_id', 
    'Fex': 'household_weight',
    'Personas': 'household_size',
    #'number_of_vehicles is made of following columns
    'AutooEquivalente': 'number_of_cars',
    'Moto': 'number_of_motorbikes',

    'NBicicletas': 'number_of_bikes', 
    "IdDepartamento": 'departement_id',
#    '': 'consumption_units', # calculated in cleaned.py
#    '': 'urban_type' # unknown - default set to urban_center
    'IngresoTotal': 'household_income', # income class is then calculated from household_income
    'IdDepartamentoDistrito': 'hts_household_district_id'
}

MAP_PERSONS_COLUMNS = {
    'Id': 'person_id',
    "IdViviendaHogar": 'household_id',
    'Fex': 'person_weight',
    'Edad': 'age',
    'sexo': 'sex',
    'ActLaboralReciente' :'employed', # can be deduced from a) worked last week b) if the person has income from primary work
    "IngresoTrabPrincipal": 'primary_job_income', 
    "IngresoTrabSecundarios": 'secondary_job_income',
    'AsistenciaActualEducacion': 'studies',
    'LicenciaConducir': 'has_license',
    # '': 'has_pt_subscription', # no data - we idicate as NaN
    "Viajes": 'number_of_trips',
    "IdDepartamento": 'departement_id',
    #'': 'trip_weight', - same as person_weight
    # '': 'is_passenger', # we calculate from trips
    # '': 'socioprofessional_class' # we use nan
}

MAP_TRIPS_COLUMNS = {
    'Id': 'trip_id', 
    "IdPersona": 'person_id',
    'Fex': 'trip_weight',
    'HoraMinutoInicio': 'departure_time',
    'HoraMinutoFin': 'arrival_time',
    'TiempoMinuto': 'trip_duration',
    # '': 'activity_duration', # calculated using hts.compute_activity_duration
    "IdCategoriaActividad": 'following_purpose',
    "IdCategoriaActividadOrigen": 'preceding_purpose',
    # '': 'is_last_trip', # calculated using hts.compute_first_last
    # '': 'is_first_trip', # calculated using hts.compute_first_last

    'CategoriaModoPrincipal': 'mode',

    'IdDistritoOrigen':'origin_district_id', # origin departement id is extracted from this
    'IdDistrito':'destination_district_id', # destination departement id is extracted from this
}

MAP_LEGS_COLUMNS = {
    'IdViaje':'trip_id',
    'distancialineal':'euclidean_distance',
    'distancia':'routed_distance',
    'Modo':'mode'
}

# --------------------------------------------------------------------------

def configure(context):
    context.config("data_path")
    context.config("asuncion.hts_households", "hts/ViviendaHogar.csv")
    context.config("asuncion.hts_persons", "hts/Persona.csv")
    context.config("asuncion.hts_trips", "hts/Viaje.csv")
    context.config("asuncion.hts_legs", "hts/Etapa.csv")


def execute(context):

    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("asuncion.hts_households"))
    print(f"Loading hts data from {FILE_PATH}")
    df_households = pd.read_csv(FILE_PATH, sep=";", dtype=str, encoding = "latin1")
    df_households = df_households[MAP_HOUSEHOLDS_COLUMNS.keys()]
    df_households = df_households.rename(columns=MAP_HOUSEHOLDS_COLUMNS)

    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("asuncion.hts_persons"))
    print(f"Loading hts data from {FILE_PATH}")
    df_persons = pd.read_csv(FILE_PATH, sep=";", dtype=str, encoding = "latin1")
    df_persons = df_persons[MAP_PERSONS_COLUMNS.keys()]
    df_persons = df_persons.rename(columns=MAP_PERSONS_COLUMNS)

    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("asuncion.hts_trips"))
    print(f"Loading hts data from {FILE_PATH}")
    df_trips = pd.read_csv(FILE_PATH, sep=";", dtype=str, encoding = "latin1")
    df_trips = df_trips[MAP_TRIPS_COLUMNS.keys()]
    df_trips = df_trips.rename(columns=MAP_TRIPS_COLUMNS)

    FILE_PATH = "{}/{}".format(context.config("data_path"), context.config("asuncion.hts_legs"))
    print(f"Loading hts data from {FILE_PATH}")
    df_legs = pd.read_csv(FILE_PATH, sep=";", dtype=str, encoding = "latin1")
    df_legs = df_legs[MAP_LEGS_COLUMNS.keys()]
    df_legs = df_legs.rename(columns=MAP_LEGS_COLUMNS)



# --------------------------------------------------------------------------
    # Set missing columns
    df_persons['trip_weight'] = df_persons['person_weight']

    # Social progessional class
    #   - we have no data
    df_persons["socioprofessional_class"] = np.nan
    # Public transport subscription 
    #   - we have no data
    df_persons["has_pt_subscription"] = np.nan 

    # -------------------------------------------------------------------------------------

    df_trips['is_valid'] = True


    # Check for whitespace-only or truly empty strings
    empty_trip_duration = df_trips["trip_duration"].isna()
    empty_departure_time = df_trips["departure_time"].isna() | (df_trips["departure_time"].str.strip() == "")
    # Delete those rows which have empty start or end time
    df_trips['is_valid'] &= ~(empty_departure_time | empty_trip_duration)

    return df_persons, df_households, df_trips, df_legs

def validate(context):
    filenames = [
        "asuncion.hts_households",
        "asuncion.hts_persons",
        "asuncion.hts_trips"
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"HTS data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
