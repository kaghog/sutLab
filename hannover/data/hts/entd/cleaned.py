from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the national HTS.
"""

def configure(context):
    context.stage("data.hts.entd.raw")

INCOME_CLASS_BOUNDS = [400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 4000, 6000, 10000, 1e6]

PURPOSE_MAP = [
    ("0", "home"),
    ("1", "work"),
    ("2", "work"),
    ("3", "education"),
    ("4", "shop"),
    ("5", "other"),
    ("6", "leisure"),
    ("7", "leisure"),
    ("8", "home"),
    ("9", "other"),
    ("10", "other"),
    ("99", "other"),
]

MODES_MAP = [
    ("1", "walk"),
    ("2", "bike"), #
    ("3", "car_passenger"),
    ("4", "car"), # taxi
    ("5", "pt"),
    ("9", "pt"), # no information assume pt
    ("703", "pt"), # Other assume pt
]

def convert_time(x):
    return np.dot(np.array(x.split(":"), dtype = float), [3600.0, 60.0, 1.0])

def execute(context):
    df_individu, df_tcm_individu, df_menage, df_tcm_menage, df_deploc = context.stage("data.hts.entd.raw")

    # Make copies
    df_persons = pd.DataFrame(df_tcm_individu, copy = True)
    df_households = pd.DataFrame(df_tcm_menage, copy = True)
    df_trips = pd.DataFrame(df_deploc, copy = True)

    # Get weights for persons that actually have trips
    df_persons = pd.merge(df_persons, df_trips[["IDENT_IND", "PONDKI"]].drop_duplicates("IDENT_IND"), on = "IDENT_IND", how = "left")
    df_persons["is_kish"] = ~df_persons["PONDKI"].isna()
    df_persons["trip_weight"] = df_persons["PONDKI"].fillna(0.0)

    # # Important: If someone did not have any trips on the reference day, ENTD asked
    # # for another day. With this flag we make sure that we only cover "reference days".
    # f = df_trips["V2_MOBILREF"] == 1
    # df_trips = df_trips[f]
    # print("Filtering out %d non-reference day trips" % np.count_nonzero(~f))

    # Merge in additional information from ENTD
    df_households = pd.merge(df_households, df_menage[[
        "idENT_MEN", "V1_JNBVEH", "V1_JNBMOTO", "V1_JNBCYCLO", "V1_JNBVELOADT"
    ]], on = "idENT_MEN", how = "left")

    df_persons = pd.merge(df_persons, df_individu[[
        "IDENT_IND", "V1_GPERMIS", "V1_GPERMIS2R", "V1_ICARTABON"
    ]], on = "IDENT_IND", how = "left")

    # Transform original IDs to integer (they are hierarchichal)
    df_persons["entd_person_id"] = df_persons["IDENT_IND"].astype(int)
    df_persons["entd_household_id"] = df_persons["IDENT_MEN"].astype(int)
    df_households["entd_household_id"] = df_households["idENT_MEN"].astype(int)
    df_trips["entd_person_id"] = df_trips["IDENT_IND"].astype(int)

    # Construct new IDs for households, persons and trips (which are unique globally)
    df_households["household_id"] = np.arange(len(df_households))

    df_persons = pd.merge(
        df_persons, df_households[["entd_household_id", "household_id"]],
        on = "entd_household_id"
    )
    df_persons["person_id"] = np.arange(len(df_persons))

    df_trips = pd.merge(
        df_trips, df_persons[["entd_person_id", "person_id", "household_id"]],
        on = ["entd_person_id"]
    )
    df_trips["trip_id"] = np.arange(len(df_trips))

    # Weight
    df_persons["person_weight"] = df_persons["PONDV1"].astype(float)
    df_households["household_weight"] = df_households["PONDV1"].astype(float)

    # Clean age
    df_persons.loc[:, "age"] = df_persons["AGE"]

    # Clean sex
    df_persons.loc[df_persons["SEXE"] == 1, "sex"] = "male"
    df_persons.loc[df_persons["SEXE"] == 2, "sex"] = "female"
    df_persons["sex"] = df_persons["sex"].astype("category")

    # Household size
    df_households["household_size"] = df_households["NPERS"]

    # Clean departement
    df_households["departement_id"] = df_households["DEP"].fillna("undefined").astype("category")
    df_persons["departement_id"] = df_persons["DEP"].fillna("undefined").astype("category")

    df_trips["origin_departement_id"] = df_trips["V2_MORIDEP"].fillna("undefined").astype("category")
    df_trips["destination_departement_id"] = df_trips["V2_MDESDEP"].fillna("undefined").astype("category")

    # Clean urban type
    df_households["urban_type"] = df_households["numcom_UU2010"].replace({
        11: "suburb",
        12: "central_city",
        21: "isolated_city",
        22: "none"
    })

    assert np.all(~df_households["urban_type"].isna())
    df_households["urban_type"] = df_households["urban_type"].astype("category")

    # Clean employment
    df_persons["employed"] = df_persons["SITUA"].isin([1, 2])

    # Studies
    # Many < 14 year old have NaN
    df_persons["studies"] = df_persons["ETUDES"].fillna(1) == 1
    df_persons.loc[df_persons["age"] < 5, "studies"] = False

    # Number of vehicles
    df_households["number_of_vehicles"] = 0
    df_households["number_of_vehicles"] += df_households["V1_JNBVEH"].fillna(0)
    df_households["number_of_vehicles"] += df_households["V1_JNBMOTO"].fillna(0)
    df_households["number_of_vehicles"] += df_households["V1_JNBCYCLO"].fillna(0)
    df_households["number_of_vehicles"] = df_households["number_of_vehicles"].astype(int)

    df_households["number_of_bikes"] = df_households["V1_JNBVELOADT"].fillna(0).astype(int)

    # License
    df_persons["has_license"] = (df_persons["V1_GPERMIS"] == 1) | (df_persons["V1_GPERMIS2R"] == 1)

    # Has subscription
    df_persons["has_pt_subscription"] = df_persons["V1_ICARTABON"] == 1

    # Household income
    df_households["income_class"] = df_households["TrancheRevenuMensuel"]
    df_households.loc[df_households["income_class"] == 95, "income_class"] = -1
    df_households["income_class"] = df_households["income_class"].astype(int)

    # Trip purpose
    df_trips["following_purpose"] = "other"
    df_trips["preceding_purpose"] = "other"

    for prefix, activity_type in PURPOSE_MAP:
        df_trips.loc[
            df_trips["V2_MMOTIFDES"].astype(str).str.startswith(prefix), "following_purpose"
        ] = activity_type

        df_trips.loc[
            df_trips["V2_MMOTIFORI"].astype(str).str.startswith(prefix), "preceding_purpose"
        ] = activity_type

    df_trips["following_purpose"] = df_trips["following_purpose"].astype("category")
    df_trips["preceding_purpose"] = df_trips["preceding_purpose"].astype("category")

    # Trip mode
    df_trips["mode"] = "pt"

    for prefix, mode in MODES_MAP:
        df_trips.loc[
            df_trips["V2_MTP"].astype(str).str.startswith(prefix), "mode"
        ] = mode

    df_trips["mode"] = df_trips["mode"].astype("category")

    # Further trip attributes
    df_trips["routed_distance"] = df_trips["V2_MDISTTOT"] * 1000.0
    df_trips["routed_distance"] = df_trips["routed_distance"].fillna(0.0) # This should be just one within Île-de-France

    # Only leave trips on days with V2_TYPJOUR values in [1,2,3,4,5]
    f = df_trips["V2_TYPJOUR"].isin([1, 2, 3, 4, 5])
    print("Removing %d trips on weekends" % np.count_nonzero(~f))
    df_trips = df_trips[f]

    # Only leave one day per person
    initial_count = len(df_trips)

    df_first_day = df_trips[["person_id", "IDENT_JOUR"]].sort_values(
        by = ["person_id", "IDENT_JOUR"]
    ).drop_duplicates("person_id")
    df_trips = pd.merge(df_trips, df_first_day, how = "inner", on = ["person_id", "IDENT_JOUR"])

    final_count = len(df_trips)
    print("Removed %d trips for non-primary days" % (initial_count - final_count))

    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Trip times
    df_trips["departure_time"] = df_trips["V2_MORIHDEP"].apply(convert_time).astype(float)
    df_trips["arrival_time"] = df_trips["V2_MDESHARR"].apply(convert_time).astype(float)
    df_trips = hts.fix_trip_times(df_trips)

    # Durations
    df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]
    hts.compute_activity_duration(df_trips)

    # Add weight to trips
    df_trips["trip_weight"] = df_trips["PONDKI"]

    # Chain length
    df_persons = pd.merge(
        df_persons, df_trips[["person_id", "NDEP"]].drop_duplicates("person_id").rename(columns = { "NDEP": "number_of_trips" }),
        on = "person_id", how = "left"
    )
    df_persons["number_of_trips"] = df_persons["number_of_trips"].fillna(-1).astype(int)
    df_persons.loc[(df_persons["number_of_trips"] == -1) & df_persons["is_kish"], "number_of_trips"] = 0
      
    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(
        df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
    )
    
    
    # Get set of household_ids in each dataframe and find missing ones
    df_size = df_persons.groupby("household_id").size().reset_index(name = "count")
    ids_households = set(df_households["household_id"])
    ids_size = set(df_size["household_id"])
    missing_ids = ids_households - ids_size
    # Remove rows with missing household_id in df_size from df_households
    df_households = df_households[~df_households["household_id"].isin(missing_ids)]
    print("Removed %d households with no persons" % len(missing_ids))
    

    df_size = pd.merge(df_households[["household_id", "household_size"]], df_size, on = "household_id")
    # Second check: find mismatched household sizes
    mismatched = df_size[df_size["household_size"] != df_size["count"]]

    if not mismatched.empty:
        print("Households with mismatched sizes:", len(mismatched))
        # Update household_size in df_households with correct count
        df_households = pd.merge(df_households, df_size[["household_id", "count"]], on="household_id", how="left")
        df_households["household_size"] = df_households["count"]
        df_households.drop(columns=["count"], inplace=True)
        print("Change household sizes in df_households to match counts in df_size.")
    else:
        print("All household sizes match counts")

    # Calculate consumption units
    hts.check_household_size(df_households, df_persons)
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on = "household_id")

    # Socioprofessional class
    df_persons["socioprofessional_class"] = df_persons["CS24"].fillna(80).astype(int) // 10

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

    return df_households, df_persons, df_trips

def calculate_income_class(df):
    assert "household_income" in df
    assert "consumption_units" in df

    return np.digitize(df["household_income"], INCOME_CLASS_BOUNDS, right = True)
