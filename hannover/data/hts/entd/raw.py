from tqdm import tqdm
import pandas as pd
import os
import numpy as np

"""
This stage loads the raw data of the MiD german ENTD (Mobilität in Deutschland) 2017 survey and converts it to the format used by the HTS.
"""

Q_MENAGE_COLUMNS = [
    "DEP", "idENT_MEN", "PONDV1", "RG",
    "V1_JNBVELOADT",
    "V1_JNBVEH", "V1_JNBMOTO", "V1_JNBCYCLO"
]

Q_TCM_MENAGE_COLUMNS = [
    "NPERS", "PONDV1", "TrancheRevenuMensuel",
    "DEP", "idENT_MEN", "RG", "numcom_UU2010"
]

Q_INDIVIDU_COLUMNS = [
    "IDENT_IND", "idENT_MEN",
    "RG", "V1_GPERMIS", "V1_ICARTABON",
    "V1_GPERMIS2R"
]

Q_TCM_INDIVIDU_COLUMNS = [
    "AGE", "ETUDES", "IDENT_IND", "IDENT_MEN",
    "PONDV1", "CS24", "SEXE", "DEP", "SITUA",
]

K_DEPLOC_COLUMNS = [
    "IDENT_IND", "V2_MMOTIFDES", "V2_MMOTIFORI",
    "V2_TYPJOUR", "V2_MORIHDEP", "V2_MDESHARR", "V2_MDISTTOT",
    "IDENT_JOUR", "V2_MTP",
    "V2_MDESDEP", "V2_MORIDEP", "NDEP", "V2_MOBILREF",
    "PONDKI"
]

# --------------------------------------------------------------------------

# Matching columns with HTS columns of German ENTD
HAUSEHOLD_MENAGE_COLUMNS = [
    "BLAND", # Region code (Bundesland)
    "H_ID",      # Household ID
    "H_GEW",  # Household weight
    "H_ANZRAD",  # Number of bicycles
    "H_ANZAUTO", # Number of cars
    "H_ANZMOT", # Number of motorcycles
    "H_ANZMOP", # Number of mopeds/scooters
]

HAUSEHOLD_TCM_MENAGE_COLUMNS = [
    "H_GR",    # Household size
    "H_GEW",  # Household weight
    "hheink_gr1",  # Monthly household income group
    "BLAND",     # Region code (Bundesland)
    "H_ID",      # Household ID
    "RegioStaR4"  # Urban unit code (urban/rural classification)
]

PERSONEN_INDIVIDU_COLUMN = [
    "HP_ID",     # Individual ID
    "H_ID",      # Household ID
    "BLAND",     # Region code (Bundesland)
    "P_FS_PKW",  # Driving license for car
    "P_FKARTE",  # Public transport subscription
    "P_FS_MOT"   # Motorcycle license
]

PERSONEN_TCM_INDIVIDU_COLUMN = [
    "HP_ALTER", # Age
    "HP_ID", # Individual ID
    "H_ID",    # Household ID
    "P_GEW", # Weight
    "HP_SEX", # Sex
    "BLAND", # Region code (Bundesland)
    "HP_TAET" # Activity status include employment and student status)
]
    
TRAVEL_DEPLOC_COLUMNS = [
    "HP_ID", # Individual ID
    "zweck", # Purpose of travel
    "ST_WOTAG", # Day of the week
    "W_SZ" , # Start time of travel 00:00 - 23:59
    "W_AZ", # End time of travel 00:00 - 23:59
    "wegkm", # Distance of travel in km
    "W_ID", # Travel ID
    "hvm", # Mode of transport
    "W_GEW", # Weight of the travel record
    "BLAND" # for Destination and Origin departement code, assume same
]
    
    

def configure(context):
    context.config("data_path")

def execute(context):
    personen_individu_all = pd.read_csv(
        "%s/mid2017/MiD2017_Personen.csv" % context.config("data_path"), 
        encoding = "latin1", dtype = { "DEP": str }
    )
    
    hausehold_menage_all = pd.read_csv(
        "%s/mid2017/MiD2017_Haushalte.csv" % context.config("data_path"), 
        encoding = "latin1", dtype = { "DEP": str }
    )
    
    travel_deploc_all = pd.read_csv(
        "%s/mid2017/MiD2017_Wege.csv" % context.config("data_path"), 
        encoding = "latin1", dtype = { "DEP": str }
    )
    
    # Filter where BLAND == 3
    personen_filtered = personen_individu_all[personen_individu_all["BLAND"] == 3]
    hausehold_filtered = hausehold_menage_all[hausehold_menage_all["BLAND"] == 3]
    travel_filtered = travel_deploc_all[travel_deploc_all["BLAND"] == 3]


    # If needed, reset index
    personen_individu_all = personen_filtered.reset_index(drop=True)
    hausehold_menage_all = hausehold_filtered.reset_index(drop=True)
    travel_deploc_all = travel_filtered.reset_index(drop=True)
    
    
    # -------------------------------------------------------------------------------------
    
    hausehold_menage = hausehold_menage_all[HAUSEHOLD_MENAGE_COLUMNS].copy()
    df_menage = hausehold_menage.rename(columns={
        "H_ID": "idENT_MEN",
        "H_GEW": "PONDV1",
        "BLAND": "DEP",  # Region code (Bundesland)
        "H_ANZRAD": "V1_JNBCYCLO",
        "H_ANZAUTO": "V1_JNBVEH",
        "H_ANZMOT": "V1_JNBMOTO",
        "H_ANZMOP": "V1_JNBVELOADT"
    })
    # We set RG and DEP are the same in German ENTD which is  BLAND (Bundesland)
    df_menage["RG"] = df_menage["DEP"]
    
    # -------------------------------------------------------------------------------------
    
    hausehold_tcm_menage = hausehold_menage_all[HAUSEHOLD_TCM_MENAGE_COLUMNS].copy()
    df_tcm_menage = hausehold_tcm_menage.rename(columns={
        "H_GR": "NPERS",
        "H_GEW": "PONDV1",
        "hheink_gr1": "TrancheRevenuMensuel",
        "BLAND": "DEP",  # Region code (Bundesland)
        "H_ID": "idENT_MEN",
        "RegioStaR4": "numcom_UU2010"  # Urban unit code (1: urban, 2: rural)
    })
    df_tcm_menage["RG"] = df_tcm_menage["DEP"]
    
    # -------------------------------------------------------------------------------------
    
    personen_individu = personen_individu_all[PERSONEN_INDIVIDU_COLUMN].copy()
    personen_individu["P_FS_PKW"] = np.where(
        personen_individu["P_FS_PKW"].isin([1]),
        1, 2)
    
    personen_individu["P_FKARTE"] = np.where(
        personen_individu["P_FKARTE"].isin([1,2,3,4,5,6]),
        1, 2)
    
    personen_individu["P_FS_MOT"] = np.where(
        personen_individu["P_FS_MOT"].isin([1]),
        1, 2)
    
    df_individu =  personen_individu.rename(columns={
    "HP_ID": "IDENT_IND",
    "H_ID": "idENT_MEN",
    "BLAND": "RG",  # Region code (Bundesland)
    "P_FS_PKW": "V1_GPERMIS",
    "P_FKARTE": "V1_ICARTABON",
    "P_FS_MOT": "V1_GPERMIS2R"
    })
    
    # -------------------------------------------------------------------------------------
    
    personen_tcm_individu = personen_individu_all[PERSONEN_TCM_INDIVIDU_COLUMN].copy()
    personen_tcm_individu["ETUDES"] = np.where(
        personen_tcm_individu["HP_TAET"].isin([8,9,10]),
        1, 2)
    personen_tcm_individu["CS24"] =  np.nan
    
    # French SITUA values:
        # 1 Employed
        # 2 Apprentice under contract or in paid internship
        # 3 Student, pupil, in training or unpaid internship
        # 4 Unemployed (registered or not with the ANPE)
        # 5 Retired or withdrawn from business or in early retirement
        # 6 Housewife or househusband
        # 7 Other situation (disabled person, etc.)
    personen_tcm_individu["SITUA" ] = personen_tcm_individu["HP_TAET"].replace({
        1: 1,  # Full-time employment Nominal
        2: 1,  # Part-time employment, i.e. 18 to less than 35 hours per week
        3: 1,  # Marginal employment, i.e. 11 to less than 18 hours per week
        4: 1,  # Employment as a secondary activity or internship
        5: 1,  # Employment without specification of scope
        6: 7,  # Child is cared for at home
        7: 7,  # Child is cared for in kindergarten, nursery, by a childminder, etc.
        8: 3,  # School pupil, including preschool
        9: 2,  # Trainee
        10: 3, # Student
        11: 6, # Housewife/househusband
        12: 5, # Retired/pensioner
        13: 4, # Currently unemployed
        14: 7, # Other activity
        15: 7, # Not employed, no details provided
        99: 7, # No information provided
        })   
    
    personen_tcm_individu["HP_SEX"] = personen_tcm_individu["HP_SEX"].replace({
        1:1,
        2:2,
        9:1 # Set unknow as male
    }) 
    
    personen_tcm_individu_draft = personen_tcm_individu.rename(columns={
    "HP_ALTER": "AGE",
    "HP_ID": "IDENT_IND",
    "H_ID": "IDENT_MEN",
    "P_GEW": "PONDV1",
    "HP_SEX": "SEXE",
    "BLAND": "DEP",  # Region code (Bundesland)
    })
    df_tcm_individu = personen_tcm_individu_draft[Q_TCM_INDIVIDU_COLUMNS].copy()
      
    # -------------------------------------------------------------------------------------
      
    travel_deploc = travel_deploc_all[TRAVEL_DEPLOC_COLUMNS].copy()
    
    # Check for whitespace-only or truly empty strings
    empty_sz = travel_deploc["W_SZ"].isna() | (travel_deploc["W_SZ"].str.strip() == "")
    empty_az = travel_deploc["W_AZ"].isna() | (travel_deploc["W_AZ"].str.strip() == "")
    # Delete those rows which have empty start or end time
    travel_deploc_cleaned = travel_deploc[~(empty_sz | empty_az)].copy()

    # Sort by person ID and trip ID
    df_trips_sor = travel_deploc_cleaned.sort_values(by=["HP_ID", "W_ID"])
    # Count number of trips per person as new column
    df_trips_sor["NDEP"] = df_trips_sor.groupby("HP_ID")["W_ID"].transform("count")

    df_deploc = df_trips_sor.rename(columns={
        "HP_ID": "IDENT_IND",
        "zweck": "V2_MMOTIFDES",
        "ST_WOTAG": "V2_TYPJOUR",
        "W_SZ": "V2_MORIHDEP",
        "W_AZ": "V2_MDESHARR",
        "wegkm": "V2_MDISTTOT",
        "hvm": "V2_MTP",
        "W_GEW": "PONDKI",
        "BLAND": "V2_MORIDEP",  # Origin département code
    })
    
    df_deploc["V2_MDESDEP"] = df_deploc["V2_MORIDEP"]  # Destination département code
    df_deploc["IDENT_JOUR"] = 1 # All trips are considered to be on the same day    
    
    # For the origin purpose, we shift destination purpose to create origin purpose within each person
    df_deploc["V2_MMOTIFORI"] = df_deploc.groupby("IDENT_IND")["V2_MMOTIFDES"].shift(1)
    # For the first trip per person (where origin purpose is NaN), set to "home" = 0
    df_deploc["V2_MMOTIFORI"] = df_deploc["V2_MMOTIFORI"].fillna(0)
    df_deploc = df_deploc.drop(columns=["W_ID"])
    
    return df_individu, df_tcm_individu, df_menage, df_tcm_menage, df_deploc

def validate(context):
    for name in ("MiD2017_Personen.csv", "MiD2017_Haushalte.csv", "MiD2017_Wege.csv"):
        if not os.path.exists("%s/mid2017/%s" % (context.config("data_path"), name)):
            raise RuntimeError("File missing from ENTD: %s" % name)

    return [
        os.path.getsize("%s/mid2017/MiD2017_Personen.csv" % context.config("data_path")),
        os.path.getsize("%s/mid2017/MiD2017_Haushalte.csv" % context.config("data_path")),
        os.path.getsize("%s/mid2017/MiD2017_Wege.csv" % context.config("data_path"))
    ]
