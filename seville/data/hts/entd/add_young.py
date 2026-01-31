import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

"""
This stage adds all the remaining household members that are not present in the df_persons. All the households members
are necessary for the IPU, we add them all.
"""

def configure(context):
    context.stage("seville.data.hts.entd.cleaned")
    context.stage("seville.locations.education")
    context.stage("seville.data.census.population")
    context.config("random_seed")

def add_young_persons(random, df_persons, df_trips, df_household_members):
    # GENERATE PEOPLE
    df_young_people = df_household_members.copy()

    df_young_people = pd.merge(df_young_people, df_persons[['person_id', 'household_id', 'person_weight']], on='household_id')
    df_young_people = (
        df_young_people
        .merge(
            df_persons[['household_id', 'age', 'sex']],
            on=['household_id', 'age', 'sex'],
            how='left',
            indicator=True
        )
        .query('_merge == "left_only"')
        .drop(columns='_merge')
    )
    
    # generate new person_id and sex for young persons
    df_young_people = df_young_people.reset_index(drop=True) 
    df_young_people['person_id'] = df_young_people.index
    # person_id negative number to mark it as mock data, avoid duplicate zero
    df_young_people['person_id'] = -df_young_people['person_id'] - 1
        
    df_young_people['person_weight'] = 1

    print(df_young_people.head())

    # Set employment
    
    df_young_people.loc[df_young_people['age'] <= 15, 'employed'] = False
    filter = (df_young_people['age'] > 15) & (df_young_people['age'] <= 20)
    df_young_people.loc[filter , 'employed'] = random.choice(
        [True, False],
        size=len(df_young_people[filter]),
        p=[0.3, 0.7]
    )
    filter = df_young_people['age'] > 20
    df_young_people.loc[filter, 'employed'] = random.choice(
        [True, False],
        size=len(df_young_people[filter]),
        p=[0.5, 0.5]
    )

    df_young_people['studies'] = df_young_people['age'].apply(lambda x: True if x <= 20 else False)
    df_young_people['has_license'] = False
    df_young_people['has_pt_subscription'] = False
    df_young_people['number_of_trips'] = -1
    df_young_people['departement_id'] = "41"
    df_young_people['trip_weight'] = df_young_people['person_weight']
    df_young_people['is_passenger'] = True
    df_young_people['socioprofessional_class'] = 8

    print(df_trips.info())

    return df_young_people



def execute(context):
    df_households, df_persons, df_trips, df_household_members = context.stage("seville.data.hts.entd.cleaned")
    random = np.random.RandomState(context.config("random_seed"))


    df_young_persons = add_young_persons(random, df_persons, df_trips, df_household_members)
    
    df_persons = pd.concat([df_persons, df_young_persons[df_persons.columns]])
    
    df_persons["person_weight"] = 1.0
    df_persons["trip_weight"] = 1.0
    
    # CALIBRATE HOUSEHOLD WEIGHTS
    calibration_data = [
        # (age_class, calibration_factor),
        (0, 4),
        (5, 2),
        (10, 1.5),
        (45, 0.65),
        (50, 0.85),
        (60, 0.85),
    ]
    for age_class, factor in calibration_data:
        age_min = age_class
        age_max = age_class + 4
        filter = (df_persons['age'] <= age_max) & (df_persons['age'] >= age_min)
        young_households_ids = df_persons.loc[filter, 'household_id']
        filter = df_households['household_id'].isin(young_households_ids)
        df_households.loc[filter, "household_weight"] *= factor


    return df_households, df_persons, df_trips
