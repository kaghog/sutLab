import numpy as np
import pandas as pd

def configure(context):
    context.stage("data.hts.entd.filtered")

def execute(context):
    df_households, df_persons, df_trips = context.stage("data.hts.entd.filtered")

    # ENTD defines multiple weights. For comparison with EGT we keep them in the
    # data set for the previous stages. In this one we override the weight,
    # because the initial person_weight is valid for all persons. However, ENTD
    # does only ask _one_ person per household for trips.. This means that some
    # people are registered with their sociodemographics, but _not_ with the
    # trips. We can identify them by searching for number_of_trips == -1.
    # Now if we simply filter them out, we clearly reduce our weights (because
    # initially they fit EGT and census well). ENTD already defines weights which
    # are used for the "kish", i.e. the responding persons. At this point they
    # are saved in the data set as trip_weight. Probably it could make sense to
    # give this attribute a more descriptive name in the future.

    # 1) Filter persons for which we don't have trip information
    df_persons = df_persons[df_persons["number_of_trips"] >= 0].copy()

    # 2) Override weights with the correct weights for the people which have trip information
    df_persons["person_weight"] = df_persons["trip_weight"]

    # We also add a routed distance, as an appxorimation and for use in the downstream algorithms
    # This is reverse of the original approach,
    df_trips["routed_distance"] = df_trips["euclidean_distance"] * 1.3

    # TODO: remove this part, as this is temporary solution
    print("================================================================================")
    print("============ REMOVE ====================")
    print("================================================================================")
    print("START: FAKING HTS PEOPLE UNDER 15 YEARS OLD")
    df_fake_persons = df_persons.copy()
    df_fake_households = df_households.copy()
    df_fake_trips = df_trips.copy()

    df_fake_persons["age"] = df_fake_persons["age"] % 20
    df_fake_persons["person_id"] = df_fake_persons["person_id"] + len(df_fake_persons) * 2
    df_fake_persons["household_id"] = df_fake_persons["household_id"] + len(df_fake_persons) * 2
    df_fake_households["household_id"] = df_fake_households["household_id"] + len(df_fake_persons) * 2
    df_fake_trips["person_id"] = df_fake_trips["person_id"] + len(df_fake_persons) * 2
    df_fake_trips["trip_id"] = df_fake_trips["trip_id"] + df_fake_trips["trip_id"]


    df_persons = pd.concat([df_persons, df_fake_persons], ignore_index=True)
    df_households = pd.concat([df_households, df_fake_households], ignore_index=True)
    df_trips = pd.concat([df_trips, df_fake_trips], ignore_index=True)
    print("FINISHED: FAKING HTS PEOPLE UNDER 15 YEARS OLD")


    return df_households, df_persons, df_trips
