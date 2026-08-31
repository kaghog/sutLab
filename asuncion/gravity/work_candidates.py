import pandas as pd
import numpy as np
import geopandas as gpd

"""
DESCRIPTION:

"""

def configure(context):
    context.stage("data.od.weighted")

    context.stage("synthesis.locations.education")
    context.stage("synthesis.locations.work")
    context.stage("asuncion.gravity.od_zones")

    context.stage("synthesis.population.spatial.home.zones")
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.trips")

    context.config("output_path")
    context.config("random_seed")
    context.config("education_location_source", "bpe")


def sample_destination_municipalities(context, arguments):
    # Load data
    origin_id, count, random_seed = arguments
    df_od = context.data("df_od")

    # Prepare state
    random = np.random.RandomState(random_seed)
    df_od = df_od[df_od["origin_id"] == origin_id].copy()

    # Sample destinations
    df_od["count"] = random.multinomial(count, df_od["weight"].values)
    df_od = df_od[df_od["count"] > 0]

    context.progress.update()
    return df_od[["origin_id", "destination_id", "count"]]

def sample_locations(context, arguments):
    # Load data
    destination_id, random_seed = arguments
    df_locations, df_flow = context.data("df_locations"), context.data("df_flow")

    # Prepare state
    random = np.random.RandomState(random_seed)
    df_locations = df_locations[df_locations["commune_id"] == destination_id]
    
    # Determine demand
    df_flow = df_flow[df_flow["destination_id"] == destination_id]
    count = df_flow["count"].sum()

    # Sample destinations
    weight = np.ones((len(df_locations),)) / len(df_locations)

    if "weight" in df_locations:
        weight = df_locations["weight"].values / df_locations["weight"].sum()
    
    location_counts = random.multinomial(count, weight)
    location_ids = df_locations["location_id"].values
    location_ids = np.repeat(location_ids, location_counts)

    # Shuffle, as otherwise it is likely that *all* copies 
    # of the first location id go to the first origin, and so on
    random.shuffle(location_ids)

    # Construct a data set for all commutes to this zone
    origin_id = np.repeat(df_flow["origin_id"].values, df_flow["count"].values)

    df_result = pd.DataFrame.from_records(dict(
        origin_id = origin_id,
        location_id = location_ids
    ))
    df_result["destination_id"] = destination_id

    return df_result

def process(context, purpose, random, df_persons, df_od, df_locations,step_name):
    df_persons = df_persons[df_persons["has_%s_trip" % purpose]]

    # Sample commute flows based on population
    df_demand = df_persons.groupby("commune_id").size().reset_index(name = "count")
    df_demand["random_seed"] = random.randint(0, int(1e6), len(df_demand))
    df_demand = df_demand[["commune_id", "count", "random_seed"]]
    df_demand = df_demand[df_demand["count"] > 0]

    df_flow = []

    with context.progress(label = "Sampling %s municipalities" % step_name, total = len(df_demand)) as progress:
        with context.parallel(dict(df_od = df_od)) as parallel:
            for df_partial in parallel.imap_unordered(sample_destination_municipalities, df_demand.itertuples(index = False, name = None)):
                df_flow.append(df_partial)

    df_flow = pd.concat(df_flow).sort_values(["origin_id", "destination_id"])

    # Sample destinations based on the obtained flows
    unique_ids = df_flow["destination_id"].unique()
    random_seeds = random.randint(0, int(1e6), len(unique_ids))

    df_result = []

    with context.progress(label = "Sampling %s destinations" % purpose, total = len(df_demand)) as progress:
        with context.parallel(dict(df_locations = df_locations, df_flow = df_flow)) as parallel:
            for df_partial in parallel.imap_unordered(sample_locations, zip(unique_ids, random_seeds)):
                df_result.append(df_partial)

    df_result = pd.concat(df_result).sort_values(["origin_id", "destination_id"])

    return df_result[["origin_id", "destination_id", "location_id"]]


# Transform commune_id into macrozone_id
# This is necessary for compatibility reasons with gravity model, because census uses census sections but
# gravity model uses macrozones which include municipalities and asuncion's districts

def fix_commune(df_persons, commune2macrozone_map):

    df_fixed = df_persons.merge(commune2macrozone_map[["commune_id", "macrozone_id"]])

    print(df_persons["commune_id"].value_counts())
    print(df_fixed["commune_id"].value_counts())
    assert len(df_persons) == len(df_fixed), f"{len(df_persons)} == {len(df_fixed)}"


    df_fixed['commune_id'] = df_fixed['macrozone_id']

    return df_fixed


def execute(context):
    # Prepare population data
    df_persons = context.stage("synthesis.population.enriched")[["person_id", "household_id", "age_range"]].copy()
    df_trips = context.stage("synthesis.population.trips")

    df_persons["has_work_trip"] = df_persons["person_id"].isin(df_trips[
        (df_trips["following_purpose"] == "work") | (df_trips["preceding_purpose"] == "work")
    ]["person_id"])
    df_persons = df_persons[df_persons["has_work_trip"] == True]
    
    df_homes = context.stage("synthesis.population.spatial.home.zones")
    df_persons = pd.merge(df_persons, df_homes, on = "household_id")

    # Prepare spatial data
    df_work_od = context.stage("data.od.weighted")

    # commune to macrozone map
    _, _, _, commune2macrozone_map = context.stage("asuncion.gravity.od_zones")


    # Align commune_id with zones used for OD matrix
    df_persons = fix_commune(df_persons, commune2macrozone_map)

    df_work_locations = context.stage("synthesis.locations.work")
    df_work_locations = fix_commune(df_work_locations, commune2macrozone_map)

    # Sampling
    random = np.random.RandomState(context.config("random_seed"))

    # Work Sampling
    df_work_locations["weight"] = df_work_locations["employees"]
    df_work = process(context, "work", random, df_persons,
        df_work_od, df_work_locations, "work"
    )


    return df_work, df_persons[["person_id", "household_id", "age_range", "commune_id", "has_work_trip"]]
