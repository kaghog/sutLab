
import pandas as pd
import os
import numpy as np
import geopandas as gpd


def configure(context):
    context.stage("seville.gravity.model_new")

    context.stage("seville.data.hts.entd.trip_distance")

    context.stage("seville.data.spatial.iris")

    context.stage("seville.data.census.employees")

    context.config("analysis_path")

    context.stage("seville.gravity.distance_matrix")
    context.stage("seville.ipu.attributed")
    context.stage("seville.data.census.employees")


    context.stage("seville.locations.education")


def execute(context):
    gravity_od = context.stage("seville.gravity.model_new")
    spatial_df = context.stage("seville.data.spatial.iris")
    _, _, df_trips, _ = context.stage("seville.data.hts.entd.trip_distance")
    employees_df = context.stage("seville.data.census.employees")


    gravity_od.to_pickle(f"{context.config('analysis_path')}/gravity_od.pkl")
    spatial_df.to_pickle(f"{context.config('analysis_path')}/spatial_df.pkl")
    df_trips.to_pickle(f"{context.config('analysis_path')}/df_trips.pkl")

    employees_df.to_pickle(f"{context.config('analysis_path')}/employees.pkl")


    df_distances = context.stage("seville.gravity.distance_matrix")
    df_population = context.stage("seville.ipu.attributed")
    df_employees = context.stage("seville.data.census.employees")

    df_distances.to_pickle(f"{context.config('analysis_path')}/df_distances_od_model.pkl")
    df_population.to_pickle(f"{context.config('analysis_path')}/df_population_od_model.pkl")
    df_employees.to_pickle(f"{context.config('analysis_path')}/df_employees_od_model.pkl")

    edu_loc = context.stage("seville.locations.education")
    edu_loc.to_file(f"{context.config('analysis_path')}/edu_loc.gpkg")
