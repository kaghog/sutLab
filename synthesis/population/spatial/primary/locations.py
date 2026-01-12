import numpy as np
import pandas as pd
import geopandas as gpd

def configure(context):
    context.stage("synthesis.population.spatial.primary.work")
    context.stage("synthesis.population.spatial.primary.education")


def execute(context):
    df_work = context.stage("synthesis.population.spatial.primary.work")
    df_education = context.stage("synthesis.population.spatial.primary.education")
    
    return df_work, df_education
