import geopandas as gpd
import pandas as pd
import os
import numpy as np

"""
This stage loads education locations of Seville.
"""

def configure(context):
    context.config("data_path")
    context.config("seville.edu_school", "education_locations/centroeducativo.gpkg")
    context.config("seville.edu_faculty", "education_locations/facultad.gpkg")
    context.config("seville.edu_campus", "education_locations/campus.gpkg")
    context.config("seville.edu_uni", "education_locations/universidad.gpkg")
    
    context.stage("seville.data.buildings")
    context.stage("seville.data.spatial.iris")

EDU_CATERGORY_MAP = {
    # Kindergarten
    "Centro de Educación Infantil": "kindergarten",
    "Escuela Infantil": "kindergarten",
    
    # Kindergarten + Elementary
    "Colegio de Educación Infantil y Primaria": "elementary school",
    
    # Elementary School
    "Colegio de Educación Primaria": "elementary school",
    "Colegio Público Rural": "elementary school",
    
    # High School
    "Instituto de Educación Secundaria": "highschool",
    "Sección de Educación Secundaria Obligatoria": "highschool",
    "Instituto de Enseñanza a Distancia de Andalucía": "highschool",
    "Centro Docente Privado": "highschool",
    "Centro Docente Privado Extranjero": "highschool",
    "Escuela Hogar": "highschool",
    
    # University / Higher Education
    "Conservatorio Superior de Música": "university",
    "Escuela Superior de Arte Dramático": "university",
    "Centro Autorizado de Enseñanzas Artísticas Superiores de Diseño": "university",

    # Other / Specialized
    "Sección de Educación Permanente": "other",
    "Centro de Educación Permanente": "other",
    "Instituto Provincial de Educación Permanente": "other",
    "Escuela Oficial de Idiomas": "other",
    "Escuela Municipal de Música": "other",
    "Escuela Municipal de Música y Danza": "other",
    "Escuela Autorizada de Música": "other",
    "Escuela Autorizada de Música y Danza": "other",
    "Conservatorio Elemental de Música": "other",
    "Conservatorio Profesional de Música": "other",
    "Conservatorio Profesional de Danza": "other",
    "Escuela de Arte": "other",
    "Centro Autorizado de Enseñanzas Deportivas": "other",
    "Centro Autorizado de Enseñanzas Artísticas Profesionales de Artes Plásticas": "other",
    "Centro Específico de Educación Especial": "other",
    "Centro Docente Privado de Educación Especial": "other",
    "Aulas hospitalarias": "other",
    "Centro de Convenio": "other",
}


def execute(context):
    df_zones = context.stage("seville.data.spatial.iris")

    

    # Import
    FILE_URL = f"{context.config('data_path')}/{context.config('seville.edu_uni')}"
    university_gdf = gpd.read_file(FILE_URL)

    FILE_URL = f"{context.config('data_path')}/{context.config('seville.edu_faculty')}"
    faculty_gdf = gpd.read_file(FILE_URL)

    FILE_URL = f"{context.config('data_path')}/{context.config('seville.edu_school')}"
    school_gdf = gpd.read_file(FILE_URL)


    # Clean university-like locations
    university_gdf = pd.concat([university_gdf, faculty_gdf])
    university_gdf = university_gdf[university_gdf['provincia'] == 'Sevilla']

    university_gdf['education_type'] = 'university'
    university_gdf = university_gdf[['education_type', 'geometry']]

    # Resolve campuses
    # because campuses are represented only as one big shape and not by individual buildings
    # we do following things:
    # 1. remove all university locations within the campus area
    # 2. we take all the buildings from the campus area
    # 3. and we assign each of those buildings as university locations

    buildings = context.stage("seville.data.buildings")
    FILE_URL = f"{context.config('data_path')}/{context.config('seville.edu_campus')}"
    campus_gdf = gpd.read_file(FILE_URL)
    
    campus_buildings = gpd.sjoin(buildings, campus_gdf[["geometry"]], predicate="intersects", how="inner")
    campus_buildings = campus_buildings[campus_buildings["type"]!="1_residential"] # exclude home locations
    campus_buildings['geometry'] = campus_buildings['shape_geometry']
    campus_buildings = campus_buildings.explode(index_parts=True).reset_index(drop=True)
    campus_buildings['education_type'] = 'university'
    campus_buildings = campus_buildings[['education_type', 'geometry']]

    campus_union = campus_gdf.union_all()
    university_gdf = university_gdf[
        ~university_gdf.geometry.intersects(campus_union)
    ]

    university_gdf = pd.concat([university_gdf, campus_buildings])


    # Clean schools
    school_gdf = school_gdf[school_gdf['provincia'] == 'Sevilla']
    school_gdf['tipo'].value_counts()


    school_gdf['education_type'] = school_gdf['tipo'].map(EDU_CATERGORY_MAP)
    school_gdf = school_gdf[school_gdf['education_type']!='other']
    school_gdf = school_gdf[['education_type', 'geometry']]


    # Merge All sources
    education_df = pd.concat([university_gdf, school_gdf])
    education_df = education_df[['education_type', 'geometry']]



    # Attributes
    start_index = 0
    education_df["building_id"] = np.arange(len(education_df)) + start_index
    start_index += len(education_df) + 1
    education_df["weight"] = 1.0 # weight is same for all locations, because most of the locations have no area to use as weight
    education_df["geometry"] = education_df['geometry'].centroid



    # Impute spatial identifiers
    education_df = gpd.sjoin(education_df, df_zones[["geometry", "commune_id", "iris_id"]], 
        how = "left", predicate = "within").reset_index(drop = True).drop(columns = ["index_right"])

    education_df = education_df.dropna(subset=["commune_id", "iris_id"])
    
    df_combined = []
    df_combined.append(education_df[[
        "building_id", "commune_id", "iris_id", "geometry", "education_type", "weight"
    ]])
    
    df_combined = gpd.GeoDataFrame(pd.concat(df_combined), crs = df_combined[0].crs)

    required_zones = set(df_zones["commune_id"].unique())
    available_zones = set(df_combined["commune_id"].unique())
    missing_zones = required_zones - available_zones

    if len(missing_zones) > 0:
        print("Adding {} centroids as buildings for missing municipalities".format(len(missing_zones)))
        df_missing = df_zones[df_zones["commune_id"].isin(missing_zones)][["commune_id", "iris_id", "geometry"]].copy()
        df_missing["geometry"] = df_missing["geometry"].centroid
        df_missing["building_id"] = np.arange(len(df_missing)) + start_index
        df_missing["weight"] = 0.001 # significantly smaller weight to prefer real education facilities
        df_missing["education_type"] = "unknown"


        df_combined = pd.concat([df_combined, df_missing])

    # Identifiers
    df_combined["location_id"] = np.arange(len(df_combined))
    df_combined["location_id"] = "edu_" + df_combined["location_id"].astype(str)


    return df_combined

def validate(context):

    filenames = [
        "seville.edu_school",
        "seville.edu_faculty",
        "seville.edu_campus",
        "seville.edu_uni",
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Education location data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list

