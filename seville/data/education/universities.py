import geopandas as gpd
import pandas as pd
import os
import numpy as np

"""
This stage loads education locations of Seville.
"""

def configure(context):
    context.config("data_path")
    context.config("seville.edu_faculty", "education/facultad.gpkg")
    context.config("seville.edu_campus", "education/campus.gpkg")
    context.config("seville.edu_uni", "education/universidad.gpkg")
    
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


def import_locations(context):

    

    # Import
    FILE_URL = f"{context.config('data_path')}/{context.config('seville.edu_uni')}"
    university_gdf = gpd.read_file(FILE_URL)

    FILE_URL = f"{context.config('data_path')}/{context.config('seville.edu_faculty')}"
    faculty_gdf = gpd.read_file(FILE_URL)

    FILE_URL = f"{context.config('data_path')}/{context.config('seville.edu_campus')}"
    campus_gdf = gpd.read_file(FILE_URL)

    # Clean university-like locations
    university_gdf = pd.concat([university_gdf, faculty_gdf, campus_gdf])
    university_gdf = university_gdf[university_gdf['provincia'] == 'Sevilla']
    university_gdf = university_gdf.rename(columns={"nombre":"location_name"})

    university_gdf['education_type'] = 'university'
    university_gdf = university_gdf[['education_type', 'geometry', 'location_name']]



    # Attributes
    university_gdf["weight"] = 1.0 # weight is same for all locations, because most of the locations have no area to use as weight
    university_gdf["geometry"] = university_gdf['geometry'].centroid

    


    return university_gdf


# for source see documentation, no official dataset found so we use information
# that is available on the university websites etc.
UNIVERSITY_STUDENT_COUNTS = [
    # Universidad de Sevilla
    ("F. DE BELLAS ARTES", "Facultad de Bellas Artes – Universidad de Sevilla", 1048),

    # Biology split across the two buildings
    ("F. DE BIOLOGÍA", "Facultad de Biología – Edificio Rojo – Universidad de Sevilla", 596),
    ("F. DE BIOLOGÍA", "Facultad de Biología – Edificio Verde – Universidad de Sevilla", 596),

    ("F. DE CIENCIAS DE LA EDUCACION", "Facultad de Ciencias de la Educación – Universidad de Sevilla", 4093),
    ("F. DE CIENCIAS DEL TRABAJO", "Facultad de Ciencias del Trabajo – Universidad de Sevilla", 1223),
    ("F. DE FARMACIA", "Facultad de Farmacia – Universidad de Sevilla", 1904),
    ("F. DE FILOLOGÍA", "Facultad de Filología – Universidad de Sevilla", 2071),
    ("F. DE FILOSOFÍA", "Facultad de Filosofía – Universidad de Sevilla", 550),
    ("F. DE FÍSICA", "Facultad de Física – Universidad de Sevilla", 701),
    ("F. DE GEOGRAFÍA E HISTORIA", "Facultad de Geografía e Historia – Universidad de Sevilla", 2181),
    ("F. DE MATEMÁTICAS", "Facultad de Matemáticas – Universidad de Sevilla", 1298),
    ("F. DE MEDICINA", "Facultad de Medicina – Universidad de Sevilla", 2093),
    ("F. DE ODONTOLOGÍA", "Facultad de Odontología – Universidad de Sevilla", 476),
    ("F. DE PSICOLOGÍA", "Facultad de Psicología – Universidad de Sevilla", 1212),
    ("F. DE QUÍMICA", "Facultad de Química – Universidad de Sevilla", 758),
    ("F. DE TURISMO Y FINANZAS", "Facultad de Turismo y Finanzas – Universidad de Sevilla", 2375),

    ("E.T.S DE ARQUITECTURA", "Escuela Técnica Superior de Arquitectura – Universidad de Sevilla", 1629),
    ("E.T.S DE INGENIERÍA", "Escuela Técnica Superior de Ingeniería – Universidad de Sevilla", 4735),
    ("E.T.S. DE INGENIERÍA AGRONÓMICA", "Escuela Técnica Superior de Ingeniería Agronómica – Universidad de Sevilla", 974),
    ("E.T.S DE INGENIERÍA DE EDIFICACIÓN", "Escuela Técnica Superior de Ingeniería de Edificación – Universidad de Sevilla", 755),
    ("E.T.S. DE INGENIERÍA INFORMÁTICA", "Escuela Técnica Superior de Ingeniería Informática – Universidad de Sevilla", 2906),
    ("E. POLITÉCNICA SUPERIOR", "Escuela Politécnica Superior – Universidad de Sevilla", 2754),

    ("CENTRO DE ESTUDIOS UNIVERSITARIOS CARDENAL SPÍNOLA", "Centro de Estudios Universitarios Cardenal Spínola", 867),
    ("CENTRO UNIVERSITARIO DE OSUNA", None, 1719),   # not in POI dataset
    ("CENTRO DE ENFERMERÍA CRUZ ROJA", "Centro de Enfermería de la Cruz Roja", 276),
    ("CENTRO DE ENFERMERÍA SAN JUAN DE DIOS", "Centro de Enfermería San Juan de Dios", 87),
    ("CENTRO UNIVERSITARIO EUSA", "Centro Universitario EUSA", 555),

    ("UNIVERSIDAD PABLO DE OLAVIDE", "Campus Pablo de Olavide", 13007),
    ("UNIVERSIDAD CEU FERNANDO III", "Universidad CEU Fernando III", 1200),
    ("UNIVERSIDAD LOYOLA (SEVILLA CAMPUS)", "Universidad de Loyola – Campus Sevilla", 3200),
]

def add_student_counts(context, df_universities):
    df_university_students = pd.DataFrame(UNIVERSITY_STUDENT_COUNTS, columns=["name1", "location_name", "students"])

    df_universities = df_universities.merge(df_university_students)
    df_universities["weight"] = df_universities["students"]

    return df_universities




def execute(context):
    df_universities = import_locations(context)
    df_weighted = add_student_counts(context, df_universities)
    return df_weighted


def validate(context):

    filenames = [
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

