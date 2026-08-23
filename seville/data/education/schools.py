import geopandas as gpd
import pandas as pd
import os
import numpy as np
from shapely.geometry import Point

"""
This stage loads education locations of Seville.
"""

def configure(context):
    context.config("data_path")
    context.config("seville.edu_school", "education/da_alumnado.csv")


KINDERGARTEN = [
    "pub_adh_inf1", "pub_noadh_inf1", "priv_adh_inf1",
    "priv_noadh_inf1", "pub_inf2", "priv_c_inf2",
    "priv_noc_inf2",
]

ELEMENTARY = [
    "pub_pri", "priv_c_pri", "priv_noc_pri",
]

HIGHSCHOOL = [
    "pub_eso", "priv_c_eso", "priv_noc_eso",
    "pub_bach_ord", "priv_c_bach_ord", "priv_noc_bach_ord",
    "pub_fpbasica", "priv_c_fpbas", "priv_noc_fpbas",
    "pub_fpgm_ord", "priv_c_fpgm_ord", "priv_noc_fpgm_ord",
    "pub_fpgm_adul", "pub_fpgm_semi_dist", "priv_noc_fpgm_semi_dist",
    "pub_fpgs_ord", "priv_c_fpgs_ord", "priv_noc_fpgs_ord",
    "pub_fpgs_adul", "priv_noc_fpgs_adul",
    "pub_fpgs_semi_dist", "priv_noc_fpgs_semi_dist",
]


def execute(context):


    FILE_PATH = f'{context.config("data_path")}/{context.config("seville.edu_school")}'
    df = pd.read_csv(FILE_PATH, sep=";", quotechar='"', decimal=",", encoding="latin1")

    records = []

    
    OTHER = [
        c for c in df.columns
        if c.startswith(("pub_", "priv_", "pri_"))
        and c not in KINDERGARTEN
        and c not in ELEMENTARY
        and c not in HIGHSCHOOL
    ]

    groups = {
        "kindergarten": KINDERGARTEN,
        "elementary": ELEMENTARY,
        "highschool": HIGHSCHOOL,
        "other": OTHER,
    }

    for _, row in df.iterrows():

        point = Point(
            float(str(row["N_LONGITUD"]).replace(",", ".")),
            float(str(row["N_LATITUD"]).replace(",", "."))
        )

        for school_type, columns in groups.items():

            students = row[columns].fillna(0).sum()

            if students == 0:
                continue

            records.append({
                "code": row["codigo"],
                "location_name": row["D_ESPECIFICA"],
                "education_type": school_type,
                "students": students,
                "geometry": point
            })

    gdf_schools = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    gdf_schools["weight"] = gdf_schools["students"]

    return gdf_schools

def validate(context):

    filenames = [
        "seville.edu_school",
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Education location data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list

