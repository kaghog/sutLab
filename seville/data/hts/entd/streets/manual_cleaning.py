from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts
from geopy.distance import geodesic
import geopy
import time

# IMPORTANT! WHEN DEBUGGING LIMIT REQUEST RATE and size of input dataframe


def configure(context):
    context.config("data_path")
    context.config("seville.street_data_2", "street_data/street_data_2.csv")
    context.config("seville.street_data_3", "street_data/street_data_3.csv")


# TODO: some borroughs are duplicated with more specific descriptions
# these were discarded
# for better results, implement coordinates even for these

BARRIOS = {
    "SAN GIL": (37.401835, -5.990958), 
    "FERIA": (37.397417, -5.990829),
    "SAN JULIAN": (37.397908, -5.984718),
    "SAN VICENTE": (37.395726, -6.001129),
    "ENCARNACIÓN-REGINA": (37.394253, -5.993576),
    "MUSEO": (37.390489, -6.002502),
    "SANTA CRUZ": (37.383212, -5.991739),
    "SAN BARTOLOME": (37.388750, -5.987001),
    "LAS AVENIDAS": (37.410688, -5.985246),
    "DOCTOR BARRAQUER-GRUPO RENFE-POLICLINICO": (37.405943, -5.990598),
    "EL CEREZO": (37.409025, -5.984143),
    "POLIGONO NORTE": (37.412325, -5.979611),
    "HERMANDADES-LA CARRASCA": (37.409325, -5.980401),
    "LOS PRINCIPES-LA FONTANILLA": (37.413688, -5.975766),
    "EL CARMEN": (37.406216, -5.984452),
    "CRUZ ROJA-CAPUCHINOS": (37.400325, -5.981259),
    "PINO FLORES": (37.407883, -5.972629),
    "EL FONTANAL-MARIA AUXILIADORA-CARRETERA DE CARMONA": (37.397458, -5.978654),
    "SAN CARLOS-TARTESSOS": (37.399013, -5.971960),
    "LAS HUERTAS": (37.398603, -5.967634),
    "HUERTA DE SANTA TERESA": (37.390612, -5.972097),
    "SAN PABLO A Y B": (37.393176, -5.966672),
    "SANTA CLARA": (37.395787, -5.950511),
    "SAN PABLO D Y E": (37.399534, -5.958667),
    "LA FLORIDA": (37.386994, -5.983661),
    "LA BUHAIRA": (37.381156, -5.973704),
    "NERVION": (37.382083, -5.966838),
    "CIUDAD JARDIN": (37.377991, -5.963816),
    "HUERTA DE LA SALUD": (37.376736, -5.981738),
    "TABLADILLA-LA ESTRELLA": (37.364895, -5.980433),
    "GIRALDA SUR": (37.374608, -5.976588),
    "EL JUNCAL-HISPALIS": (37.368715, -5.964434),
    "COLORES, ENTREPARQUES": (37.409059, -5.904507),
    "POLIGONO SUR": (37.358823, -5.959868),
    "BAMI": (37.360078, -5.978133),
    "LA OLIVA": (37.365645, -5.969687),
    "LOS PAJAROS": (37.383434, -5.956572),
    "SANTA AURELIA-CANTABRICO-ATLANTICO-LA ROMERIA": (37.382997, -5.948264),
    "JUAN XXIII": (37.376504, -5.948676),
    "ROCHELAMBERT": (37.375577, -5.954306),
    "EL CERRO": (37.372739, -5.959662),
    "LA PLATA": (37.370284, -5.948195),
    "PALMETE": (37.378687, -5.929930),
    "PALACIO DE CONGRESOS, URBADIEZ, ENTREPUENTES": (37.397618, -5.938926),
    "PARQUE ALCOSA-JARDINES DEL EDÉN": (37.409291, -5.928832),
    "TORREBLANCA": (37.387144, -5.899306),
    "SECTOR SUR-LA PALMERA-REINA MERCEDES": (37.360737, -5.987870),
    "HELIOPOLIS": (37.354607, -5.983969),
    "BARRIADA DE PINEDA": (37.350677, -5.970374),
    "BELLAVISTA": (37.333317, -5.977103),
    "CARTUJA": (37.408227, -6.002097),
    "TRIANA OESTE": (37.392777, -6.013598),
    "TRIANA CASCO ANTIGUO": (37.384266, -6.002475),
    "TRIANA ESTE": (37.379574, -6.004741),
    "EL TARDON-EL CARMEN": (37.376136, -6.010577),
    "LOS REMEDIOS": (37.374554, -5.996913),
    "TABLADA": (37.356326, -6.002269),
    "LOS ARCOS": (37.408432, -5.968039),
    "AEROPUERTO VIEJO": (37.412250, -5.954066),
    "VALDEZORRAS": (37.418930, -5.941054),
    "CONSOLACION": (37.417294, -5.967936),
    "SAN DIEGO": (37.411704, -5.968863),
    "BARRIADA PINO MONTANO": (37.423702, -5.968245),
    "TORREBLANCA": (37.386588, -5.902151),
    "LA BACHILLERA": (37.437861, -5.990647),
    "SAN JERONIMO": (37.424229, -5.985360),
    "LA PAZ-LAS GOLONDRINAS": (37.412068, -5.988038),
    "ARENAL": (37.386594, -5.997239),
    "EL PRADO-PARQUE MARIA LUISA": (37.374802, -5.988330),
    "EL JUNCAL-HISPALIS": (37.368635, -5.964297),
}

MUNICIPALITIES= {
    "Paradas": (37.28983570971475, -5.4976628439230835),
    "La Algaba": (37.46139359455287, -6.012231250102343),
    "Carmona": (37.4707779893395, -5.644710440930651),
    "Tomares": (37.37367686919577, -6.04630100056516),
    "Sevilla": (37.392783382893874, -5.987856200329914),
    "La Rinconada": (37.486182193139285, -5.98158989448696),
    "Burguillos": (37.58535602065082, -5.967895113856253),
    "Olivares": (37.41888682807106, -6.157682234637542),
    "Marchena": (37.32805716920212, -5.4168645472017),
    "Herrera": (37.362042687398805, -4.848296470385442),
    
    "": (37.392783382893874, -5.987856200329914), # Empty is set as Seville
    np.nan: (37.392783382893874, -5.987856200329914),
}

def execute(context):

    print("Replacing missing values with manually aquired locations.")


    CSV_PATH = f"{context.config('data_path')}/{context.config('seville.street_data_2')}"

    df_streets = pd.read_csv(CSV_PATH, sep='\t', dtype={"location":str})



#    df_streets = df_streets[
#        (df_streets["location"] == "(None, None)")
#    ]


    # If there is no street, assign by zone if there is zone
    condition = (
        (
            (df_streets["street"] == "LA CALLE NO APARECE EN EL LISTADO")
            | (df_streets["street"] == "")
            | (df_streets["street"] == "-")
            | (df_streets["street"].isna())
            | (df_streets["location"] == "(None, None)") # Remove if you want to manually resolve known streets
        )
        & (
            (df_streets["zone"] != "")
            & (df_streets["zone"] != "-")
            & (~(df_streets["zone"].isna()))
        )
    )

    df_streets.loc[condition, "location"] = df_streets.loc[condition, "zone"].apply(lambda x: BARRIOS[x] if x in BARRIOS.keys() else "(None, None)")

    condition = (
        (df_streets["location"] == "(None, None)")
        | (
            (df_streets["municipality"].isin(MUNICIPALITIES.keys()))
            & (
                (df_streets["zone"] == "")
                | (df_streets["zone"] == "-")
                | (df_streets["zone"].isna())
            )
        )
    )
    df_streets.loc[condition, "location"] = df_streets.loc[condition, "municipality"].apply(lambda x: MUNICIPALITIES[x] if x in MUNICIPALITIES.keys() else "(None, None)")

    print("Checking if locations with missing coordinates exist.")
    df_streets = df_streets[df_streets["municipality"] != "Otros"] # Filter out unknown
    assert df_streets[df_streets["location"] == "(None, None)"].empty


    TARGET_PATH = f"{context.config('data_path')}/{context.config('seville.street_data_3')}"
    # Append the chunk to the file
    df_streets.to_csv(TARGET_PATH, sep='\t', index=False)

    return df_streets
