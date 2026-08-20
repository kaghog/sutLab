import pandas as pd
import os
import numpy as np
import geopandas as gpd
import re

"""
This stage outputs the number of employees per district.
"""

def configure(context):
    context.config("data_path")

    context.config("asuncion.employees_data", "census/employees/49d4bCEN2011_DIST_CUADRO_1.xlsx")

def execute(context):
    EXCEL_PATH = "{}/{}".format(context.config("data_path"),context.config("asuncion.employees_data"))
    
    print(f"Loading employees data from {EXCEL_PATH}")

    raw = pd.read_excel(EXCEL_PATH, header=None)

    # Department headers: uppercase rows with a total number of employees
    dept_mask = (
        raw[0].notna()
        & raw[1].isna()
        & raw[3].notna()
        & raw[0].astype(str).str.upper().eq(raw[0].astype(str))
    )

    dept_rows = raw.index[
        dept_mask & ~raw[0].astype(str).eq("TOTAL")
    ].tolist()

    records = []
    current_dept = None

    for i, row in raw.iterrows():
        value = row[0]

        if pd.isna(value):
            continue

        name = str(value).strip()

        # New department
        if i in dept_rows:
            current_dept = name.title()

            # Asunción is both a department-level entity and its district
            if name == "ASUNCIÓN":
                records.append([
                    current_dept,
                    "Asunción",
                    row[3]
                ])

            continue

        # District total rows
        if (
            current_dept
            and pd.isna(row[1])
            and pd.notna(row[3])
            and name not in {"Industria", "Comercio", "Servicios"}
            and not re.fullmatch(r"\d+", name)
        ):
            records.append([
                current_dept,
                name,
                row[3]
            ])

    employees = pd.DataFrame(
        records,
        columns=["departement", "district", "employees"]
    )

    employees["weight"] = (
        pd.to_numeric(employees["employees"], errors="coerce")
          .round()
          .astype("Int64")
    )

    return employees

def validate(context):
    filenames = [
        "asuncion.employees_data",
    ]

    FILE_LIST = [f"{context.config('data_path')}/{context.config(filename)}" for filename in filenames]

    for FILE in FILE_LIST:
        if not os.path.exists(FILE):
            raise RuntimeError(f"Census data is not available at location {FILE}")

    size_list = [os.path.getsize(FILE) for FILE in FILE_LIST]

    return size_list
