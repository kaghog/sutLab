"""Urban district household-size counts, Paraguay Census 2022 / REDATAM.

Data parsing stage that returns columns:
    departement_id, district_id, household_size, weight

Categories 1..5 are exact numbers of household members; 6 means SIX OR MORE.
Weights are observed household COUNTS (REDATAM 'Casos'), not persons.

Expected source: REDATAM frequency export, sheet Output, variable
'Total de personas', restricted with 'VIVIENDA.area = 1'.
"""

import os
import re
import pandas as pd

AREA_PATTERN = re.compile(r"^AREA\s*#\s*(\d+)\s*$", re.IGNORECASE)
SIZE_BINS = range(1, 7)


def configure(context):
    context.config("data_path")
    context.config(
        "asuncion.household-size-redatam",
        "census/households/REDATAM_household_size_urban_2022.xlsx",
    )
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.data.select_agglomeration")


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _id_key(value):
    """Match official REDATAM district IDs to the IDs used by the pipeline."""
    value = _text(value)
    if value.endswith(".0"):
        value = value[:-2]
    if not value.isdigit():
        raise ValueError(f"Expected numeric district ID, received {value!r}")
    return str(int(value))


def _integer(value, label, excel_row):
    if pd.isna(value):
        raise ValueError(f"Missing {label} in Excel row {excel_row}")
    if isinstance(value, str):
        value = value.strip().replace(" ", "")
        # Strings from REDATAM may use dots as thousands separators.
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
            value = value.replace(".", "")
    number = float(value)
    if not number.is_integer() or number < 0:
        raise ValueError(f"Invalid {label}={value!r} in Excel row {excel_row}")
    return int(number)


def parse_redatam(path):
    """Return raw REDATAM district codes + size classes (including all areas)."""
    raw = pd.read_excel(path, sheet_name="Output", header=None,
                        usecols="A:E", dtype=object)
    if raw.shape[1] != 5:
        raise ValueError("Unexpected workbook: expected five columns (A:E)")

    front = " ".join(_text(x) for x in raw.iloc[:15].to_numpy().ravel())
    if not re.search(r"VIVIENDA\.area\s*=\s*1", front, re.IGNORECASE):
        raise ValueError("REDATAM export is not filtered to urban area (VIVIENDA.area = 1)")
    if "Total de personas" not in front:
        raise ValueError("Expected REDATAM 'Total de personas' frequency table")

    counts = {}
    declared_totals = {}
    current_area = None
    in_summary = False

    for excel_row, values in enumerate(raw.itertuples(index=False, name=None), start=1):
        _, col_b, col_c, _, _ = values
        label = _text(col_b)
        if label.upper() == "RESUMEN":
            in_summary = True
            current_area = None
            continue  # Final summary repeats district counts; never double-count it.

        if in_summary or not label:
            continue

        area_match = AREA_PATTERN.fullmatch(label)
        if area_match:
            current_area = area_match.group(1).zfill(4)
            if current_area in declared_totals or any(k[0] == current_area for k in counts):
                raise ValueError(f"Duplicate REDATAM area {current_area}")
            continue

        if current_area is None:
            continue

        if label.lower() == "total":
            declared_totals[current_area] = _integer(col_c, "area total", excel_row)
            current_area = None
            continue

        # Labels for the six or more members start at 6 and can be much higher.
        # All other headings (including 'Casos') are ignored.
        try:
            size = _integer(col_b, "household size", excel_row)
        except (ValueError, TypeError):
            continue
        if size < 1:
            raise ValueError(f"Unexpected household size {size} at row {excel_row}")
        key = (current_area, min(size, 6))
        counts[key] = counts.get(key, 0) + _integer(col_c, "household count", excel_row)

    if not counts or not declared_totals:
        raise ValueError("No REDATAM district household-size records found")
    if set(area for area, _ in counts) != set(declared_totals):
        raise ValueError("Parsed areas do not match district total rows")

    for area, expected in declared_totals.items():
        actual = sum(counts.get((area, size), 0) for size in SIZE_BINS)
        if expected != actual:
            raise ValueError(f"District {area}: parsed {actual}, workbook total {expected}")

    rows = [
        {"redatam_area_code": area, "household_size": size,
         "weight": float(counts.get((area, size), 0))}
        for area in sorted(declared_totals)
        for size in SIZE_BINS
    ]
    return pd.DataFrame(rows)


def execute(context):
    path = os.path.join(context.config("data_path"),
                        context.config("asuncion.household-size-redatam"))
    df = parse_redatam(path)

    # Resolve source codes to the precise district / department ID types used
    # by the existing eqasim population stage, without assuming padding.
    population = context.stage("asuncion.data.census.population")
    required = {"departement_id", "district_id"}
    if not required.issubset(population.columns):
        raise ValueError(f"Population stage is missing {sorted(required - set(population.columns))}")

    districts = population[["departement_id", "district_id"]].drop_duplicates().copy()
    districts["id_key"] = districts["district_id"].map(_id_key)

    study_area = context.stage("asuncion.data.select_agglomeration")
    selected_keys = set(study_area["id"].map(_id_key))
    districts = districts[districts["id_key"].isin(selected_keys)].copy()
    if not len(districts):
        raise ValueError("No selected AMA districts found in the census population stage")
    if districts["id_key"].duplicated().any():
        raise ValueError("Ambiguous district ID mapping in population stage")

    df["id_key"] = df["redatam_area_code"].map(_id_key)
    missing = selected_keys - set(df["id_key"])
    if missing:
        raise ValueError(f"Missing selected districts in REDATAM workbook: {sorted(missing)}")
    result = df.merge(districts, on="id_key", how="inner", validate="many_to_one")

    # REDATAM geographically identifies 0000 as Asuncion, 11xx as Central.
    # ID values are inherited from the population stage, not reconstructed.
    result = result[["departement_id", "district_id", "household_size", "weight"]]
    result = result.sort_values(["departement_id", "district_id", "household_size"])
    if result.duplicated(["district_id", "household_size"]).any():
        raise ValueError("Duplicate household-size category after district mapping")
    if (result["weight"] < 0).any() or result.isna().any().any():
        raise ValueError("Invalid or missing household-size counts")

    print(f"REDATAM: {len(result)} categories across "
          f"{result['district_id'].nunique()} districts; "
          f"{result['weight'].sum():,.0f} urban households")
    result = result.reset_index(drop=True)

    result = result.rename(columns={"district_id":"commune_id"})
    return result


def validate(context):
    path = os.path.join(context.config("data_path"),
                        context.config("asuncion.household-size-redatam"))
    if not os.path.isfile(path):
        raise RuntimeError(f"REDATAM household-size file not found: {path}")
    return [os.path.getsize(path)]
