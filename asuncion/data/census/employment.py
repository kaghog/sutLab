"""Urban, district-level employed-person targets from Paraguay REDATAM Census 2022.


Expected REDATAM export (sheet 'Output'):
  - Geography headers in Excel columns B/C:  AREA # 0000 / ASUNCION,
    AREA # 1101 / AREGUA, etc.
  - Male/female headers in column B: Hombre, Mujer.
  - Five-year age labels in column C.
  - Column D: Si (employed), E: No, F: Total.
  - The export must be filtered to VIVIENDA.area = 1 (urban).

Output: departement_id, commune_id, sex, age_class, weight
where weight is the observed number of EMPLOYED persons aged 15+.
The IPU prepare stage derives non-employed counts as a population residual.
"""

import os
import re
import warnings

import pandas as pd

AGE_CLASSES = list(range(15, 65, 5)) + [65]  # 15, 20, ..., 60, 65+
RE_AREA = re.compile(r"^AREA\s*#\s*(\d+)\s*$", re.IGNORECASE)
RE_AGE = re.compile(r"^(\d+)\s*a\s*\d+\s*años$", re.IGNORECASE)
RE_OLD = re.compile(r"^(\d+)\s*años\s+y\s+m[aá]s$", re.IGNORECASE)


def configure(context):
    context.config("data_path")
    context.config(
        "asuncion.employment-redatam",
        "census/employment/REDATAM_employment_urban_2022.xlsx",
    )
    context.config("commune_equivalent")
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.data.select_agglomeration")


def _text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _count(value, row_number):
    """REDATAM prints '-' for cells with a count of zero."""
    if pd.isna(value):
        raise ValueError(f"Missing employment count at Excel row {row_number}")
    if isinstance(value, str):
        value = value.strip()
        if value in {"-", "–", "—"}:
            return 0
        value = value.replace(".", "").replace(" ", "")
    n = float(value)
    if not n.is_integer() or n < 0:
        raise ValueError(f"Invalid count {value!r} at Excel row {row_number}")
    return int(n)


def _id_key(value):
    """Normalize zero-padded numeric geography IDs without changing semantics."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return None
    return text.lstrip("0") or "0"


def _parse_redatam(path):
    # A includes metadata (including urban filter); B-F contain the table.
    all_columns = pd.read_excel(
        path, sheet_name="Output", header=None, usecols="A:F",
        dtype=object, engine="openpyxl"
    )
    if all_columns.shape[1] != 6:
        raise ValueError("Unexpected REDATAM structure: expected Excel columns A:F")

    # Verify that this is the urban tabulation, not a national/urban+rural export.
    top = all_columns.iloc[:20].astype(str).to_numpy().ravel()
    if not any(re.search(r"VIVIENDA\.area\s*=\s*1", x, re.I) for x in top):
        raise ValueError("Expected an urban-only export: 'VIVIENDA.area = 1'")
    rows = all_columns.iloc[:, 1:6]

    current_area = None
    current_sex = None
    output = []
    observed_areas = set()

    for excel_row, (b, c, yes, no, total) in enumerate(
        rows.itertuples(index=False, name=None), start=1
    ):
        label_b = _text(b)
        label_c = _text(c)

        if label_b.upper() == "RESUMEN":
            break  # The last section totals all districts. NEVER count it twice.

        area = RE_AREA.fullmatch(label_b)
        if area:
            current_area = area.group(1).zfill(4)
            current_sex = None
            observed_areas.add(current_area)
            continue

        if current_area is None:
            continue

        if label_b in ("Hombre", "Mujer", "Total"):
            current_sex = {"Hombre": "male", "Mujer": "female"}.get(label_b)
            continue

        if current_sex is None:
            continue  # District combined-sex totals and other headings.

        age_match = RE_AGE.fullmatch(label_c) or RE_OLD.fullmatch(label_c)
        if not age_match:
            continue  # Sex totals, 'No Aplica', blank separator rows, etc.

        age = int(age_match.group(1))
        employed = _count(yes, excel_row)
        not_employed = _count(no, excel_row)
        population = _count(total, excel_row)
        if employed + not_employed != population:
            raise ValueError(
                f"REDATAM count mismatch on Excel row {excel_row}: "
                f"{employed} + {not_employed} != {population}"
            )

        if age < 15:
            continue  # Existing IPU assumes under-15s are never employed.

        output.append({
            "area_code": current_area,
            "sex": current_sex,
            "age_class": min(age, 65),
            "weight": float(employed),
            "redatam_population": float(population),
        })

    if not output or "0000" not in observed_areas:
        raise ValueError("No ASUNCIÓN district data found in REDATAM workbook")

    df = pd.DataFrame(output)
    # Combine 65-69, 70-74, ... 100+ into the single 65+ class.
    df = df.groupby(["area_code", "sex", "age_class"], as_index=False)[
        ["weight", "redatam_population"]
    ].sum()
    return df


def execute(context):
    if context.config("commune_equivalent") != "district":
        raise NotImplementedError("REDATAM employment currently supports district aggregation only")

    path = os.path.join(
        context.config("data_path"),
        context.config("asuncion.employment-redatam")
    )
    df = _parse_redatam(path)

    # Determine the pipeline's own IDs from its population stage. This avoids
    # guessing whether IDs are saved as '1101', '01101', etc.
    pop = context.stage("asuncion.data.census.population").copy()
    id_columns = ["departement_id", "district_id", "sex", "age_class", "weight"]
    missing = set(id_columns) - set(pop.columns)
    if missing:
        raise ValueError(f"Population stage lacks required columns: {sorted(missing)}")

    agglomeration = context.stage("asuncion.data.select_agglomeration")
    allowed = set(agglomeration["id"].astype(str))
    lookup = pop[["district_id", "departement_id"]].drop_duplicates().copy()
    lookup["district_id"] = lookup["district_id"].astype(str)
    lookup["departement_id"] = lookup["departement_id"].astype(str)
    lookup = lookup[lookup["district_id"].isin(allowed)].copy()
    lookup["area_key"] = lookup["district_id"].map(_id_key)

    # Fail rather than silently attach employment to the wrong district.
    if lookup["area_key"].isna().any():
        invalid = lookup.loc[lookup["area_key"].isna(), "district_id"].tolist()
        raise ValueError(f"Non-numeric district IDs cannot be matched: {invalid}")
    if lookup["area_key"].duplicated().any():
        raise ValueError("Ambiguous district IDs after normalization")

    observed_keys = set(df["area_code"].map(_id_key))
    missing_areas = sorted(set(lookup["area_key"]) - observed_keys)
    if missing_areas:
        raise ValueError(
            "Selected agglomeration districts are missing in the REDATAM "
            f"file: {missing_areas}. Check the district codes in the population stage."
        )

    df["area_key"] = df["area_code"].map(_id_key)
    df = df.merge(
        lookup[["district_id", "departement_id", "area_key"]],
        on="area_key", how="inner", validate="many_to_one"
    )

    # Retain missing age/sex combinations with a true zero employed count.
    # REDATAM omits some oldest-age rows in the smallest districts.
    grid = pd.MultiIndex.from_product(
        [sorted(lookup["district_id"].unique()), ["male", "female"], AGE_CLASSES],
        names=["district_id", "sex", "age_class"]
    ).to_frame(index=False)
    df = grid.merge(
        df[["district_id", "sex", "age_class", "weight", "redatam_population"]],
        on=["district_id", "sex", "age_class"], how="left", validate="one_to_one"
    )
    df[["weight", "redatam_population"]] = df[[
        "weight", "redatam_population"
    ]].fillna(0)
    df = df.merge(
        lookup[["district_id", "departement_id"]],
        on="district_id", how="left", validate="many_to_one"
    )

    # Check that the population used to compute the IPU non-employed residual
    # is consistent with this REDATAM urban-only population (age >= 15).
    pop = pop[pop["district_id"].astype(str).isin(allowed)].copy()
    pop["district_id"] = pop["district_id"].astype(str)
    pop["age_class"] = pop["age_class"].astype(int)
    pop = pop[pop["age_class"] >= 15]
    pop_totals = (
        pop.groupby(["district_id", "sex", "age_class"], as_index=False)["weight"]
        .sum().rename(columns={"weight": "pipeline_population"})
    )
    check = df.merge(
        pop_totals, on=["district_id", "sex", "age_class"],
        how="left", validate="one_to_one"
    )
    if check["pipeline_population"].isna().any():
        raise ValueError("Missing age/sex population cells for REDATAM districts")

    # Do not change employment counts to compensate for a conflicting
    # population source. Report the conflict, so it can be fixed upstream.
    bad = check[
        (check["redatam_population"] > 0)
        & (abs(check["pipeline_population"] - check["redatam_population"])
           > (0.05 * check["redatam_population"]).clip(lower=10))
    ]
    if not bad.empty:
        warnings.warn(
            f"{len(bad)} district/sex/age cells differ by >5% "
            "between the pipeline population and REDATAM's urban population. "
            "Ensure asuncion.data.census.population uses matching urban "
            "coverage and age classes. Sample: "
            + bad[["district_id", "sex", "age_class", "redatam_population", "pipeline_population"]]
                 .head(4).to_string(index=False),
            RuntimeWarning,
        )
    if (check["weight"] > check["pipeline_population"] + 1e-6).any():
        examples = check[check["weight"] > check["pipeline_population"] + 1e-6]
        raise ValueError(
            "Observed employed count exceeds census population. "
            "Check urban filters and district ID mapping. Sample: "
            + examples.head(4).to_string(index=False)
        )

    result = df.rename(columns={"district_id": "commune_id"})[
        ["departement_id", "commune_id", "sex", "age_class", "weight"]
    ]
    result["age_class"] = result["age_class"].astype(int)
    result["weight"] = result["weight"].astype(float)
    assert not result.isna().any().any()
    print(
        f"Loaded REDATAM urban employment: {result['commune_id'].nunique()} "
        f"districts, {result['weight'].sum():,.0f} employed people aged 15+"
    )
    return result


def validate(context):
    path = os.path.join(
        context.config("data_path"),
        context.config("asuncion.employment-redatam")
    )
    if not os.path.isfile(path):
        raise RuntimeError(f"REDATAM employment data not found: {path}")
    return [os.path.getsize(path)]
