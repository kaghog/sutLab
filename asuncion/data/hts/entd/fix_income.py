"""HTS income repair stage for asuncion.

returns a pickleable callable object serving as 
function(households, persons) -> (households, persons) that:
  * converts income components to numeric and converts special missing codes to NaN;
  * replaces blank components for children <15 with structural zeros, but NEVER
    replaces a special missing code with a structural zero;
  * imputes only missing PRIMARY and SECONDARY employment-income components,
    by weighted, conditional hot-deck sampling from observed HTS respondents;
  * recalculates person and household job-only income and total income in PYG;
  * preserves NaN for a total if ANY non-employment component is unresolved;
  * retains a report of which persons/households were imputed or remain missing.

This hot-deck is stochastic and conditional on the observed respondents. It is 
not a model-based imputation, and it does not attempt to predict income for a person 
with no observed donors. It is intended to repair the HTS data for use in synthetic 
population generation, not to produce a statistically valid estimate of the true income distribution.
The callable requires both dataframes in the intermediate format from `raw.py`, with
"""

from functools import partial
import numpy as np
import pandas as pd

NONJOB_INCOME_MAP = {
    "IngresoAlquilerRenta": "other_income_rent",
    "IngresoInteresDividendos": "other_income_interest_dividends",
    "IngresoAyudaFamiliar": "other_income_family_assistance",
    "IngresoAyudaFamiliarExt": "other_income_family_assistance_abroad",
    "IngresoDivorcioCuidadoHijos": "other_income_child_support",
    "IngresoJubilacion": "other_income_retirement",
    "IngresoTekopora": "other_income_tekopora",
    "IngresoPension": "other_income_pension",
    "IngresoAdultoMayor": "other_income_older_adult_benefit",
    "IngresoOtros": "other_income_other",
}
JOB_COLUMNS = ["primary_job_income", "secondary_job_income"]
OTHER_COLUMNS = list(NONJOB_INCOME_MAP.values())
INCOME_COLUMNS = JOB_COLUMNS + OTHER_COLUMNS
MISSING_CODES = (99999999999, 9999999999)  # HTS 2021 uses 11 nines.


def configure(context):
    context.config("random_seed")
    context.config("asuncion.income_imputation_min_donors", 10)


def execute(context):
    # `partial` of a module-level function is serializable unlike many nested
    # closures and can be used as `context.stage(...)(households, persons)`.
    return partial(
        fix_income,
        seed=int(context.config("random_seed")),
        min_donors=int(context.config("asuncion.income_imputation_min_donors")),
    )


def _to_money(series):
    """Parse input numeric strings without interpreting unknowns as zeros."""
    text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(text, errors="coerce").astype(float)
    return numeric.mask(numeric.isin(MISSING_CODES))


def _weighted_donor_sample(target, donors, keys, rng):
    """Sample one observed donor amount by survey weight."""
    eligible = donors
    for key in keys:
        value = target[key]
        if pd.isna(value):
            return None
        eligible = eligible.loc[eligible[key].eq(value)]
        if eligible.empty:
            return None
    if eligible.empty:
        return None
    weights = eligible["_donor_weight"].to_numpy(dtype=float)
    positive = np.isfinite(weights) & (weights > 0)
    eligible = eligible.loc[positive]
    weights = weights[positive]
    if not len(eligible):
        return None
    return eligible, weights / weights.sum()


def _impute_column(people, col, rng, min_donors):
    """Fill missing job component with an observed value from an appropriate donor."""
    target_mask = people[col].isna() & people["age"].ge(15)
    # Blank components of minors are structural zeros, processed before this step.
    # A rare *special-coded* missing minor employment income remains unresolved;
    # it is deliberately not silently converted to zero.
    donors = people.loc[people[col].notna() & people["_donor_weight"].gt(0)]
    levels = [
        ["_employed", "_department", "_sex", "_age_group", "_car"],
        ["_employed", "_department", "_sex", "_age_group"],
        ["_employed", "_department", "_sex"],
        ["_employed", "_department"],
        ["_employed"],
    ]
    counts = {"imputed": 0, "fallback": 0, "unresolved": 0}
    flag = pd.Series(False, index=people.index)

    for index, person in people.loc[target_mask].iterrows():
        selected = None
        chosen_level = None
        # The employment-status variable is a HARD restriction at every level.
        # Relax car, age, sex and department in that order if needed.
        for level_no, keys in enumerate(levels):
            pool = _weighted_donor_sample(person, donors, keys, rng)
            if pool is None:
                continue
            values, probabilities = pool
            if len(values) < min_donors and level_no < len(levels) - 1:
                continue
            selected = rng.choice(values[col].to_numpy(float), p=probabilities)
            chosen_level = level_no
            break

        if selected is None:
            counts["unresolved"] += 1
            continue
        people.at[index, col] = float(selected)
        flag.at[index] = True
        counts["imputed"] += 1
        if chosen_level:
            counts["fallback"] += 1
    return flag, counts


def _strict_household_sum(persons, col):
    """Any person-level NaN propagates to household total, never skipped."""
    return persons.groupby("household_id", sort=False)[col].agg(
        lambda values: values.sum(min_count=len(values))
    )


def fix_income(households, persons, *, seed=1234, min_donors=10):
    """Return corrected (households, persons) without mutating either input.

    Requires both dataframes in the intermediate format from `raw.py`, with
    person age/sex/employment already cleaned as shown in `cleaned.py`.
    All created `*_pyg` columns are in FULL PYG. Divide by 1,000 in cleaned.py
    to match its existing `household_income` units (thousand PYG).
    """
    if min_donors < 1:
        raise ValueError("min_donors must be >= 1")
    h = households.copy()
    p = persons.copy()
    required_h = {"household_id", "departement_id", "number_of_cars"}
    required_p = {"person_id", "household_id", "age", "sex", "employed"} | set(INCOME_COLUMNS)
    miss_h = required_h - set(h.columns)
    miss_p = required_p - set(p.columns)
    if miss_h or miss_p:
        raise ValueError(
            f"Income fix needs household columns {sorted(miss_h)} and person columns {sorted(miss_p)}. "
            "The ten non-job fields must first be retained in raw.py using NONJOB_INCOME_MAP."
        )
    if h.household_id.duplicated().any() or p.person_id.duplicated().any():
        raise ValueError("Household/person IDs are not unique")
    if not p.household_id.isin(h.household_id).all():
        raise ValueError("Some HTS people do not have a household")

    p["age"] = pd.to_numeric(p["age"], errors="raise").astype(int)
    p["_employed"] = p["employed"].astype(bool) if p["employed"].dtype==bool else p["employed"].astype("string").str.lower().isin(("1", "true"))
    p["_sex"] = p["sex"].astype("string").str.lower().replace({"hombre": "male", "mujer": "female"})
    p["_age_group"] = pd.cut(
        p.age, bins=[-1, 17, 24, 34, 49, 64, np.inf],
        labels=["0-17", "18-24", "25-34", "35-49", "50-64", "65+"],
    ).astype("string")
    # Use household department and vehicles to avoid any discrepancy between
    # original person and household department columns.
    h_aux = h[["household_id", "departement_id", "number_of_cars"]].copy()
    h_aux["_department"] = h_aux.departement_id.astype("string").str.strip().str.zfill(2)
    h_aux["_car"] = pd.to_numeric(h_aux.number_of_cars, errors="coerce").gt(0)
    p = p.merge(h_aux[["household_id", "_department", "_car"]], how="left", on="household_id", validate="many_to_one")
    if "person_weight" in p:
        p["_donor_weight"] = pd.to_numeric(p.person_weight, errors="coerce")
    else:
        p["_donor_weight"] = 1.0  # Only for standalone/synthetic-data tests.
    if p["_donor_weight"].isna().any() or p["_donor_weight"].le(0).any():
        raise ValueError("Person expansion weights must be positive numeric values")

    unknown_before = pd.DataFrame(index=p.index)
    blank_minor_count = 0
    for col in INCOME_COLUMNS:
        before = pd.to_numeric(
            p[col].astype("string").str.strip().str.replace(",", ".", regex=False),
            errors="coerce"
        ).astype(float)
        unknown_before[col] = before.isin(MISSING_CODES)
        parsed = before.mask(before.isin(MISSING_CODES))
        # Survey did not ask children younger than 15 about income.
        # These are *blank* and should not be treated as unresolved nonresponse.
        structural = p.age.lt(15) & before.isna()
        blank_minor_count += int(structural.sum())
        p[col] = parsed.mask(structural, 0.0)
        if p[col].lt(0).any():
            raise ValueError(f"Negative reported amount in {col}")

    p["income_was_special_code"] = unknown_before.any(axis=1)
    p["income_job_was_special_code"] = unknown_before[JOB_COLUMNS].any(axis=1)
    rng = np.random.default_rng(seed)
    total_counts = {}
    for col in JOB_COLUMNS:
        p[f"{col}_imputed"], counts = _impute_column(p, col, rng, min_donors)
        total_counts[col] = counts

    # Person totals: sum(min_count=N) means ANY still-missing component -> NaN.
    p["job_income_pyg"] = p[JOB_COLUMNS].sum(axis=1, min_count=len(JOB_COLUMNS))
    p["total_income_pyg"] = p[INCOME_COLUMNS].sum(axis=1, min_count=len(INCOME_COLUMNS))
    p["income_nonjob_unresolved"] = p[OTHER_COLUMNS].isna().any(axis=1)
    p["income_job_unresolved"] = p[JOB_COLUMNS].isna().any(axis=1)

    job_by_hh = _strict_household_sum(p, "job_income_pyg")
    total_by_hh = _strict_household_sum(p, "total_income_pyg")
    h["income_reported_total_pyg"] = _to_money(h["household_income"]) if "household_income" in h else np.nan
    h["job_income_pyg"] = h.household_id.map(job_by_hh)
    h["total_income_pyg"] = h.household_id.map(total_by_hh)
    h["income_job_imputed"] = h.household_id.map(
        p.assign(_did=p[[f"{c}_imputed" for c in JOB_COLUMNS]].any(axis=1))
        .groupby("household_id")["_did"].any()
    ).fillna(False)
    h["income_nonjob_unresolved"] = h.household_id.map(
        p.groupby("household_id")["income_nonjob_unresolved"].any()
    ).fillna(False)
    h["income_total_unresolved"] = h["total_income_pyg"].isna()

    print("[fix_income] Job imputation:", total_counts)
    print(f"[fix_income] Children under 15: {blank_minor_count} structural blank fields -> 0")
    print(f"[fix_income] Household job income unresolved: {h.job_income_pyg.isna().sum()} / {len(h)}")
    print(f"[fix_income] Household total income unresolved: {h.total_income_pyg.isna().sum()} / {len(h)}")

    # Drop only temporary matching columns; keep income provenance flags.
    p = p.drop(columns=["_employed", "_sex", "_age_group", "_department", "_car", "_donor_weight"])
    return h, p
