"""Seville city-only HTS calibration support; metro support remains for application.

The original Seville ``od_zones`` mixes seven-character city district IDs
(e.g. 4109101) with five-character surrounding municipality IDs.  The HTS
calibration must use only origin AND destination districts inside municipality
41091; do not change ``od_zones`` because workplace choice runs across the
full modeled metropolitan area.
"""
import re

import pandas as pd


def city_district_mask(ids, municipality_id="41091"):
    """Return a boolean Series selecting 7-digit IDs inside Seville municipality."""
    code = str(municipality_id).strip()
    if not re.fullmatch(r"\d{5}", code):
        raise ValueError("gravity_calibration_municipality_id must be five digits (Seville: 41091)")
    normalized = (ids.astype("string").str.strip()
                  .str.replace(r"^(\d+)\.0+$", r"\1", regex=True))
    return normalized.str.fullmatch(re.escape(code) + r"\d{2}").fillna(False)


def select_city_od(od, municipality_id="41091"):
    """Retain internal city-district-to-city-district OD rows only."""
    if not {"origin_id", "destination_id"}.issubset(od.columns):
        raise ValueError("OD input needs origin_id and destination_id")
    return od.loc[city_district_mask(od["origin_id"], municipality_id) &
                  city_district_mask(od["destination_id"], municipality_id)].copy()


def select_calibration_inputs(distances, population, employees, municipality_id="41091"):
    """Filter BOTH ends of the calibration support; leave model data untouched."""
    dist = select_city_od(distances, municipality_id)
    pop = population.loc[city_district_mask(population["macrozone_id"], municipality_id)].copy()
    emp = employees.loc[city_district_mask(employees["macrozone_id"], municipality_id)].copy()
    if len(pop) < 2 or len(emp) < 2 or dist.empty:
        raise ValueError("Seville city-only calibration requires at least two city districts "
                         "with population, employees, and OD distances. "
                         "Check od_zones and the municipality code.")
    return dist, pop, emp


def conditional_city_probabilities(model, municipality_id="41091"):
    """Condition metro model p(j|i) on work destination j being in the city.

    Survey OD is restricted to city-to-city commutes, so comparing survey
    probabilities to unconditioned metropolitan probabilities biases fit.
    """
    city = select_city_od(model, municipality_id)
    if city.empty:
        raise ValueError("Metropolitan gravity output has no Seville city-to-city OD rows")
    city["weight"] = pd.to_numeric(city["weight"], errors="raise")
    totals = city.groupby("origin_id")["weight"].transform("sum")
    if totals.isna().any() or (totals <= 0).any():
        raise ValueError("No modeled city destination probability for an origin")
    city["weight"] /= totals
    return city
