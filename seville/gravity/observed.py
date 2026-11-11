"""Observed respondent-weighted commute OD for Seville municipality only.

All home/work endpoints, including
chained trips, are spatially joined to `seville.gravity.od_zones` and only
city-to-city district pairs are retained for calibration. A person
contributes at most one survey expansion weight, shared over their distinct
home/work location combinations. Unmapped/outside places are retained in the
person-level shares and excluded only AFTER weighting, so the retained regional
OD is not artificially inflated.

The output is weighted *respondents*, not a count of trip legs.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from seville.gravity.calibration_area import select_city_od


def configure(context):
    # Reweighted converts the point geometries from WGS84 to EPSG:25830 and
    # retains Kish/respondent trip weights. Do not take `filtered` instead.
    context.stage("seville.data.hts.entd.reweighted")
    context.stage("seville.gravity.od_zones")
    context.config("gravity_calibration_municipality_id", "41091")


def normalize_ids(values):
    """String-ify macrozone IDs without losing leading zeros in string IDs."""
    return values.astype("string").str.strip().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)


def assign_endpoints(locations, zones, trip_crs="EPSG:25830"):
    """Spatially match each individual point occurrence to ONE OD macrozone.

    Returns the original records with the column `macrozone_id` (null outside).
    Boundary points that are unambiguously in one zone are included; ambiguous
    overlapping polygons cause an error instead of arbitrary assignments.
    """
    if zones.crs is None:
        raise ValueError("Seville gravity OD zones have no CRS")
    if not {"macrozone_id", "geometry"}.issubset(zones.columns):
        raise ValueError("OD zones need 'macrozone_id' and 'geometry'")
    if zones["macrozone_id"].isna().any():
        raise ValueError("OD zone IDs cannot be null")
    polygons = zones[["macrozone_id", "geometry"]].copy()
    polygons["macrozone_id"] = normalize_ids(polygons["macrozone_id"])
    if polygons["macrozone_id"].duplicated().any():
        # The polygons must match the exact modeling universe: do not dissolve
        # silently when there are duplicate zone identifiers.
        raise ValueError("OD-zone macrozone_id values are not unique")
    points = gpd.GeoDataFrame(
        locations.drop(columns="geometry").copy(),
        geometry=gpd.GeoSeries(locations["geometry"].to_numpy(),
                               index=locations.index, crs=trip_crs),
        crs=trip_crs,
    ).to_crs(polygons.crs)
    points["_endpoint_row"] = np.arange(len(points))
    valid = points.geometry.notna() & ~points.geometry.is_empty
    mapped = gpd.sjoin(points.loc[valid], polygons, how="left", predicate="within")
    # gpd.sjoin retains each input index and can duplicate it for overlaps.
    repeated = mapped.loc[mapped["_endpoint_row"].duplicated(keep=False)]
    if not repeated.empty:
        raise ValueError("An HTS endpoint falls inside multiple OD macrozones")
    assigned = mapped.set_index("_endpoint_row")["macrozone_id"]

    # Within excludes exactly-on-the-border points. Include a boundary point
    # ONLY when intersection finds exactly one model zone.
    missing = points.loc[valid & ~points["_endpoint_row"].isin(assigned.dropna().index)]
    if not missing.empty:
        boundary = gpd.sjoin(missing, polygons, how="inner", predicate="intersects")
        counts = boundary.groupby("_endpoint_row").size()
        if (counts > 1).any():
            raise ValueError("An HTS endpoint intersects multiple zones on a boundary")
        assigned = pd.concat([assigned.dropna(), boundary.set_index("_endpoint_row")["macrozone_id"]])
    out = locations.copy()
    out["macrozone_id"] = pd.Series(np.arange(len(out)), index=out.index).map(assigned)
    return out


def get_home_work_trips(df_trips, df_zones, trip_crs="EPSG:25830"):
    """Return one person's fractional, *internal* home/work OD contributions."""
    required = {"person_id", "trip_weight", "preceding_purpose", "following_purpose",
                "origin_location", "destination_location"}
    missing = required - set(df_trips.columns)
    if missing:
        raise ValueError(f"Seville HTS trips missing fields: {sorted(missing)}")
    trips = df_trips[list(required)].copy()
    trips["trip_weight"] = pd.to_numeric(trips["trip_weight"], errors="coerce")
    trips = trips.loc[trips.person_id.notna() & np.isfinite(trips.trip_weight)
                      & (trips.trip_weight > 0)].copy()
    if trips.empty:
        raise ValueError("No positive finite HTS respondent weights")

    weights = trips.groupby("person_id").trip_weight.agg(["first", "min", "max"])
    ok = np.isclose(weights["min"], weights["max"], rtol=1e-6, atol=1e-8)
    if not ok.all():
        raise ValueError("Seville HTS trip_weight varies for respondents: "
                         f"{weights.index[~ok][:5].tolist()}")
    weights = weights["first"].rename("respondent_weight")

    starts = trips[["person_id", "preceding_purpose", "origin_location"]].rename(
        columns={"preceding_purpose": "purpose", "origin_location": "geometry"})
    ends = trips[["person_id", "following_purpose", "destination_location"]].rename(
        columns={"following_purpose": "purpose", "destination_location": "geometry"})
    locations = pd.concat([starts, ends], ignore_index=True)
    locations["purpose"] = locations.purpose.astype("string").str.lower().str.strip()
    locations = locations.loc[locations.purpose.isin(["home", "work"])].copy()
    if locations.empty:
        raise ValueError("HTS lacks home/work activity endpoints")
    locations = assign_endpoints(locations, df_zones, trip_crs)
    # Preserve unmapped endpoints as a *choice*, not a missing value dropped
    # before sharing weights. All internal/external combinations then sum to
    # exactly one respondent weight for any person with home and work activity.
    outside = "__OUTSIDE_MODEL_AREA__"
    locations["macrozone_id"] = locations.macrozone_id.fillna(outside)
    counts = (locations.groupby(["person_id", "purpose", "macrozone_id"],
                                as_index=False).size().rename(columns={"size": "n"}))
    counts["share"] = counts["n"] / counts.groupby(
        ["person_id", "purpose"])["n"].transform("sum")
    homes = counts.loc[counts.purpose.eq("home"), ["person_id", "macrozone_id", "share"]].rename(
        columns={"macrozone_id": "origin_id", "share": "home_share"})
    works = counts.loc[counts.purpose.eq("work"), ["person_id", "macrozone_id", "share"]].rename(
        columns={"macrozone_id": "destination_id", "share": "work_share"})
    pairs = homes.merge(works, on="person_id", validate="many_to_many")
    if pairs.empty:
        raise ValueError("No HTS respondents with both home and work activities")
    pairs = pairs.join(weights, on="person_id", validate="many_to_one")
    pairs["weight"] = pairs.respondent_weight * pairs.home_share * pairs.work_share
    check = pairs.groupby("person_id").weight.sum()
    if not np.allclose(check.to_numpy(), weights.loc[check.index].to_numpy(), rtol=1e-9, atol=1e-7):
        raise AssertionError("HTS respondent OD weights are not conserved")
    internal = pairs.loc[pairs.origin_id.ne(outside) & pairs.destination_id.ne(outside)].copy()
    if internal.empty:
        raise ValueError("No internal HTS home/work OD pairs for Seville model zones")
    print(f"Seville observed OD: {len(check)} home/work respondents, "
          f"{internal.person_id.nunique()} with internal pairs; "
          f"retained {internal.weight.sum():.1f} / {pairs.weight.sum():.1f} "
          f"({internal.weight.sum()/pairs.weight.sum():.1%}) weighted respondents")
    return internal[["person_id", "origin_id", "destination_id", "weight"]]


def execute(context):
    _, _, trips = context.stage("seville.data.hts.entd.reweighted")
    zones, _, _ = context.stage("seville.gravity.od_zones")
    pairs = get_home_work_trips(trips, zones)
    # Preserve respondent-weight splitting first, THEN exclude commutes with
    # either endpoint outside Seville municipality (city's 7-digit districts).
    code = context.config("gravity_calibration_municipality_id")
    pairs = select_city_od(pairs, code)
    if pairs.empty:
        raise ValueError("No HTS city-to-city commuters in municipality " + str(code))
    print(f"Seville HTS CALIBRATION ONLY: {len(pairs)} city-to-city respondent pairs "
          f"in municipality {code} ({pairs.weight.sum():.1f} weighted).")
    result = (pairs.groupby(["origin_id", "destination_id"], as_index=False).weight.sum()
              .sort_values(["origin_id", "destination_id"]).reset_index(drop=True))
    result["trips"] = result.weight  # Legacy alias; not trip-leg frequency.
    return result[["origin_id", "destination_id", "weight", "trips"]]
