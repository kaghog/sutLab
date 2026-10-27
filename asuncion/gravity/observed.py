"""Build a weighted home-to-work OD matrix from HTS activity locations.

Each respondent contributes at most one respondent expansion weight to the OD
matrix. The weight is distributed across the Cartesian product of their
observed home and work districts, in proportion to how often those districts
appear as home/work trip endpoints.

The OD matrix is *not* a trip-frequency matrix. It estimates a fractional
residential/workplace distribution for respondents with observed home and work
locations. Pairs outside the modeled geography are removed after the person
shares are calculated, so outside-work shares are not reassigned to inside jobs.
"""

import numpy as np
import pandas as pd


def configure(context):
    context.stage("asuncion.gravity.od_zones")
    context.stage("asuncion.data.hts.entd.filtered")


def _as_zone_ids(values):
    """Convert HTS and OD-zone IDs to comparable, non-null strings."""
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def get_home_work_trips(df_trips, level, df_zones):
    """Return fractional respondent-weighted OD pairs (not yet aggregated).

    Each *occurrence* of home or work at either endpoint of an HTS trip counts
    toward that person's district shares. Trip endpoint occurrences should not
    be interpreted as unique activity episodes when successive trips share a
    stop; see discussion in the accompanying notes.
    """
    origin_col = f"origin_{level}"
    destination_col = f"destination_{level}"
    required = {
        "person_id", "trip_weight", "preceding_purpose", "following_purpose",
        origin_col, destination_col,
    }
    missing = required.difference(df_trips.columns)
    if missing:
        raise ValueError(f"HTS trips are missing columns: {sorted(missing)}")
    if "macrozone_id" not in df_zones.columns:
        raise ValueError("OD zones must contain macrozone_id")

    trips = df_trips.loc[:, list(required)].copy()
    trips["trip_weight"] = pd.to_numeric(trips["trip_weight"], errors="coerce")
    trips = trips.loc[
        trips["person_id"].notna()
        & np.isfinite(trips["trip_weight"])
        & (trips["trip_weight"] > 0)
    ].copy()
    if trips.empty:
        raise ValueError("No HTS trips with positive, finite respondent weights")

    # This stage assumes the trip weight is the same respondent expansion weight
    # on all trips by a person. Detect violations rather than multiplying a
    # person's contribution by an arbitrary trip-specific weight.
    person_weights = trips.groupby("person_id")["trip_weight"].agg(
        respondent_weight="first", min_weight="min", max_weight="max"
    )
    consistent = np.isclose(
        person_weights["min_weight"].to_numpy(),
        person_weights["max_weight"].to_numpy(),
        rtol=1e-6, atol=1e-8,
    )
    if not consistent.all():
        bad = person_weights.index[~consistent][:5].tolist()
        raise ValueError(
            "trip_weight is not constant within some respondents "
            f"(examples: {bad}). Join a person-level survey weight first."
        )
    person_weights = person_weights[["respondent_weight"]].reset_index()

    # A location can be observed at the start OR end of a trip. Unlike direct
    # home->work filtering, this also captures chained home->other->work trips.
    origins = trips.loc[:, ["person_id", origin_col, "preceding_purpose"]].rename(
        columns={origin_col: "district_id", "preceding_purpose": "purpose"}
    )
    destinations = trips.loc[:, ["person_id", destination_col, "following_purpose"]].rename(
        columns={destination_col: "district_id", "following_purpose": "purpose"}
    )
    locations = pd.concat([origins, destinations], ignore_index=True)
    locations["purpose"] = locations["purpose"].astype("string").str.lower().str.strip()
    locations["district_id"] = _as_zone_ids(locations["district_id"])
    locations = locations.loc[
        locations["purpose"].isin(["home", "work"])
        & locations["district_id"].notna()
        & locations["district_id"].ne("")
    ].copy()
    if locations.empty:
        raise ValueError("No home/work district locations found in HTS endpoints")

    counts = (
        locations.groupby(["person_id", "purpose", "district_id"], as_index=False)
        .size()
        .rename(columns={"size": "appearances"})
    )
    counts["share"] = counts["appearances"] / counts.groupby(
        ["person_id", "purpose"]
    )["appearances"].transform("sum")

    homes = counts.loc[counts["purpose"].eq("home"), [
        "person_id", "district_id", "share"
    ]].rename(columns={"district_id": "origin_id", "share": "home_share"})
    works = counts.loc[counts["purpose"].eq("work"), [
        "person_id", "district_id", "share"
    ]].rename(columns={"district_id": "destination_id", "share": "work_share"})

    # The only Cartesian product is the set of DISTINCT home and work
    # districts for the SAME person, not raw trip records.
    pairs = homes.merge(works, on="person_id", how="inner", validate="many_to_many")
    if pairs.empty:
        raise ValueError("No HTS respondents have both home and work locations")
    pairs = pairs.merge(person_weights, on="person_id", how="left", validate="many_to_one")
    pairs["weight"] = (
        pairs["respondent_weight"] * pairs["home_share"] * pairs["work_share"]
    )

    # Before geography filtering, every respondent with both types of location
    # must contribute exactly one respondent weight.
    check = pairs.groupby("person_id")["weight"].sum()
    expected = person_weights.set_index("person_id").loc[check.index, "respondent_weight"]
    if not np.allclose(check.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-7):
        raise AssertionError("Respondent weights are not conserved")

    valid_zones = set(_as_zone_ids(df_zones["macrozone_id"]).dropna())
    within = pairs["origin_id"].isin(valid_zones) & pairs["destination_id"].isin(valid_zones)
    retained = pairs.loc[within].copy()
    if retained.empty:
        raise ValueError("No respondent-weighted OD pairs within the model zones")

    # Report excluded weight, including partial exclusion of a multi-location
    # respondent. Do NOT renormalize: this would redirect outside work choices
    # to workplaces inside the modeled region.
    before = float(pairs["weight"].sum())
    after = float(retained["weight"].sum())
    print(
        "HTS fractional home/work OD: "
        f"{len(check)} respondents with home and work; "
        f"{retained['person_id'].nunique()} with an internal OD; "
        f"weighted mass retained {after:,.2f}/{before:,.2f} "
        f"({after / before:.1%})."
    )
    return retained[["person_id", "origin_id", "destination_id", "weight"]]


def execute(context):
    df_trips = context.stage("asuncion.data.hts.entd.filtered")[2].copy()
    df_zones, _, _, _ = context.stage("asuncion.gravity.od_zones")

    pairs = get_home_work_trips(df_trips, "district_id", df_zones)
    od_matrix = (
        pairs.groupby(["origin_id", "destination_id"], as_index=False)["weight"]
        .sum()
        .sort_values(["origin_id", "destination_id"], kind="stable")
        .reset_index(drop=True)
    )
    # Compatibility with the existing 'asuncion.gravity.observed' interface.
    # 'trips' is a legacy alias: these values represent *weighted workers*,
    # not a count of commute legs.
    od_matrix["trips"] = od_matrix["weight"]
    return od_matrix[["origin_id", "destination_id", "weight", "trips"]]
