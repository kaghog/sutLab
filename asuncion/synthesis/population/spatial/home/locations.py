"""This stage Assigns home addresses to households, balancing Asunción boroughs by resident population.

Only Asuncion district is handled this way, because it is only district we
have borough-level data.


Asunción: allocate INTACT HOUSEHOLDS to boroughs to closely match census
resident shares, using household_size; then draw an address inside the assigned
borough using that address's weight. This preserves the synthetic district's
total residents; it does not force the total to the absolute census total.

The expensive grouping is performed once before launching Synpp parallel jobs.
Each worker receives only its district's homes and address candidates.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

ASUNCION_DISTRICT_ID = "0000"
ASUNCION_IRIS_ID = ASUNCION_DISTRICT_ID + "0000"


def configure(context):
    context.stage("synthesis.population.spatial.home.zones")
    context.stage("synthesis.locations.home.locations")
    context.stage("asuncion.data.census.population")
    context.config("home_location_source", "addresses")
    context.config("random_seed")


def _draw_addresses(rng, weights, n):
    """Return candidate indices sampled with replacement, weighted by address."""
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) == 0 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Missing, nonfinite, or negative home-location weights")
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        raise ValueError("Home-location weights must sum to a positive number")
    draws = rng.random_sample(n) * cumulative[-1]
    return np.searchsorted(cumulative, draws, side="right").clip(max=len(weights) - 1)


def _allocate_by_residents(rng, household_sizes, probabilities):
    """Greedily balance *residents*, keeping households intact.

    Randomize household order and borough tie-breaking. Each assignment selects
    the borough furthest below its target share of residents assigned so far.
    Returns borough indices and realised resident counts, respectively.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    household_sizes = np.asarray(household_sizes, dtype=np.int64)
    assignments = np.empty(len(household_sizes), dtype=np.int32)
    assigned_residents = np.zeros(len(probabilities), dtype=np.int64)
    cumulative_residents = 0

    borough_order = rng.permutation(len(probabilities))
    reordered_p = probabilities[borough_order]

    for home_position in rng.permutation(len(household_sizes)):
        size = household_sizes[home_position]
        cumulative_residents += size
        deficits = cumulative_residents * reordered_p - assigned_residents[borough_order]
        borough_index = borough_order[np.argmax(deficits)]
        assignments[home_position] = borough_index
        assigned_residents[borough_index] += size

    return assignments, assigned_residents


def _sample_district(context, task):
    """Worker called by context.parallel; update progress inside its active block."""
    homes_by_iris = context.data("homes_by_iris")
    locations_by_iris = context.data("locations_by_iris")
    asuncion_reference = context.data("asuncion_reference")

    iris_id, seed = task
    homes = homes_by_iris[iris_id]
    locations = locations_by_iris.get(iris_id)
    if locations is None or locations.empty:
        raise ValueError(f"No candidate addresses for iris_id={iris_id}")

    rng = np.random.RandomState(seed)
    n = len(homes)
    location_weights = locations["weight"].to_numpy(dtype=np.float64)
    chosen = np.empty(n, dtype=np.int64)

    if iris_id == ASUNCION_IRIS_ID:
        # Census weights are resident counts, not household counts.
        reference = asuncion_reference
        if reference.empty:
            raise ValueError("Asunción has no positive borough census population")
        borough_ids = reference["borough_id"].to_numpy(dtype=str)
        weights = reference["weight"].to_numpy(dtype=np.float64)
        borough_probabilities = weights / weights.sum()
        sizes = homes["household_size"].to_numpy(dtype=np.int64)
        if locations["borough_id"].isna().any():
            raise ValueError("Asunción candidates have null borough IDs")
        candidates_borough = locations["borough_id"].to_numpy(dtype=str)

        missing_boroughs = set(borough_ids) - set(candidates_borough)
        if missing_boroughs:
            raise ValueError(
                f"Asunción census boroughs without home candidates: {sorted(missing_boroughs)}"
            )

        allocations, actual_residents = _allocate_by_residents(
            rng, sizes, borough_probabilities
        )

        # Draw actual addresses within each borough (weighted; replacement allowed).
        for borough_index, borough_id in enumerate(borough_ids):
            household_positions = np.flatnonzero(allocations == borough_index)
            if len(household_positions) == 0:
                continue
            candidate_positions = np.flatnonzero(candidates_borough == borough_id)
            candidate_draws = _draw_addresses(
                rng, location_weights[candidate_positions], len(household_positions)
            )
            chosen[household_positions] = candidate_positions[candidate_draws]

        expected_residents = sizes.sum() * borough_probabilities
        max_gap = np.abs(actual_residents - expected_residents).max()
        max_gap_pp = 100.0 * max_gap / sizes.sum()
        print(
            f"[Asunción] {sizes.sum():,} residents, {n:,} households, "
            f"{len(borough_ids)} boroughs; max absolute target gap "
            f"{max_gap:.2f} residents ({max_gap_pp:.4f} percentage points)"
        )
    else:
        # Central and other districts: preserve ordinary weighted address choice.
        chosen[:] = _draw_addresses(rng, location_weights, n)

    result = homes[["_row_id", "household_id", "commune_id"]].copy()
    result["home_location_id"] = locations["home_location_id"].to_numpy()[chosen]
    result["geometry"] = locations.geometry.array.take(chosen)

    context.progress.update()  # Valid because execute() has entered context.progress(...)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=locations.crs)


def execute(context):
    homes = context.stage("synthesis.population.spatial.home.zones").copy()
    locations = context.stage("synthesis.locations.home.locations").copy()
    census = context.stage("asuncion.data.census.population").copy()

    required = [
        (homes, {"household_id", "commune_id", "iris_id", "household_size"}, "Homes"),
        (locations, {"iris_id", "borough_id", "home_location_id", "weight", "geometry"}, "Locations"),
        (census, {"district_id", "borough_id", "weight"}, "Census"),
    ]
    for frame, fields, name in required:
        missing = fields.difference(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing: {sorted(missing)}")

    homes["_row_id"] = np.arange(len(homes))

    # Only Asunción requires borough-level census reference.
    asuncion_reference = (
        census.loc[census["district_id"] == ASUNCION_DISTRICT_ID]
        .groupby("borough_id", as_index=False)["weight"].sum()
    )
    asuncion_reference = (
        asuncion_reference.loc[asuncion_reference["weight"] > 0]
        .sort_values("borough_id")
        .reset_index(drop=True)
    )

    # Pre-split once so parallel tasks do not repeatedly scan full tables.
    homes_by_iris = {
        str(iris): frame[["_row_id", "household_id", "commune_id", "household_size"]]
        .reset_index(drop=True)
        for iris, frame in homes.groupby("iris_id", sort=True)
    }
    locations_by_iris = {
        str(iris): frame[["home_location_id", "borough_id", "weight", "geometry"]]
        .reset_index(drop=True)
        for iris, frame in locations.groupby("iris_id", sort=False)
    }

    rng = np.random.RandomState(context.config("random_seed"))
    district_ids = sorted(homes_by_iris)
    seeds = rng.randint(np.iinfo(np.int32).max, size=len(district_ids))
    tasks = list(zip(district_ids, seeds))
    chunks = []


    # Parallel matching
    with context.progress(label="Sampling home locations ...", total=len(tasks)):
        with context.parallel(data=dict(
            homes_by_iris=homes_by_iris,
            locations_by_iris=locations_by_iris,
            asuncion_reference=asuncion_reference,
        )) as parallel:
            for sampled_district in parallel.imap_unordered(_sample_district, tasks):
                chunks.append(sampled_district)

    if not chunks:
        return gpd.GeoDataFrame(
            columns=["household_id", "commune_id", "home_location_id", "geometry"],
            geometry="geometry", crs=locations.crs,
        )

    result = pd.concat(chunks, ignore_index=True).sort_values("_row_id", kind="stable")
    return gpd.GeoDataFrame(
        result[["household_id", "commune_id", "home_location_id", "geometry"]]
        .reset_index(drop=True), geometry="geometry", crs=locations.crs,
    )
