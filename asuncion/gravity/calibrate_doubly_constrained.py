"""Calibrate Asuncion gravity impedance against weighted HTS OD flows.

Doubly constrained model.

Set context.config("gravity_model_type") to:
  * "power_law"   : f(d) = d ** gamma
  * "exponential" : f(d) = exp(slope * d)

Both models include a separately calibrated same-zone multiplier q and use
identical zero-intrazonal-distance handling. For each candidate parameter set,
predicted OD flows are IPF-balanced to the observed HTS row/column margins.
Optimization of the profiled Poisson likelihood estimates distance deterrence
and q while accounting for origin/destination fixed effects.

Calibration only uses HTS OD pairs for which the distance matrix supplies a
valid distance. External-zone HTS trips are dropped, as are HTS trips for
incomplete internal OD links. Unobserved survey flows are zero ONLY on OD
pairs with known distances; missing distance links are excluded, not assigned
a zero-kilometre distance. This mimics the pairwise inner join of the original
calibration while supporting IPF/PPML on an incomplete OD grid.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

DEFAULT_GRAVITY_MODEL_TYPE = "power_law"
VALID_GRAVITY_MODEL_TYPES = ("power_law", "exponential")

# Apply to diagonal OD pairs with distance == 0 only, identically in both stages.
INTRAZONAL_DISTANCE_KM = 1.0

# Starting values and bounds for the distance term, not fixed model parameters.
INITIAL_DISTANCE_PARAMETER = {
    "power_law": -0.6,       # gamma (dimensionless)
    "exponential": -0.1,     # slope (per km)
}
DISTANCE_PARAMETER_BOUNDS = {
    "power_law": (-6.0, 0.0),
    "exponential": (-1.0, 0.0),
}
INITIAL_LOG_INTRA = 0.0
LOG_INTRA_BOUNDS = (np.log(0.05), np.log(20.0))

IPF_TOLERANCE = 1.0e-9
IPF_MAX_ITERATIONS = 10000


def configure(context):
    context.stage("asuncion.gravity.observed")
    context.stage("asuncion.gravity.distance_matrix")
    context.config("gravity_model_type", DEFAULT_GRAVITY_MODEL_TYPE)


def _validate_model_type(model_type):
    if model_type not in VALID_GRAVITY_MODEL_TYPES:
        raise ValueError(
            f"gravity_model_type must be one of {VALID_GRAVITY_MODEL_TYPES}; "
            f"got {model_type!r}"
        )
    return model_type


def _balanced_flow(friction, origin_totals, dest_totals):
    """Return OD flows matching both strictly positive marginal vectors."""
    flow = friction.astype(float, copy=True)
    for _ in range(IPF_MAX_ITERATIONS):
        row = flow.sum(axis=1)
        if (row <= 0).any() or not np.isfinite(row).all():
            raise ValueError("Invalid origin total during IPF")
        flow *= (origin_totals / row)[:, None]

        col = flow.sum(axis=0)
        if (col <= 0).any() or not np.isfinite(col).all():
            raise ValueError("Invalid destination total during IPF")
        flow *= (dest_totals / col)[None, :]

        row_error = np.max(
            np.abs(flow.sum(axis=1) - origin_totals) / origin_totals
        )
        col_error = np.max(
            np.abs(flow.sum(axis=0) - dest_totals) / dest_totals
        )
        if max(row_error, col_error) < IPF_TOLERANCE:
            return flow
    raise RuntimeError(
        f"HTS IPF did not converge in {IPF_MAX_ITERATIONS} iterations. "
        "Inspect distance values and parameter bounds."
    )


def _prepare_hts(df_obs, df_dist):
    """Prepare the HTS OD grid, treating missing distances as unavailable cells."""
    required_obs = {"origin_id", "destination_id", "weight"}
    required_dist = {"origin_id", "destination_id", "distance_km"}
    if not required_obs.issubset(df_obs.columns):
        raise ValueError(f"HTS missing columns: {required_obs - set(df_obs.columns)}")
    if not required_dist.issubset(df_dist.columns):
        raise ValueError(f"Distances missing columns: {required_dist - set(df_dist.columns)}")

    obs = df_obs[["origin_id", "destination_id", "weight"]].copy()
    if obs.isna().any().any():
        raise ValueError("Null IDs / weights in observed HTS")
    obs["weight"] = pd.to_numeric(obs["weight"], errors="raise")
    if not np.isfinite(obs["weight"]).all() or (obs["weight"] < 0).any():
        raise ValueError("HTS OD weights must be finite and nonnegative")

    distances = df_dist[["origin_id", "destination_id", "distance_km"]].copy()
    if distances.duplicated(["origin_id", "destination_id"]).any():
        raise ValueError("Duplicate pairs in distance matrix")
    distances["distance_km"] = pd.to_numeric(distances["distance_km"], errors="raise")
    is_diagonal = distances["origin_id"] == distances["destination_id"]
    dvals = distances["distance_km"].to_numpy(dtype=float)
    invalid_nonmissing = (
        (np.isfinite(dvals) & (dvals < 0))
        | (np.isfinite(dvals) & (dvals == 0) & ~is_diagonal.to_numpy())
        | np.isinf(dvals)
    )
    if invalid_nonmissing.any():
        raise ValueError(
            "Invalid distances (negative, infinite, or off-diagonal zero); "
            "correct these in distance_matrix.py"
        )

    # Filter by ACTUAL OD pair availability. Checking each ID separately is
    # insufficient when the distance table doesn't contain every combination.
    known_pairs = distances.loc[
        distances["distance_km"].notna(),
        ["origin_id", "destination_id"],
    ]
    matched = obs.merge(
        known_pairs.assign(_known_distance=True),
        on=["origin_id", "destination_id"],
        how="left",
        validate="many_to_one",
    )
    keep = matched["_known_distance"].eq(True)
    excluded = matched.loc[~keep]
    if not excluded.empty:
        # Distinguish external-zone trips from incomplete internal OD links.
        origin_ids = set(known_pairs["origin_id"])
        destination_ids = set(known_pairs["destination_id"])
        external = (
            ~excluded["origin_id"].isin(origin_ids)
            | ~excluded["destination_id"].isin(destination_ids)
        )
        lost_weight = float(excluded["weight"].sum())
        original_weight = float(obs["weight"].sum())
        print(
            f"HTS: excluded {len(excluded):,} trips with no modeled OD "
            f"distance ({lost_weight:,.1f} weighted; "
            f"{lost_weight / original_weight:.2%} of HTS weighted trips)."
            if original_weight > 0 else
            f"HTS: excluded {len(excluded):,} trips with no modeled OD distance."
        )
        print(
            f"  Outside modeled zones: {int(external.sum()):,} rows; "
            f"missing internal OD distances: {int((~external).sum()):,} rows."
        )
        print(
            "  Excluded OD examples:",
            list(excluded[["origin_id", "destination_id"]]
                 .itertuples(index=False, name=None))[:10],
        )

    obs = matched.loc[
        keep, ["origin_id", "destination_id", "weight"]
    ].copy()
    obs = obs.groupby(["origin_id", "destination_id"], as_index=False)["weight"].sum()
    obs = obs.loc[obs["weight"] > 0].copy()
    if obs.empty:
        raise ValueError(
            "No positive HTS work trips have a matching modeled OD distance. "
            "Check macrozone IDs and distance-matrix coverage."
        )

    origins = pd.Index(sorted(obs["origin_id"].unique()), name="origin_id")
    destinations = pd.Index(sorted(obs["destination_id"].unique()), name="destination_id")
    if len(origins) < 2 or len(destinations) < 2:
        raise ValueError("At least two observed origins and destinations are required")
    pairs = pd.MultiIndex.from_product(
        [origins, destinations], names=["origin_id", "destination_id"]
    )

    d_series = distances.set_index(
        ["origin_id", "destination_id"]
    )["distance_km"].reindex(pairs)
    raw_distances = d_series.to_numpy(dtype=float).reshape(
        len(origins), len(destinations)
    )
    supported = np.isfinite(raw_distances)
    same_zone = origins.to_numpy()[:, None] == destinations.to_numpy()[None, :]

    # Cells with no distance are NOT zeros. They are outside the set of OD
    # pairs on which calibration is conditioned, so their friction is masked.
    # 1.0 km is only a computational placeholder in the feature array for
    # unsupported cells: their friction is exactly zero (see execute()).
    d = np.where(supported, raw_distances, 1.0)
    d[supported & same_zone & (d == 0)] = INTRAZONAL_DISTANCE_KM
    if (d[supported] <= 0).any():
        raise ValueError("Valid OD distances must be positive after diagonal replacement")

    missing = int((~supported).sum())
    if missing:
        print(
            f"HTS calibration: {missing:,} origin/destination grid cells "
            "lack distances; excluding these cells from calibration."
        )

    observed = (
        obs.set_index(["origin_id", "destination_id"])["weight"]
        .reindex(pairs, fill_value=0.0)
        .to_numpy(dtype=float)
        .reshape(d.shape)
    )
    if (observed[~supported] > 0).any():
        raise AssertionError("Observed positive flow without matching OD distance")
    O = observed.sum(axis=1)
    D = observed.sum(axis=0)
    return pairs, observed, d, same_zone.astype(float), supported, O, D


def execute(context):
    model_type = _validate_model_type(context.config("gravity_model_type"))
    df_obs = context.stage("asuncion.gravity.observed").copy()
    df_dist = context.stage("asuncion.gravity.distance_matrix").copy()

    # _prepare_hts filters HTS trips on exact OD-pair matches with the
    # distance matrix. Merely checking valid origin/destination ID sets can
    # incorrectly accept OD combinations that have no distance available.
    pairs, observed, distance, same_zone, supported, O, D = _prepare_hts(
        df_obs, df_dist
    )

    # The only difference between the two models is the distance feature.
    # power_law: log f = gamma * log(d); exponential: log f = slope * d.
    distance_feature = (
        np.log(distance) if model_type == "power_law" else distance
    )
    parameter_name = "gamma" if model_type == "power_law" else "slope"
    total_observed = float(observed.sum())

    # Check identification on OD pairs that have distances (the supported
    # graph). Two-way demeaning over the full rectangle would be incorrect
    # when some OD distances are missing.
    row_indices, col_indices = np.nonzero(supported)
    n_links = len(row_indices)
    X = coo_matrix(
        (
            np.ones(2 * n_links),
            (
                np.tile(np.arange(n_links), 2),
                np.r_[row_indices, len(O) + col_indices],
            ),
        ),
        shape=(n_links, len(O) + len(D)),
    ).tocsr()

    def residualize(feature):
        values = feature[supported]
        fixed_effects = lsqr(X, values, atol=1e-11, btol=1e-11)[0]
        return values - X @ fixed_effects

    residual_distance = residualize(distance_feature)
    residual_diagonal = residualize(same_zone)
    if np.linalg.norm(residual_distance) < 1e-10 * max(
        1.0, np.linalg.norm(distance_feature[supported])
    ):
        raise ValueError("Cannot identify distance deterrence from this HTS OD grid")
    feature_matrix = np.column_stack((residual_distance, residual_diagonal))
    estimate_intra = np.linalg.matrix_rank(feature_matrix) == 2
    if not estimate_intra:
        warnings.warn(
            "HTS cannot separately identify intrazonal preference; "
            "fixing intra_multiplier=1.0.",
            RuntimeWarning,
        )

    def log_friction(theta):
        distance_param = float(theta[0])
        log_intra = float(theta[1]) if estimate_intra else 0.0
        return distance_param * distance_feature + log_intra * same_zone

    def objective(theta):
        log_f = log_friction(theta)
        # IPF is invariant under global positive rescaling of friction.
        friction = np.exp(log_f - np.max(log_f[supported]))
        friction[~supported] = 0.0  # Exclude missing OD pairs, never zero-km friction.
        fitted = _balanced_flow(friction, O, D)

        fitted_p = np.maximum(fitted / total_observed, np.finfo(float).tiny)
        nll = -np.sum((observed / total_observed) * np.log(fitted_p))

        # Gradient of profiled PPML by the envelope theorem.
        delta = (fitted - observed) / total_observed
        gradient = [np.sum(delta * distance_feature)]
        if estimate_intra:
            gradient.append(np.sum(delta * same_zone))
        return nll, np.asarray(gradient, dtype=float)

    initial = [INITIAL_DISTANCE_PARAMETER[model_type]]
    bounds = [DISTANCE_PARAMETER_BOUNDS[model_type]]
    if estimate_intra:
        initial.append(INITIAL_LOG_INTRA)
        bounds.append(LOG_INTRA_BOUNDS)

    result = minimize(
        objective,
        x0=np.asarray(initial, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"HTS gravity calibration failed: {result.message}")

    distance_parameter = float(result.x[0])
    intra_multiplier = float(np.exp(result.x[1])) if estimate_intra else 1.0
    log_f = log_friction(result.x)
    friction = np.exp(log_f - np.max(log_f[supported]))
    friction[~supported] = 0.0
    predicted = _balanced_flow(friction, O, D)
    obs_supported = observed[supported]
    pred_supported = predicted[supported]
    correlation = (
        float(np.corrcoef(obs_supported, pred_supported)[0, 1])
        if obs_supported.size > 1 and obs_supported.std() > 0
        and pred_supported.std() > 0
        else float("nan")
    )

    for parameter, (lower, upper) in zip(result.x, bounds):
        if min(abs(parameter - lower), abs(parameter - upper)) < 1e-4:
            warnings.warn(
                "Gravity calibration reached a parameter bound; inspect HTS "
                "coverage, intrazonal distances and model fit.",
                RuntimeWarning,
            )
            break

    diagnostics = pd.DataFrame({
        "origin_id": pairs.get_level_values("origin_id"),
        "destination_id": pairs.get_level_values("destination_id"),
        "distance_km": distance.ravel(),
        "observed_flow": observed.ravel(),
        "predicted_flow": predicted.ravel(),
    }).loc[supported.ravel()].reset_index(drop=True)

    print(f"Gravity model:                 {model_type}")
    print(f"HTS observed commuter total:   {total_observed:,.1f}")
    print(f"Distance parameter {parameter_name}: {distance_parameter:.6f}")
    print(f"Intrazonal multiplier:         {intra_multiplier:.6f}")
    print(f"HTS OD correlation:            {correlation:.4f}")
    print(f"Estimated parameters:          {len(result.x)}")
    print(f"Optimization evaluations:      {result.nfev}")

    return {
        "gravity_model_type": model_type,
        "distance_parameter": distance_parameter,
        parameter_name: distance_parameter,
        "intra_multiplier": intra_multiplier,
        "intra_km": INTRAZONAL_DISTANCE_KM,
        "correlation": correlation,
        "data": diagnostics,
    }
