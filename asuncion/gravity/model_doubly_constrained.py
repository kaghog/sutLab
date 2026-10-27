"""Apply the calibrated Asuncion doubly constrained work gravity model.

The gravity_model_type config:
  "power_law"   : distance friction = d ** gamma
  "exponential" : distance friction = exp(slope * d)

Balancing targets use origin workers (preferred; population is fallback) and
scaled destination employees. This assumes a closed study region; inspect the
warning when total workers and jobs differ.
"""

import warnings

import numpy as np
import pandas as pd

DEFAULT_GRAVITY_MODEL_TYPE = "power_law"
VALID_GRAVITY_MODEL_TYPES = ("power_law", "exponential")
IPF_TOLERANCE = 1.0e-9
IPF_MAX_ITERATIONS = 10000


def configure(context):
    context.stage("asuncion.gravity.distance_matrix")
    context.stage("asuncion.gravity.od_zones")
    context.stage("asuncion.gravity.calibrate_doubly_constrained")
    context.config("gravity_model_type", DEFAULT_GRAVITY_MODEL_TYPE)


def _validate_model_type(model_type):
    if model_type not in VALID_GRAVITY_MODEL_TYPES:
        raise ValueError(
            f"gravity_model_type must be one of {VALID_GRAVITY_MODEL_TYPES}; "
            f"got {model_type!r}"
        )
    return model_type


def _balanced_flow(friction, origin_totals, destination_totals):
    flow = friction.astype(float, copy=True)
    for iteration in range(IPF_MAX_ITERATIONS):
        row = flow.sum(axis=1)
        if (row <= 0).any() or not np.isfinite(row).all():
            raise ValueError("Invalid row totals in gravity IPF")
        flow *= (origin_totals / row)[:, None]

        col = flow.sum(axis=0)
        if (col <= 0).any() or not np.isfinite(col).all():
            raise ValueError("Invalid column totals in gravity IPF")
        flow *= (destination_totals / col)[None, :]

        row_error = np.max(
            np.abs(flow.sum(axis=1) - origin_totals) / origin_totals
        )
        col_error = np.max(
            np.abs(flow.sum(axis=0) - destination_totals) / destination_totals
        )
        if max(row_error, col_error) < IPF_TOLERANCE:
            return flow, iteration + 1

    raise RuntimeError(
        f"Application IPF did not converge in {IPF_MAX_ITERATIONS} iterations"
    )


def execute(context):
    model_type = _validate_model_type(context.config("gravity_model_type"))
    df_distances = context.stage("asuncion.gravity.distance_matrix").copy()
    _, df_population, df_employees, _ = context.stage("asuncion.gravity.od_zones")
    calibration = context.stage("asuncion.gravity.calibrate_doubly_constrained")

    # Fail explicitly on stale / mismatched calibrated parameters.
    if calibration.get("gravity_model_type") != model_type:
        raise ValueError(
            f"Calibration model type {calibration.get('gravity_model_type')!r} "
            f"does not match configured model type {model_type!r}. "
            "Re-run asuncion.gravity.calibrate with the same configuration."
        )

    parameter_name = "gamma" if model_type == "power_law" else "slope"
    distance_parameter = float(calibration[parameter_name])
    intra_multiplier = float(calibration["intra_multiplier"])
    intra_km = float(calibration["intra_km"])
    if not all(np.isfinite(v) for v in (distance_parameter, intra_multiplier, intra_km)):
        raise ValueError("Non-finite gravity parameters")
    if intra_multiplier <= 0 or intra_km <= 0:
        raise ValueError("Intrazonal parameters must be positive")

    if "macrozone_id" not in df_population or "macrozone_id" not in df_employees:
        raise ValueError("od_zones requires macrozone_id in population and employees")

    # Prefer the number of employed residents actually needing workplaces.
    if "workers" in df_population.columns:
        production_column = "workers"
    elif "employed_population" in df_population.columns:
        production_column = "employed_population"
    else:
        production_column = "population"
        warnings.warn(
            "Using total population as gravity origin production. "
            "For correct destination job counts, supply employed residents "
            "per origin as a 'workers' column in od_zones.",
            RuntimeWarning,
        )

    productions = (
        df_population[["macrozone_id", production_column]]
        .rename(columns={"macrozone_id": "origin_id", production_column: "workers"})
        .copy()
    )
    employments = (
        df_employees[["macrozone_id", "employees"]]
        .rename(columns={"macrozone_id": "destination_id"})
        .copy()
    )
    if productions.isna().any().any() or employments.isna().any().any():
        raise ValueError("Null zone IDs or worker/job values")

    productions["workers"] = pd.to_numeric(productions["workers"], errors="raise")
    employments["employees"] = pd.to_numeric(employments["employees"], errors="raise")
    O_series = productions.groupby("origin_id")["workers"].sum().astype(float)
    D_series = employments.groupby("destination_id")["employees"].sum().astype(float)
    if (
        not np.isfinite(O_series.to_numpy()).all()
        or not np.isfinite(D_series.to_numpy()).all()
        or (O_series < 0).any()
        or (D_series < 0).any()
    ):
        raise ValueError("Invalid negative / non-finite workers or employees")

    O_series = O_series[O_series > 0].sort_index()
    D_series = D_series[D_series > 0].sort_index()
    if O_series.empty or D_series.empty:
        raise ValueError("No active worker origins or employment destinations")

    original_job_total = float(D_series.sum())
    worker_total = float(O_series.sum())
    scale = worker_total / original_job_total
    if abs(scale - 1.0) > 0.01:
        warnings.warn(
            f"Total origin production ({worker_total:,.0f}) differs from "
            f"employment ({original_job_total:,.0f}). "
            f"Employment targets are scaled by {scale:.5f} for IPF. "
            "Check commuters entering/leaving the study area and whether "
            "'population' rather than 'workers' was supplied.",
            RuntimeWarning,
        )
    D_series = D_series * scale

    origins, destinations = O_series.index, D_series.index
    pairs = pd.MultiIndex.from_product(
        [origins, destinations], names=["origin_id", "destination_id"]
    )
    required = {"origin_id", "destination_id", "distance_km"}
    if not required.issubset(df_distances.columns):
        raise ValueError(f"Missing distance columns: {required - set(df_distances.columns)}")
    df_distances = df_distances[["origin_id", "destination_id", "distance_km"]]
    if df_distances.duplicated(["origin_id", "destination_id"]).any():
        raise ValueError("Duplicate OD distances")
    d_series = (
        df_distances.set_index(["origin_id", "destination_id"])["distance_km"]
        .reindex(pairs)
    )
    if d_series.isna().any():
        raise ValueError(
            f"Missing distances for {int(d_series.isna().sum())} active OD pairs. "
            "Complete the distance matrix rather than treating them as zero."
        )
    distances = pd.to_numeric(d_series, errors="raise").to_numpy(dtype=float).reshape(
        len(origins), len(destinations)
    )
    same_zone = origins.to_numpy()[:, None] == destinations.to_numpy()[None, :]
    distances[(distances == 0) & same_zone] = intra_km
    if not np.isfinite(distances).all() or (distances <= 0).any():
        raise ValueError("Distances must be positive (only diagonal zero can be replaced)")

    distance_feature = (
        np.log(distances) if model_type == "power_law" else distances
    )
    log_friction = (
        distance_parameter * distance_feature
        + np.log(intra_multiplier) * same_zone
    )
    # IPF results are unchanged by one global rescaling of friction.
    friction = np.exp(log_friction - log_friction.max())
    flow, iterations = _balanced_flow(
        friction, O_series.to_numpy(), D_series.to_numpy()
    )
    probabilities = flow / O_series.to_numpy()[:, None]

    df_matrix = pd.DataFrame({
        "origin_id": pairs.get_level_values("origin_id"),
        "destination_id": pairs.get_level_values("destination_id"),
        "weight": probabilities.ravel(),
    })
    row_error = np.max(np.abs(probabilities.sum(axis=1) - 1.0))
    col_error = np.max(np.abs(flow.sum(axis=0) - D_series.to_numpy()))
    print(f"Gravity model:                     {model_type}")
    print(f"Distance parameter {parameter_name}:    {distance_parameter:.6f}")
    print(f"Intrazonal multiplier:             {intra_multiplier:.6f}")
    print(f"Gravity IPF iterations:            {iterations}")
    print(f"Origin production total:           {worker_total:,.1f}")
    print(f"Original destination job total:    {original_job_total:,.1f}")
    print(f"Applied destination scale:         {scale:.6f}")
    print(f"Maximum row probability error:     {row_error:.3e}")
    print(f"Maximum destination flow error:    {col_error:.3e}")

    return df_matrix[["origin_id", "destination_id", "weight"]]
