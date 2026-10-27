"""Calibrate a production-constrained Asuncion work-location choice model.

Model for a resident of origin i choosing destination j:
    p_ij ∝ jobs_j ** beta * f(distance_ij) * q ** [i == j]

    gravity_model_type == 'power_law':   f(d) = d ** gamma
    gravity_model_type == 'exponential': f(d) = exp(slope * d)

Unlike doubly constrained gravity, jobs are attraction *weights*, not exact
workplace capacities. The HTS weighted work OD matrix supplies observations.
Every eligible workplace is included as an alternative, even with zero HTS
observations. Out-of-area HTS records are excluded before fitting.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

MODEL_TYPES = ("power_law", "exponential")
INITIAL_DISTANCE = {"power_law": -0.6, "exponential": -0.10}
DISTANCE_BOUNDS = {"power_law": (-6.0, 0.0), "exponential": (-2.0, 0.0)}


def configure(context):
    context.stage("asuncion.gravity.observed")
    context.stage("asuncion.gravity.distance_matrix")
    context.stage("asuncion.gravity.od_zones")
    context.config("gravity_model_type", "power_law")
    context.config("gravity_intra_km", 1.0)


def check_model_type(model_type):
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Invalid gravity_model_type={model_type!r}. Choose one of {MODEL_TYPES}."
        )
    return model_type


def normalized_ids(df, columns):
    """Canonical string IDs permit matching numeric/string IDs across sources.

    The application stage also outputs canonical strings; in the supplied
    Asuncion data, macrozone IDs are string-like district identifiers.
    """
    result = df.copy()
    for col in columns:
        if result[col].isna().any():
            raise ValueError(f"Null zone ID in {col}")
        # Avoid the common 1234.0 -> '1234.0' mismatch for numeric IDs.
        values = result[col].astype(str).str.strip()
        values = values.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
        if values.eq("").any():
            raise ValueError(f"Blank zone ID in {col}")
        result[col] = values
    return result


def load_choice_inputs(df_distances, df_population, df_employees, intra_km):
    """Common model choice set, job sizes, and positive OD distances.

    Origins: populated modeled zones; destinations: zones with positive jobs.
    All active OD distances must be supplied. Out-of-study HTS zones are NOT
    added here. `population` selects eligible origins only; it is not used as
    a number of travelers in the production-constrained probabilities.
    """
    required_dist = {"origin_id", "destination_id", "distance_km"}
    if not required_dist.issubset(df_distances.columns):
        raise ValueError(f"Missing distance columns: {required_dist-set(df_distances.columns)}")
    if "macrozone_id" not in df_population or "macrozone_id" not in df_employees:
        raise ValueError("od_zones must provide macrozone_id for population and jobs")
    if "employees" not in df_employees:
        raise ValueError("od_zones employee centroids must provide 'employees'")
    if not np.isfinite(intra_km) or intra_km <= 0:
        raise ValueError("gravity_intra_km must be finite and > 0")

    distances = normalized_ids(
        df_distances[list(required_dist)].copy(), ["origin_id", "destination_id"]
    )
    distances["distance_km"] = pd.to_numeric(distances["distance_km"], errors="raise")
    if distances.duplicated(["origin_id", "destination_id"]).any():
        raise ValueError("Duplicate (origin_id, destination_id) pairs in distance matrix")

    p = normalized_ids(df_population.copy(), ["macrozone_id"])
    j = normalized_ids(df_employees.copy(), ["macrozone_id"])
    # A zone can be retained as an origin by population or worker counts.
    if "workers" in p.columns:
        origin_column = "workers"
    elif "employed_population" in p.columns:
        origin_column = "employed_population"
    elif "population" in p.columns:
        origin_column = "population"
    else:
        raise ValueError("Population centroids require population, workers, or employed_population")

    p[origin_column] = pd.to_numeric(p[origin_column], errors="raise")
    j["employees"] = pd.to_numeric(j["employees"], errors="raise")
    for name, values in [("origin counts", p[origin_column]), ("job counts", j["employees"])]:
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise ValueError(f"Invalid {name}: expected nonnegative finite values")

    production = p.groupby("macrozone_id")[origin_column].sum().sort_index()
    jobs = j.groupby("macrozone_id")["employees"].sum().sort_index()
    production = production[production > 0]
    jobs = jobs[jobs > 0]
    if production.empty or jobs.empty:
        raise ValueError("No positive origin-production zones or workplace-job zones")

    origins = pd.Index(production.index, name="origin_id")
    destinations = pd.Index(jobs.index, name="destination_id")
    pairs = pd.MultiIndex.from_product([origins, destinations], names=["origin_id", "destination_id"])
    ds = distances.set_index(["origin_id", "destination_id"])["distance_km"].reindex(pairs)
    if ds.isna().any():
        examples = ds[ds.isna()].index.tolist()[:8]
        raise ValueError(
            f"Missing distances for {int(ds.isna().sum())} ACTIVE modeled OD pairs, "
            f"e.g. {examples}. Fix distance_matrix.py / od_zones.py. "
            "HTS zones outside the model are handled separately."
        )
    d = ds.to_numpy(dtype=float).reshape(len(origins), len(destinations))
    same_zone = origins.to_numpy()[:, None] == destinations.to_numpy()[None, :]
    d[(d == 0) & same_zone] = intra_km
    if not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError("Nonpositive or nonfinite active OD distances (excluding treated diagonal zeros)")
    return {
        "origins": origins,
        "destinations": destinations,
        "origin_counts": production,
        "origin_count_column": origin_column,
        "jobs": jobs,
        "distance_km": d,
        "same_zone": same_zone,
    }


def utilities(choices, beta, distance_parameter, log_intra, model_type):
    d = choices["distance_km"]
    distance_feature = np.log(d) if model_type == "power_law" else d
    return (
        beta * np.log(choices["jobs"].to_numpy(dtype=float))[None, :]
        + distance_parameter * distance_feature
        + log_intra * choices["same_zone"].astype(float)
    )


def choice_probabilities(choices, beta, distance_parameter, log_intra, model_type):
    u = utilities(choices, beta, distance_parameter, log_intra, model_type)
    return np.exp(u - logsumexp(u, axis=1, keepdims=True))


def execute(context):
    model_type = check_model_type(context.config("gravity_model_type"))
    intra_km = float(context.config("gravity_intra_km"))
    df_obs = context.stage("asuncion.gravity.observed").copy()
    df_dist = context.stage("asuncion.gravity.distance_matrix").copy()
    _, df_pop, df_emp, _ = context.stage("asuncion.gravity.od_zones")

    choices = load_choice_inputs(df_dist, df_pop, df_emp, intra_km)
    required = {"origin_id", "destination_id", "weight"}
    if not required.issubset(df_obs.columns):
        raise ValueError(f"Observed HTS missing columns: {required-set(df_obs.columns)}")
    obs = normalized_ids(df_obs[["origin_id", "destination_id", "weight"]],
                         ["origin_id", "destination_id"])
    obs["weight"] = pd.to_numeric(obs["weight"], errors="raise")
    if not np.isfinite(obs["weight"].to_numpy(dtype=float)).all() or (obs["weight"] < 0).any():
        raise ValueError("HTS weights must be nonnegative finite numbers")

    # Drop observations outside the actual application choice set. This
    # includes HTS-only zones (1119, 1118, 1105, 1504 in earlier data).
    valid = obs["origin_id"].isin(choices["origins"]) & obs["destination_id"].isin(choices["destinations"])
    excluded = obs.loc[~valid].copy()
    retained = obs.loc[valid].copy()
    all_weight = float(obs["weight"].sum())
    lost_weight = float(excluded["weight"].sum())
    coverage = {
        "hts_rows_input": int(len(obs)),
        "hts_rows_excluded": int(len(excluded)),
        "hts_rows_retained": int(len(retained)),
        "hts_weight_input": all_weight,
        "hts_weight_excluded": lost_weight,
        "hts_weight_retained": float(retained["weight"].sum()),
        "hts_weight_excluded_share": lost_weight / all_weight if all_weight else 0.0,
        "hts_origins_excluded": sorted(set(excluded["origin_id"]) - set(choices["origins"])),
        "hts_destinations_excluded": sorted(set(excluded["destination_id"]) - set(choices["destinations"])),
        "modeled_origin_zones": int(len(choices["origins"])),
        "modeled_destination_zones": int(len(choices["destinations"])),
    }
    print(
        f"HTS filtering: retained {len(retained):,}/{len(obs):,} rows, "
        f"excluded {lost_weight:,.2f} weighted work trips "
        f"({coverage['hts_weight_excluded_share']:.2%})."
    )
    if coverage["hts_weight_excluded_share"] > 0.1:
        warnings.warn("Over 10% of weighted HTS work trips are outside the model choice set.", RuntimeWarning)

    retained = retained.loc[retained["weight"] > 0]
    if retained.empty:
        raise ValueError("No positive weighted HTS work trips remain after filtering")

    # Full modeled destination choice set for every surveyed origin; an
    # unobserved OD pair is an available alternative, not a forbidden link.
    observed_origins = pd.Index(sorted(retained["origin_id"].unique()), name="origin_id")
    oi = choices["origins"].get_indexer(observed_origins)
    fit_choices = dict(choices)
    fit_choices["origins"] = observed_origins
    fit_choices["distance_km"] = choices["distance_km"][oi, :]
    fit_choices["same_zone"] = choices["same_zone"][oi, :]
    pairs = pd.MultiIndex.from_product(
        [observed_origins, choices["destinations"]],
        names=["origin_id", "destination_id"]
    )
    observed = (
        retained.groupby(["origin_id", "destination_id"])["weight"].sum()
        .reindex(pairs, fill_value=0.0)
        .to_numpy(dtype=float)
        .reshape(len(observed_origins), len(choices["destinations"]))
    )
    origin_total = observed.sum(axis=1)
    sample_size = float(origin_total.sum())
    distance_feature = np.log(fit_choices["distance_km"]) if model_type == "power_law" else fit_choices["distance_km"]
    log_jobs = np.log(choices["jobs"].to_numpy(dtype=float))[None, :]
    same_zone = fit_choices["same_zone"].astype(float)
    features = (np.broadcast_to(log_jobs, observed.shape), distance_feature, same_zone)

    def objective(x):
        beta, distance_param, log_intra = x
        u = beta * features[0] + distance_param * features[1] + log_intra * features[2]
        log_norm = logsumexp(u, axis=1, keepdims=True)
        probs = np.exp(u - log_norm)
        loss = float((np.sum(origin_total * log_norm[:, 0]) - np.sum(observed * u)) / sample_size)
        delta = origin_total[:, None] * probs - observed
        gradient = np.array([np.sum(delta * f) / sample_size for f in features])
        return loss, gradient

    x0 = np.array([1.0, INITIAL_DISTANCE[model_type], 0.0])
    bounds = [(0.0, 3.0), DISTANCE_BOUNDS[model_type], (np.log(0.05), np.log(20.0))]
    result = minimize(objective, x0, jac=True, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9})
    if not result.success:
        raise RuntimeError(f"Gravity HTS calibration did not converge: {result.message}")

    beta, distance_parameter, log_intra = map(float, result.x)
    intra_multiplier = float(np.exp(log_intra))
    probs = choice_probabilities(fit_choices, beta, distance_parameter, log_intra, model_type)
    predicted = probs * origin_total[:, None]

    df_diagnostics = pd.DataFrame({
        "origin_id": pairs.get_level_values("origin_id"),
        "destination_id": pairs.get_level_values("destination_id"),
        "distance_km": fit_choices["distance_km"].ravel(),
        "same_zone": fit_choices["same_zone"].ravel(),
        "jobs_2011": np.broadcast_to(choices["jobs"].to_numpy()[None, :], observed.shape).ravel(),
        "observed_flow": observed.ravel(),
        "predicted_flow": predicted.ravel(),
        "predicted_probability": probs.ravel(),
        "hts_origin_total": np.broadcast_to(origin_total[:, None], observed.shape).ravel(),
    })
    corr = (float(df_diagnostics[["observed_flow", "predicted_flow"]].corr().iloc[0, 1])
            if df_diagnostics["observed_flow"].std() > 0 else None)
    logloss = float(objective(result.x)[0])
    initial_logloss = float(objective(x0)[0])

    answer = {
        "gravity_model_type": model_type,
        "beta": beta,
        "distance_parameter": distance_parameter,
        "gamma": distance_parameter if model_type == "power_law" else None,
        "slope": distance_parameter if model_type == "exponential" else None,
        "intra_multiplier": intra_multiplier,
        "intra_km": intra_km,
        "mean_negative_log_likelihood": logloss,
        "initial_mean_negative_log_likelihood": initial_logloss,
        "correlation": corr,
        "hts_total_weight_used": sample_size,
        "hts_origins_used": int(len(observed_origins)),
        "coverage": coverage,
        "data": df_diagnostics,
    }
    print(f"Production-constrained HTS calibration: {model_type}")
    print(f"  beta (job attraction) = {beta:.6f}")
    print(f"  {'gamma' if model_type == 'power_law' else 'slope'} (distance) = {distance_parameter:.6f}")
    print(f"  intrazonal multiplier = {intra_multiplier:.6f}")
    print(f"  weighted mean negative log-likelihood = {logloss:.6f}")
    if any(abs(v - low) < 1e-4 or abs(v - high) < 1e-4
           for v, (low, high) in zip(result.x, bounds)):
        warnings.warn("At least one parameter is at a bound. Inspect survey coverage and identification.", RuntimeWarning)
    return answer
