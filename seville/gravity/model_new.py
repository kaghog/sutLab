"""Seville workplace destination-choice probabilities from calibrated PPML.

`data.od.weighted` expects (origin_id, destination_id, weight) with weights
summing to one per origin. The PPML intercept and origin exponent cancel in
that normalization, while employment/distance coefficients govern choices.

Set `gravity_parameters_source: legacy` to reproduce the historic notebook's
hard-coded parameters without invoking calibration; default is `calibrated`.
"""

import numpy as np
import pandas as pd

from seville.gravity.observed import normalize_ids
from seville.gravity.calibrate import build_calibration_frame

DEFAULT_K = 3.46163624e-07
DEFAULT_A = 1.069159
DEFAULT_B = 1.108626
DEFAULT_G = -0.430266


def configure(context):
    context.stage("seville.gravity.distance_matrix")
    context.stage("seville.gravity.od_zones")
    source = context.config("gravity_parameters_source", "calibrated")
    if source not in ("calibrated", "legacy"):
        raise ValueError("gravity_parameters_source must be 'calibrated' or 'legacy'")
    if source == "calibrated":
        context.stage("seville.gravity.calibrate")
    context.config("gravity_intra_km", 0.1)


def make_probabilities(distances, population, employees, parameters, intra_km=0.1):
    """Compute probabilities on the same valid zone support as calibration."""
    # Build the full model design while ignoring HTS observations. This keeps
    # distance-zero and zero-attractiveness handling identical to calibration.
    blank = pd.DataFrame(columns=["origin_id", "destination_id", "weight"])
    df = build_calibration_frame(blank, distances, population, employees, intra_km,
                                 require_observed=False)
    k = float(parameters["k"]); a = float(parameters["alpha"])
    b = float(parameters["beta"]); g = float(parameters["gamma"])
    if not np.isfinite([k, a, b, g]).all() or k <= 0:
        raise ValueError("Invalid calibrated gravity coefficients")
    # Log weights avoid overflow/underflow when job counts are large.
    df["score"] = np.log(k) + a * df.ln_pop + b * df.ln_emp + g * df.ln_dist
    df["score"] -= df.groupby("origin_id").score.transform("max")
    df["weight"] = np.exp(df.score)
    totals = df.groupby("origin_id").weight.transform("sum")
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Nonfinite/zero gravity mass for an origin")
    df["weight"] /= totals
    result = df[["origin_id", "destination_id", "weight"]].copy()
    sums = result.groupby("origin_id").weight.sum()
    if not np.allclose(sums, 1.0, atol=1e-10, rtol=0):
        raise AssertionError("Gravity OD probabilities do not sum to one")
    return result


def execute(context):
    distances = context.stage("seville.gravity.distance_matrix")
    _, population, employees = context.stage("seville.gravity.od_zones")
    source = context.config("gravity_parameters_source")
    if source == "calibrated":
        parameters = context.stage("seville.gravity.calibrate")
    elif source == "legacy":
        parameters = {"k": DEFAULT_K, "alpha": DEFAULT_A,
                      "beta": DEFAULT_B, "gamma": DEFAULT_G}
    else:
        raise ValueError("Unsupported gravity_parameters_source")
    result = make_probabilities(distances, population, employees, parameters,
                                float(context.config("gravity_intra_km")))
    print(f"Seville gravity ({source}): {result.origin_id.nunique()} origins, "
          f"{result.destination_id.nunique()} destinations")
    return result
