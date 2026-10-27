"""Build Asuncion production-constrained workplace destination probabilities.

For every eligible home zone i, return p(destination j | home i) with
sum_j p_ij = 1. Destination job counts are attractiveness weights, NOT
capacity constraints; consequently no IPF and no forced destination totals.

"""

import numpy as np
import pandas as pd

from asuncion.gravity.calibrate_production_constrained import (
    check_model_type, choice_probabilities, load_choice_inputs,
)


def configure(context):
    context.stage("asuncion.gravity.distance_matrix")
    context.stage("asuncion.gravity.od_zones")
    context.stage("asuncion.gravity.calibrate_production_constrained")
    context.config("gravity_model_type", "power_law")
    context.config("gravity_intra_km", 1.0)


def execute(context):
    model_type = check_model_type(context.config("gravity_model_type"))
    intra_km = float(context.config("gravity_intra_km"))
    df_dist = context.stage("asuncion.gravity.distance_matrix")
    _, df_pop, df_emp, _ = context.stage("asuncion.gravity.od_zones")
    calibration = context.stage("asuncion.gravity.calibrate_production_constrained")

    if calibration["gravity_model_type"] != model_type:
        raise ValueError("Gravity configuration and calibration model types differ; rerun calibration")
    if not np.isclose(float(calibration["intra_km"]), intra_km):
        raise ValueError("Gravity intra_km configuration differs from calibration")
    beta = float(calibration["beta"])
    distance_parameter = float(calibration["distance_parameter"])
    intra_multiplier = float(calibration["intra_multiplier"])
    if not np.isfinite([beta, distance_parameter, intra_multiplier]).all() or intra_multiplier <= 0:
        raise ValueError("Invalid calibrated parameters")

    choices = load_choice_inputs(df_dist, df_pop, df_emp, intra_km)
    probabilities = choice_probabilities(
        choices, beta, distance_parameter, np.log(intra_multiplier), model_type
    )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10
    ):
        raise RuntimeError("Gravity probabilities are not finite and origin-normalized")

    pairs = pd.MultiIndex.from_product(
        [choices["origins"], choices["destinations"]],
        names=["origin_id", "destination_id"]
    )
    result = pd.DataFrame({
        "origin_id": pairs.get_level_values("origin_id"),
        "destination_id": pairs.get_level_values("destination_id"),
        "weight": probabilities.ravel(),
    })
    print(
        f"Production-constrained gravity ({model_type}): "
        f"{len(choices['origins'])} origins × {len(choices['destinations'])} destinations, "
        f"beta={beta:.4f}, distance={distance_parameter:.4f}, "
        f"intra_multiplier={intra_multiplier:.4f}"
    )
    print("Destination employment is attraction only; no job totals are enforced.")
    return result
