"""Fit the Seville gravity kernel by PPML on the CITY-ONLY OD support.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from seville.gravity.observed import normalize_ids
from seville.gravity.calibration_area import select_calibration_inputs, select_city_od


def configure(context):
    context.stage("seville.gravity.observed")
    context.stage("seville.gravity.distance_matrix")
    context.stage("seville.gravity.od_zones")
    context.config("gravity_intra_km", 0.1)
    context.config("gravity_calibration_municipality_id", "41091")


def build_calibration_frame(observed, distances, population, employees, intra_km=0.1,
                            require_observed=True):
    """Dense origin-destination design with zero observed flows and valid logs."""
    if not np.isfinite(intra_km) or intra_km <= 0:
        raise ValueError("gravity_intra_km must be a positive finite distance in km")
    obs = observed[["origin_id", "destination_id", "weight"]].copy()
    dist = distances[["origin_id", "destination_id", "distance_km"]].copy()
    pop = population[["macrozone_id", "population"]].rename(
        columns={"macrozone_id": "origin_id"}).copy()
    emp = employees[["macrozone_id", "employees"]].rename(
        columns={"macrozone_id": "destination_id"}).copy()
    for df, cols in ((obs, ["origin_id", "destination_id"]),
                     (dist, ["origin_id", "destination_id"]),
                     (pop, ["origin_id"]), (emp, ["destination_id"])):
        for col in cols:
            df[col] = normalize_ids(df[col])
    for f, v in ((obs, "weight"), (dist, "distance_km"),
                 (pop, "population"), (emp, "employees")):
        f[v] = pd.to_numeric(f[v], errors="raise")
        if f[v].isna().any() or not np.isfinite(f[v]).all() or (f[v] < 0).any():
            raise ValueError(f"Invalid {v}: nonfinite or negative")
    for name, frame, cols in (("population", pop, ["origin_id"]),
                              ("jobs", emp, ["destination_id"]),
                              ("distances", dist, ["origin_id", "destination_id"])):
        if frame.duplicated(cols).any():
            raise ValueError(f"Duplicate {name} IDs")
    obs = obs.groupby(["origin_id", "destination_id"], as_index=False).weight.sum()
    modeled_o = sorted(pop.origin_id.unique())
    modeled_d = sorted(emp.destination_id.unique())
    pairs = pd.MultiIndex.from_product([modeled_o, modeled_d],
                                       names=["origin_id", "destination_id"]).to_frame(index=False)
    df = (pairs.merge(pop, on="origin_id", how="left", validate="many_to_one")
           .merge(emp, on="destination_id", how="left", validate="many_to_one")
           .merge(dist, on=["origin_id", "destination_id"], how="left", validate="one_to_one")
           .merge(obs.rename(columns={"weight": "observed_flow"}),
                  on=["origin_id", "destination_id"], how="left", validate="one_to_one"))
    if df.distance_km.isna().any():
        raise ValueError(f"Missing distances for {int(df.distance_km.isna().sum())} modeled OD pairs")
    unsupported = obs.merge(pairs, on=["origin_id", "destination_id"], how="left", indicator=True)
    lost = unsupported.loc[unsupported._merge.eq("left_only"), "weight"].sum()
    if lost:
        print(f"Seville PPML: excluded {lost:.2f} observed weight outside centroid support")
    df["observed_flow"] = df.observed_flow.fillna(0)
    df.loc[df.distance_km.eq(0), "distance_km"] = intra_km
    df = df.loc[(df.population > 0) & (df.employees > 0) & (df.distance_km > 0)].copy()
    if df.empty or (require_observed and df.observed_flow.sum() <= 0):
        raise ValueError("No positive observed OD weight on usable calibration support")
    for field, log_name in (("population", "ln_pop"), ("employees", "ln_emp"),
                            ("distance_km", "ln_dist")):
        df[log_name] = np.log(df[field])
    return df


def fit_gravity(df):
    model = smf.glm("observed_flow ~ ln_pop + ln_emp + ln_dist", data=df,
                    family=sm.families.Poisson()).fit()
    p = model.params
    if not np.isfinite(p).all():
        raise ValueError("PPML calibration produced nonfinite coefficients")
    fitted = df.copy()
    fitted["predicted_flow"] = model.predict(fitted)
    corr = fitted[["observed_flow", "predicted_flow"]].corr().iloc[0, 1]
    result = {"k": float(np.exp(p["Intercept"])), "alpha": float(p["ln_pop"]),
              "beta": float(p["ln_emp"]), "gamma": float(p["ln_dist"]),
              "correlation": float(corr), "n_od_pairs": len(fitted),
              "n_observed_positive": int((fitted.observed_flow > 0).sum()),
              "observed_weight_total": float(fitted.observed_flow.sum()),
              "data": fitted}
    print("Seville PPML: " + ", ".join(f"{k}={v:.6g}" for k, v in result.items()
                                        if k in ("k", "alpha", "beta", "gamma", "correlation")))
    return result


def execute(context):
    code = context.config("gravity_calibration_municipality_id")
    _, population, employees = context.stage("seville.gravity.od_zones")
    distances, population, employees = select_calibration_inputs(
        context.stage("seville.gravity.distance_matrix"), population, employees, code)
    observed = select_city_od(context.stage("seville.gravity.observed"), code)
    if observed.empty:
        raise ValueError("No city-to-city Seville HTS OD for calibration")
    df = build_calibration_frame(observed, distances, population, employees,
                                 float(context.config("gravity_intra_km")))
    result = fit_gravity(df)
    result["calibration_area"] = "Seville municipality city districts only"
    result["calibration_municipality_id"] = str(code)
    result["application_area"] = "Metropolitan area (all seville.gravity.od_zones)"
    return result
