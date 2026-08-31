import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def configure(context):
    context.stage("asuncion.gravity.observed")
    context.stage("asuncion.gravity.distance_matrix")
    context.stage("asuncion.gravity.od_zones")


def execute(context):
    df_obs = context.stage("asuncion.gravity.observed").copy()
    df_dist = context.stage("asuncion.gravity.distance_matrix").copy()
    _, df_pop, df_emp, _ = context.stage("asuncion.gravity.od_zones")


    # Prepare inputs
    df_obs = (
        df_obs[["origin_id", "destination_id", "weight"]]
        .rename(columns={"weight": "observed_flow"})
    )

    df_pop = (
        df_pop.rename(columns={"macrozone_id": "origin_id"})
        [["origin_id", "population"]]
    )

    df_emp = (
        df_emp.rename(columns={"macrozone_id": "destination_id"})
        [["destination_id", "employees"]]
    )

    # Full OD matrix, including zero observed flows
    zones = sorted(
        set(df_pop.origin_id)
        | set(df_emp.destination_id)
        | set(df_dist.origin_id)
        | set(df_dist.destination_id)
    )

    df = pd.DataFrame(
        [(o, d) for o in zones for d in zones],
        columns=["origin_id", "destination_id"]
    )

    df = (
        df.merge(df_pop, on="origin_id", how="left")
          .merge(df_emp, on="destination_id", how="left")
          .merge(df_dist, on=["origin_id", "destination_id"], how="left")
          .merge(df_obs, on=["origin_id", "destination_id"], how="left")
    )

    df["observed_flow"] = df["observed_flow"].fillna(0)
    df["distance_km"] = df["distance_km"].replace(0, 0.1)

    df = df[
        (df.population > 0) &
        (df.employees > 0) &
        (df.distance_km > 0)
    ].copy()

    # Log variables
    df["ln_pop"] = np.log(df["population"])
    df["ln_emp"] = np.log(df["employees"])
    df["ln_dist"] = np.log(df["distance_km"])

    # PPML estimation
    model = smf.glm(
        "observed_flow ~ ln_pop + ln_emp + ln_dist",
        data=df,
        family=sm.families.Poisson()
    ).fit()

    p = model.params
    k = np.exp(p["Intercept"])
    a = p["ln_pop"]
    b = p["ln_emp"]
    g = p["ln_dist"]

    # Predictions
    df["predicted_flow"] = (
        k
        * df["population"]**a
        * df["employees"]**b
        * df["distance_km"]**g
    )

    corr = df[["observed_flow", "predicted_flow"]].corr().iloc[0, 1]

    print(f"k     = {k:.8e}")
    print(f"alpha = {a:.6f}")
    print(f"beta  = {b:.6f}")
    print(f"gamma = {g:.6f}")
    print(f"corr  = {corr:.4f}")

    return {
        "k": k,
        "alpha": a,
        "beta": b,
        "gamma": g,
        "correlation": corr,
        "data": df,
    }