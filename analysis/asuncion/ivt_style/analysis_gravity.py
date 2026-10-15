import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

"""

This stage serves to validate Gravity model and its OD model.

"""


def configure(context):
    context.stage("asuncion.gravity.model_new")
    context.stage("asuncion.data.hts.entd.reweighted")
    context.stage("asuncion.data.spatial.iris")
    context.config("analysis_path")
    context.stage("asuncion.gravity.od_zones")


def locations(df, purpose):
    a = df[df.preceding_purpose == purpose][
        ["person_id", "origin_district_id", "trip_weight"]
    ].rename(columns={"origin_district_id": "district_id"})

    b = df[df.following_purpose == purpose][
        ["person_id", "destination_district_id", "trip_weight"]
    ].rename(columns={"destination_district_id": "district_id"})

    return pd.concat([a, b]).dropna(subset=["district_id"]).drop_duplicates()


def to_districts(df, districts):
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32721")
    return gpd.sjoin(
        df.to_crs(districts.crs),
        districts[["district_id", "geometry"]],
        how="left",
        predicate="within",
    )


def execute(context):
    path = context.config("analysis_path")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    gravity = context.stage("asuncion.gravity.model_new").copy()
    _, _, trips = context.stage("asuncion.data.hts.entd.reweighted")
    trips = trips.copy()

    districts, _, _, _ = context.stage("asuncion.gravity.od_zones")
    districts = districts.copy()
    districts = districts.rename(columns={"macrozone_id": "district_id"})[["district_id", "geometry"]]

    # --------------------------------------------------
    # Observed HTS OD
    # --------------------------------------------------
    home = locations(trips, "home")

    work = locations(trips, "work")

    observed = (
        home.merge(
            work,
            on="person_id",
            suffixes=("_home", "_work"),
        )
        .dropna(subset=["district_id_home", "district_id_work"])
        .groupby(["district_id_home", "district_id_work"])
        ["trip_weight_home"]
        .sum()
        .reset_index(name="weight")
        .rename(columns={
            "district_id_home": "origin_id",
            "district_id_work": "destination_id",
        })
    )

    # --------------------------------------------------
    # Gravity: census sections -> districts
    # --------------------------------------------------

    gravity = (
        gravity.dropna(
            subset=["origin_id", "destination_id"]
        )
        .groupby(["origin_id", "destination_id"])["weight"]
        .sum()
        .reset_index()
    )

    # --------------------------------------------------
    # Normalize each OD by origin
    # --------------------------------------------------
    for df in [observed, gravity]:
        df["weight"] /= df.groupby("origin_id")["weight"].transform("sum")

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------
    df = gravity.merge(
        observed,
        on=["origin_id", "destination_id"],
        how="left",
        suffixes=("_gravity", "_observed"),
    )

    df["weight_observed"] = df["weight_observed"].fillna(0)
    df["diff"] = df.weight_gravity - df.weight_observed

    rmse = np.sqrt((df["diff"] ** 2).mean())
    corr = df.weight_gravity.corr(df.weight_observed)

    print(f"Gravity OD RMSE: {rmse:.6f}")
    print(f"Gravity OD correlation: {corr:.6f}")

    # --------------------------------------------------
    # Scatter
    # --------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(
        df.weight_observed,
        df.weight_gravity,
        alpha=0.3,
    )

    m = max(
        df.weight_observed.max(),
        df.weight_gravity.max(),
    )

    plt.plot([0, m], [0, m])
    plt.xlabel("Observed")
    plt.ylabel("Gravity")
    plt.title(f"OD comparison (RMSE={rmse:.4f}, r={corr:.4f})")
    plt.grid(linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{path}/gravity_od_scatter.png")
    plt.close()

    # --------------------------------------------------
    # OD matrices
    # --------------------------------------------------
    obs = observed.pivot(
        index="origin_id",
        columns="destination_id",
        values="weight",
    ).fillna(0)

    grav = gravity.pivot(
        index="origin_id",
        columns="destination_id",
        values="weight",
    ).fillna(0)

    origins = sorted(set(obs.index) | set(grav.index), key=str)
    destinations = sorted(set(obs.columns) | set(grav.columns), key=str)

    obs = obs.reindex(
        index=origins,
        columns=destinations,
        fill_value=0,
    )

    grav = grav.reindex(
        index=origins,
        columns=destinations,
        fill_value=0,
    )

    gravity["weight"] = pd.to_numeric(gravity["weight"], errors="coerce")
    observed["weight"] = pd.to_numeric(observed["weight"], errors="coerce")

    vmin = min(obs.min().min(), grav.min().min())
    vmax = max(obs.max().max(), grav.max().max())

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    obs = obs.astype(float)
    grav = grav.astype(float)

    ax[0].imshow(obs, cmap="viridis", vmin=vmin, vmax=vmax)
    ax[0].set_title("Observed OD Matrix")

    im = ax[1].imshow(
        grav,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax[1].set_title("Gravity OD Matrix")

    cbar_ax = fig.add_axes([1.40, 0.15, 0.02, 0.70])
    fig.colorbar(im, cax=cbar_ax)    
    plt.tight_layout()

    plt.savefig(f"{path}/gravity_od_matrices.png")
    plt.close()
