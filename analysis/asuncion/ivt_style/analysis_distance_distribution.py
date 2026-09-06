import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def configure(context):
    context.stage("synthesis.population.spatial.locations")
    context.stage("data.hts.entd.reweighted")
    context.config("analysis_path")

def execute(context):
    df = context.stage("synthesis.population.spatial.locations")
    _, _, df_hts = context.stage("data.hts.entd.reweighted")

    df = df.sort_values(["person_id", "activity_index"]).copy()
    df["origin_geometry"] = df.groupby("person_id")["geometry"].shift()
    df["preceding_purpose"] = df.groupby("person_id")["purpose"].shift()
    df["origin_activity_index"] = df.groupby("person_id")["activity_index"].shift()
    df = df[df["origin_geometry"].notna()].copy()
    df["following_purpose"] = df["purpose"]
    df["crowfly_distance"] = gpd.GeoSeries(df["origin_geometry"], crs=df.crs).distance(df.geometry) / 1000
    df = df[(df["crowfly_distance"] > 0) & (df["crowfly_distance"] < 25)]

    df_hts = df_hts.copy()
    df_hts["crowfly_distance"] = df_hts["euclidean_distance"] / 1000
    df_hts = df_hts[df_hts["crowfly_distance"].notna()]
    df_hts = df_hts[(df_hts["crowfly_distance"] > 0) & (df_hts["crowfly_distance"] < 25)]

    purposes = ["home", "work", "education", "shop", "leisure", "other"]
    bins = np.arange(0, 25.5, 0.5)

    # Distance distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, purpose in zip(axes.flat, purposes):
        syn = df.loc[df["following_purpose"] == purpose, "crowfly_distance"]
        hts = df_hts.loc[df_hts["preceding_purpose"] == purpose, "crowfly_distance"]
        if len(syn) > 0:
            ax.hist(syn, bins=bins, density=True, alpha=0.5, label="Synthetic")
        if len(hts) > 0:
            ax.hist(hts, bins=bins, weights=df_hts.loc[hts.index, "trip_weight"], density=True, alpha=0.5, label="HTS")
        ax.set_title(purpose)
        ax.grid(alpha=0.2)
    axes[1, 1].set_xlabel("Crow-fly distance [km]")
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(context.config("analysis_path") + "/distance_distribution_by_purpose.png", dpi=300)
    plt.close(fig)

    # Distance CDFs
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, purpose in zip(axes.flat, purposes):
        syn = np.sort(df.loc[df["following_purpose"] == purpose, "crowfly_distance"])
        hts = df_hts[df_hts["preceding_purpose"] == purpose].sort_values("crowfly_distance")
        if len(syn) > 0:
            ax.plot(syn, np.arange(1, len(syn) + 1) / len(syn), label="Synthetic")
        if len(hts) > 0:
            ax.plot(hts["crowfly_distance"], hts["trip_weight"].cumsum() / hts["trip_weight"].sum(), label="HTS")
        ax.set_title(purpose)
        ax.grid(alpha=0.2)
    axes[1, 1].set_xlabel("Crow-fly distance [km]")
    axes[0, 0].set_ylabel("Cumulative share")
    axes[1, 0].set_ylabel("Cumulative share")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(context.config("analysis_path") + "/distance_cdf_by_purpose.png", dpi=300)
    plt.close(fig)