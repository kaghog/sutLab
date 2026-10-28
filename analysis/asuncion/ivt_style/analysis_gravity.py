"""Compare the precomputed HTS work OD with the gravity work OD probabilities.

Uses asuncion.gravity.observed directly: the HTS OD is NOT re-extracted
from trip chains in this stage.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.od.weighted")
    context.stage("asuncion.gravity.observed")
    context.config("analysis_path")
    context.config("gravity_run_name")


def _prepare_od(frame, name):
    """Validate, normalize ID representation, and aggregate duplicate OD pairs."""
    required = {"origin_id", "destination_id", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    df = frame[["origin_id", "destination_id", "weight"]].copy()
    df = df.dropna(subset=["origin_id", "destination_id"])

    for col in ("origin_id", "destination_id"):
        df[col] = (
            df[col].astype("string").str.strip()
            .str.replace(r"\.0+$", "", regex=True)
        )

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any() or not np.isfinite(df["weight"]).all():
        raise ValueError(f"{name} contains invalid OD weights")
    if (df["weight"] < 0).any():
        raise ValueError(f"{name} contains negative OD weights")

    return (
        df.groupby(["origin_id", "destination_id"], as_index=False)
        ["weight"].sum()
    )


def execute(context):
    run_name = str(context.config("gravity_run_name"))
    output_dir = Path(context.config("analysis_path"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load exactly the OD observations already generated for calibration.
    observed = _prepare_od(
        context.stage("asuncion.gravity.observed"), "Observed HTS OD"
    )
    gravity = _prepare_od(
        context.stage("data.od.weighted"), "Gravity OD"
    )

    # Restrict the HTS comparison to OD pairs available in the model.
    # Do this BEFORE normalizing, to use a common destination choice set.
    observed_total = observed["weight"].sum()
    n_observed_pairs = len(observed)
    observed = observed.merge(
        gravity[["origin_id", "destination_id"]],
        on=["origin_id", "destination_id"],
        how="inner",
        validate="one_to_one",
    )
    excluded_weight = observed_total - observed["weight"].sum()
    print(
        f"HTS: excluded {n_observed_pairs - len(observed)} OD pairs "
        f"not supported by the model "
        f"({excluded_weight:.3f} weighted respondents)."
    )
    if observed.empty or observed["weight"].sum() <= 0:
        raise ValueError("No observed HTS work OD flows remain in the model area")

    # Compare only origins with HTS observations. Keep ALL model destinations.
    origins = sorted(observed["origin_id"].unique(), key=str)
    destinations = sorted(gravity["destination_id"].unique(), key=str)
    gravity = gravity[gravity["origin_id"].isin(origins)].copy()

    complete_pairs = pd.MultiIndex.from_product(
        [origins, destinations], names=["origin_id", "destination_id"]
    )
    obs_flow = (
        observed.set_index(["origin_id", "destination_id"])["weight"]
        .reindex(complete_pairs, fill_value=0.0)
        .unstack("destination_id")
    )
    grav_flow = (
        gravity.set_index(["origin_id", "destination_id"])["weight"]
        .reindex(complete_pairs, fill_value=0.0)
        .unstack("destination_id")
    )

    obs_totals = obs_flow.sum(axis=1)
    grav_totals = grav_flow.sum(axis=1)
    if (obs_totals <= 0).any() or (grav_totals <= 0).any():
        raise ValueError("At least one compared origin has no valid OD weight")

    # Row-normalized destination-choice probabilities.
    obs_prob = obs_flow.div(obs_totals, axis=0)
    grav_prob = grav_flow.div(grav_totals, axis=0)

    obs_array = obs_prob.to_numpy(dtype=float)
    grav_array = grav_prob.to_numpy(dtype=float)
    differences = grav_array - obs_array

    rmse = float(np.sqrt(np.mean(differences ** 2)))
    mae = float(np.mean(np.abs(differences)))
    correlation = float(np.corrcoef(obs_array.ravel(), grav_array.ravel())[0, 1])
    origin_tv = 0.5 * np.sum(np.abs(differences), axis=1)
    weighted_tv = float(np.average(origin_tv, weights=obs_totals.to_numpy()))

    print(f"Compared HTS origins: {len(origins)}; model destinations: {len(destinations)}")
    print(f"OD RMSE: {rmse:.6f}")
    print(f"OD MAE: {mae:.6f}")
    print(f"OD correlation: {correlation:.6f}")
    print(f"Origin-weighted total variation: {weighted_tv:.6f}")

    # Save the exact OD values underlying both plots for further analysis.
    comparison = pd.DataFrame({
        "origin_id": complete_pairs.get_level_values("origin_id"),
        "destination_id": complete_pairs.get_level_values("destination_id"),
        "weight_observed": obs_array.ravel(),
        "weight_gravity": grav_array.ravel(),
        "difference": differences.ravel(),
    })
    comparison.to_csv(output_dir / f"{run_name}_gravity_od_comparison.csv", index=False)

    # Scatter plot: observed vs modeled conditional probabilities.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs_array.ravel(), grav_array.ravel(), alpha=0.35, s=20)
    limit = float(max(obs_array.max(), grav_array.max(), 0.01))
    ax.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Observed HTS probability", ylabel="Gravity probability",
           title=f"OD comparison (RMSE={rmse:.4f}, r={correlation:.3f})",
           xlim=(0, limit), ylim=(0, limit))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_name}_gravity_od_scatter.png", dpi=180)
    plt.close(fig)

    # Identical axes and color scale for a direct heatmap comparison.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmax = float(max(obs_array.max(), grav_array.max(), 0.01))
    for ax, values, title in zip(
        axes, (obs_array, grav_array), ("Observed HTS OD", "Gravity OD")
    ):
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Destination macrozone")
        ax.set_ylabel("Origin macrozone")
        ax.set_xticks(range(len(destinations)), destinations, rotation=90)
        ax.set_yticks(range(len(origins)), origins)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Probability within origin")
    fig.savefig(output_dir / f"{run_name}_gravity_od_matrices.png", dpi=180)
    plt.close(fig)

    return comparison
