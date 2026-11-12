"""Shared Seville gravity validation utilities; independent from SynPP context."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from seville.gravity.observed import normalize_ids


def clean_od(df, value="weight"):
    fields = ["origin_id", "destination_id", value]
    if not set(fields).issubset(df.columns):
        raise ValueError(f"OD table is missing {set(fields)-set(df.columns)}")
    out = df[fields].copy().dropna(subset=fields[:2])
    for c in fields[:2]:
        out[c] = normalize_ids(out[c])
    out[value] = pd.to_numeric(out[value], errors="raise")
    if not np.isfinite(out[value]).all() or (out[value] < 0).any():
        raise ValueError(f"Negative or nonfinite {value}")
    return out.groupby(fields[:2], as_index=False)[value].sum()


def reference_hashes(observed, zones):
    obs = clean_od(observed).sort_values(["origin_id", "destination_id"])
    hts = "\n".join(f"{x.origin_id}\t{x.destination_id}\t{x.weight:.17g}"
                    for x in obs.itertuples(index=False))
    geo = zones[["macrozone_id", "geometry"]].copy()
    geo["macrozone_id"] = normalize_ids(geo.macrozone_id)
    if geo.macrozone_id.isna().any() or geo.macrozone_id.duplicated().any():
        raise ValueError("Missing or duplicate OD-zone identifiers")
    geometry = "\n".join(f"{r.macrozone_id}\t{r.geometry.wkb_hex}"
                         for r in geo.sort_values("macrozone_id").itertuples(index=False))
    return {"hts_observed_od_sha256": hashlib.sha256(hts.encode()).hexdigest(),
            "model_zones_geometry_sha256": hashlib.sha256(
                (str(zones.crs) + "\n" + geometry).encode()).hexdigest(),
            "model_zones_crs": str(zones.crs), "n_model_zones": int(len(geo)),
            "n_hts_od_pairs": int(len(obs))}


def compare_od(observed, modeled, distances=None):
    """Return common-support conditional OD cells, destination shares, metrics.

    The model is conditional on the origin; predictions are weighted with the
    *same HTS home-origin mass* for both model and survey (fair comparison).
    No simulated-job stock is inferred from survey OD weights.
    """
    hts = clean_od(observed).rename(columns={"weight": "observed_flow"})
    model = clean_od(modeled).rename(columns={"weight": "model_probability"})
    total = float(hts.observed_flow.sum())
    if total <= 0:
        raise ValueError("Empty observed home/work OD")
    row_totals = model.groupby("origin_id").model_probability.sum()
    if not np.isfinite(row_totals).all() or not np.allclose(row_totals, 1, atol=1e-6):
        raise ValueError("Model OD rows must sum to one")
    # All model destinations, only HTS origins for which model has positive
    # observations on this support. Restricting outside pairs is explicit.
    covered = hts.merge(model[["origin_id", "destination_id"]],
                        on=["origin_id", "destination_id"], how="inner")
    if covered.empty or covered.observed_flow.sum() == 0:
        raise ValueError("No overlap between HTS OD and modeled OD")
    origins = sorted(covered.origin_id.unique())
    dests = sorted(model.destination_id.unique())
    grid = pd.MultiIndex.from_product([origins, dests],
            names=["origin_id", "destination_id"]).to_frame(index=False)
    grid = (grid.merge(covered, on=["origin_id", "destination_id"], how="left")
                 .merge(model, on=["origin_id", "destination_id"], how="left"))
    grid["observed_flow"] = grid.observed_flow.fillna(0.)
    grid["model_probability"] = grid.model_probability.fillna(0.)
    grid["hts_origin_total"] = grid.groupby("origin_id").observed_flow.transform("sum")
    grid["model_origin_total"] = grid.groupby("origin_id").model_probability.transform("sum")
    if (grid.hts_origin_total <= 0).any() or (grid.model_origin_total <= 0).any():
        raise ValueError("Empty origin after OD restriction")
    grid["observed_probability"] = grid.observed_flow / grid.hts_origin_total
    grid["model_probability"] /= grid.model_origin_total
    grid["predicted_flow"] = grid.model_probability * grid.hts_origin_total
    grid["difference"] = grid.model_probability - grid.observed_probability
    obs = grid.observed_probability.to_numpy(dtype=float)
    pred = grid.model_probability.to_numpy(dtype=float)
    corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) and np.std(pred) else None
    base = float(grid.observed_flow.sum())
    is_intra = grid.origin_id.eq(grid.destination_id)
    metrics = {
        "n_observed_pairs": int(len(hts)), "n_comparable_origins": len(origins),
        "n_comparable_destinations": len(dests), "n_comparable_od_pairs": len(grid),
        "hts_weight_total": total, "hts_weight_on_model_support": base,
        "hts_coverage_fraction": base/total,
        "hts_od_probability_correlation": corr,
        "hts_od_probability_rmse": float(np.sqrt(np.mean((obs-pred)**2))),
        "hts_od_probability_mae": float(np.mean(np.abs(obs-pred))),
        "hts_origin_weighted_tv": float(.5*np.abs(grid.difference).mul(grid.hts_origin_total).sum()/base),
        "hts_nll": float(-(grid.observed_flow*np.log(grid.model_probability.clip(lower=1e-300))).sum()/base),
        "hts_intrazonal_share": float(grid.loc[is_intra,"observed_flow"].sum()/base),
        "fitted_intrazonal_share": float(grid.loc[is_intra,"predicted_flow"].sum()/base),
    }
    shares = grid.groupby("destination_id", as_index=False)[["observed_flow", "predicted_flow"]].sum()
    shares["hts_share"] = shares.observed_flow/base
    shares["model_share"] = shares.predicted_flow/base
    metrics["destination_share_tv"] = float(.5*(shares.hts_share-shares.model_share).abs().sum())
    if distances is not None:
        dist = clean_od(distances, "distance_km")
        grid = grid.merge(dist, on=["origin_id", "destination_id"], how="left", validate="one_to_one")
        if grid.distance_km.notna().all():
            metrics["hts_mean_distance_km"] = float(np.average(grid.distance_km, weights=grid.observed_flow))
            metrics["model_mean_distance_km"] = float(np.average(grid.distance_km, weights=grid.predicted_flow))
    return grid, shares, metrics


def activity_od(activities, zones):
    """One home and one work point per synthetic person; map both to macrozones."""
    if activities.crs is None or zones.crs is None:
        raise ValueError("Spatial activities/zones must have a valid CRS")
    polygons = zones[["macrozone_id", "geometry"]].to_crs(activities.crs).copy()
    polygons["macrozone_id"] = normalize_ids(polygons.macrozone_id)
    if polygons.macrozone_id.duplicated().any():
        raise ValueError("OD zones have duplicate IDs")
    if not {"person_id", "purpose", "geometry"}.issubset(activities.columns):
        raise ValueError("Activities need person_id, purpose and geometry")

    def mapped(purpose):
        a = activities.loc[activities.purpose.astype("string").str.lower().eq(purpose),
                           ["person_id", "geometry"]].dropna(subset=["geometry"])
        a = a.drop_duplicates("person_id")
        if a.empty:
            return pd.DataFrame(columns=["person_id", f"{purpose}_zone"])
        geo = gpd.sjoin(gpd.GeoDataFrame(a, geometry="geometry", crs=activities.crs),
                        polygons, how="left", predicate="within")
        if geo.index.duplicated().any():
            raise ValueError("Synthetic activity mapped to multiple model zones")
        return geo[["person_id", "macrozone_id"]].rename(columns={"macrozone_id": f"{purpose}_zone"})

    homes = mapped("home")
    works = mapped("work")
    paired = homes.merge(works, on="person_id", validate="one_to_one")
    covered = paired.dropna(subset=["home_zone", "work_zone"])
    od = (covered.groupby(["home_zone", "work_zone"]).size().rename("persons_sample")
          .reset_index().rename(columns={"home_zone": "origin_id", "work_zone": "destination_id"}))
    return paired, od


def json_save(path, payload):
    def convert(v):
        if isinstance(v, dict): return {str(k):convert(x) for k,x in v.items()}
        if isinstance(v, (list,tuple,np.ndarray)):return [convert(x) for x in v]
        if isinstance(v,np.integer): return int(v)
        if isinstance(v,np.floating): return convert(float(v))
        if isinstance(v,float) and not np.isfinite(v): return None
        if isinstance(v,(np.bool_,)):return bool(v)
        if isinstance(v,Path): return str(v)
        if v is pd.NA:return None
        return v
    Path(path).write_text(json.dumps(convert(payload),indent=2,allow_nan=False),encoding="utf-8")
