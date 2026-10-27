"""Archives different gravity model scenarios of Asuncion pipeline.

Records fitted calibration, model probabilities, population outputs, modeled
workplace OD, originating workforce and job attraction inputs, reference hashes,
and fit metrics. A separate future comparison stage will produce report figures.
Uses existing configuration keys, with optional known gravity defaults.
"""

from pathlib import Path
import datetime as dt
import hashlib
import tempfile
import json
import re
import shutil
import unicodedata

import numpy as np
import pandas as pd


# Only existing keys in config_asuncion.yml. In particular, there is no new
# gravity_constraint_type / gravity_runs_directory / archive_overwrite flag.
RUN_CONFIG_KEYS = (
    "gravity_run_name", "run_description", "gravity_model_type",
    "config_file", "gravity_calibration_stage", "gravity_od_stage",
    "processes", "hts", "sampling_rate", "acquisition_sample_size",
    "random_seed", "data_path", "output_path", "output_prefix",
    "analysis_path", "pipeline_crs", "work_assignment_method",
    "IPU_aggregation_level", "commune_equivalent", "home_location_sampling",
    "mode_choice", "matching_attributes", "education_assignment_method",
)


# These keys have already appeared in the gravity calibration configuration;
# no additional YAML setting is required when they are absent.
OPTIONAL_GRAVITY_CONFIG = {
    "gravity_intra_km": 1.0,
    "gravity_use_intra_multiplier": True,
}


def configure(context):
    # Declaring synthesis.output as dependency makes this stage an end-of-run
    # archive even when it is the only entry in config.run.
    context.stage("synthesis.output")
    context.stage("data.od.weighted")  # OD actually exposed to workplace assignment
    context.stage(context.config("gravity_calibration_stage"))
    context.stage(context.config("gravity_od_stage"))
    context.stage("asuncion.gravity.observed")
    context.stage("asuncion.gravity.distance_matrix")
    context.stage("asuncion.gravity.od_zones")
    context.stage("asuncion.data.codes")
    context.stage("asuncion.data.census.employees")
    context.stage("asuncion.data.census.employment")
    for key in RUN_CONFIG_KEYS:
        context.config(key)
    for key, default in OPTIONAL_GRAVITY_CONFIG.items():
        context.config(key, default)


def validate(context):
    # If input YAML changes, invalidate the archive stage's SynPP cache.
    path = Path(context.config("config_file")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Cannot archive configured YAML: {path}")
    return [(str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)]


def _native(x):
    if isinstance(x, dict):
        return {str(k): _native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [_native(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return _native(float(x))
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (Path, dt.datetime)):
        return str(x)
    if isinstance(x, (np.datetime64, pd.Timestamp)):
        return str(x)
    if x is pd.NA:
        return None
    return x


def _json(path, data):
    path.write_text(json.dumps(_native(data), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding="utf-8")


def _ids(df, cols=("origin_id", "destination_id")):
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = (out[col].astype("string").str.strip()
                        .str.replace(r"^(\d+)\.0$", r"\1", regex=True))
    return out


def _asuncion_zone_ids(codes, zones):
    """Infer Asuncion macrozones from the already-used spatial code table.

    Missing/incompatible metadata is nonfatal; Asuncion-specific metrics will
    then be absent, rather than fabricated.
    """
    if not isinstance(codes, pd.DataFrame):
        return set()
    geo_col = next((c for c in ("departement", "department") if c in codes), None)
    id_col = next((c for c in ("district_id", "macrozone_id") if c in codes), None)
    if not geo_col or not id_col:
        return set()
    def norm(s):
        return "".join(c for c in unicodedata.normalize("NFKD", str(s).upper())
                       if not unicodedata.combining(c)).strip()
    subset = codes[codes[geo_col].map(norm).eq("ASUNCION")]
    available = set(_ids(zones, ["macrozone_id"]).macrozone_id.dropna())
    return set(_ids(subset, [id_col])[id_col].dropna()) & available


def _copy_synthesis_files(output_path, output_prefix, destination):
    """Copy only synthesis.output artifacts, NOT analysis/ or old experiments.

    synthesis.output writes files named <output_prefix>* into output_path.
    The archive is itself below output/analysis in the supplied config, hence
    a recursive copy of output_path would be incorrect and self-referential.
    """
    source = Path(output_path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Synthesis output directory does not exist: {source}")
    if not output_prefix:
        raise ValueError("output_prefix must be nonempty to identify synthesis.output files")
    files = sorted(p for p in source.glob(f"{output_prefix}*") if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No synthesis.output files matching {output_prefix!r} in {source}")
    copied = []
    for src in files:
        dst = destination / src.name
        shutil.copy2(src, dst)
        copied.append({
            "filename": src.name,
            "bytes": dst.stat().st_size,
            "sha256": _hash_file(dst),
        })
    # These are the minimum inputs needed for later population-level checks.
    names = {item["filename"] for item in copied}
    required = {f"{output_prefix}persons.csv", f"{output_prefix}activities.csv"}
    missing = required - names
    if missing:
        raise FileNotFoundError(f"Missing required synthesis outputs: {sorted(missing)}")
    spatial_options = {
        f"{output_prefix}activities.gpkg",
        f"{output_prefix}activities.geoparquet",
    }
    if not names.intersection(spatial_options):
        raise FileNotFoundError(
            "Missing geocoded activities; future workplace-location validation "
            "would not be reproducible"
        )
    return copied



def _hash_file(path, chunk_bytes=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_od(frame):
    """Normalize and aggregate weighted HTS OD; used solely for a reference hash."""
    required = {"origin_id", "destination_id", "weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing HTS OD columns: {sorted(required-set(frame.columns))}")
    od = _ids(frame[["origin_id", "destination_id", "weight"]]).dropna()
    od["weight"] = pd.to_numeric(od["weight"], errors="raise")
    if not np.isfinite(od.weight.to_numpy(dtype=float)).all() or (od.weight < 0).any():
        raise ValueError("Invalid HTS weights")
    return (od.groupby(["origin_id", "destination_id"], as_index=False).weight.sum()
            .sort_values(["origin_id", "destination_id"]).reset_index(drop=True))


def _reference_hashes(observed, zones, codes, distances):
    """Detect changes to common HTS/geographies instead of silently comparing them."""
    od = _canonical_od(observed)
    rows = [f"{r.origin_id}\t{r.destination_id}\t{r.weight:.17g}"
            for r in od.itertuples(index=False)]
    hts_sha = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()

    if not {"macrozone_id", "geometry"}.issubset(zones.columns):
        raise ValueError("OD zones must contain macrozone_id and geometry")
    zones_norm = _ids(zones[["macrozone_id", "geometry"]], ["macrozone_id"])
    if zones_norm.macrozone_id.isna().any() or zones_norm.macrozone_id.duplicated().any():
        raise ValueError("OD zones have missing or duplicated macrozone identifiers")
    geometries = [f"{r.macrozone_id}\t{r.geometry.wkb_hex}"
                  for r in zones_norm.sort_values("macrozone_id").itertuples(index=False)]
    crs = str(getattr(zones, "crs", None))
    zone_sha = hashlib.sha256((crs + "\n" + "\n".join(geometries)).encode("utf-8")).hexdigest()

    # Distance treatment can differ between experiments (especially diagonals),
    # so it is stored as run input rather than assumed constant.
    return {
        "hts_observed_od_sha256": hts_sha,
        "model_zones_geometry_sha256": zone_sha,
        "model_zones_crs": crs,
        "n_model_zones": len(zones_norm),
        "n_hts_od_pairs": len(od),
        "n_distance_od_pairs": len(distances),
    }


def _write_model_inputs(folder, codes, zones, population_centroids,
                        employment_centroids, employees, employment, distances):
    """Preserve job origins, attraction counts and district labels for reuse."""
    inputs = folder / "model_inputs"
    inputs.mkdir()

    # A codes snapshot is small and preserves the district ID/name relationship.
    # Polygons need not be duplicated for every experiment; their hash is recorded.
    if not isinstance(codes, pd.DataFrame):
        raise TypeError("Expected a DataFrame from asuncion.data.codes")
    codes.to_csv(inputs / "district_codes.csv", index=False)
    _ids(zones.drop(columns="geometry"), ["macrozone_id"]).to_csv(
        inputs / "od_zone_attributes.csv", index=False)

    for name, df in [
        ("origin_centroids.csv", population_centroids),
        ("workplace_centroids.csv", employment_centroids),
    ]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for {name}")
        out = df.drop(columns="geometry", errors="ignore").copy()
        if "geometry" in df.columns:
            out["centroid_x"] = df.geometry.x.to_numpy()
            out["centroid_y"] = df.geometry.y.to_numpy()
            out["centroid_crs"] = str(getattr(df, "crs", None))
        _ids(out, ["macrozone_id"]).to_csv(inputs / name, index=False)

    if not isinstance(employees, pd.DataFrame) or not isinstance(employment, pd.DataFrame):
        raise TypeError("Employment and employees census stages must return DataFrames")
    # employees: original unscaled 2011 counts ('employees') AND scaled
    # attraction weights ('weight') from the actual pipeline stage.
    _ids(employees, ["district_id"]).to_csv(
        inputs / "workplace_census_employees.csv", index=False)
    _ids(employment, ["district_id"]).to_csv(
        inputs / "resident_census_employment.csv", index=False)
    _ids(distances).to_csv(inputs / "distance_matrix.csv", index=False)


def _write_synthesis_summary(synthesis_folder, prefix, output):
    """Store population denominators; work activities are NOT all employed workers."""
    filename = synthesis_folder / f"{prefix}persons.csv"
    persons = pd.read_csv(filename, sep=";", usecols=lambda c: c in ("person_id", "employed"))
    if "person_id" not in persons or "employed" not in persons:
        raise ValueError("Synthesis persons.csv needs person_id and employed fields")
    employed = persons.employed.astype(str).str.strip().str.lower().isin(["true", "1"])
    result = {
        "synthetic_persons_sample": int(persons.person_id.nunique()),
        "synthetic_employed_persons_sample": int(employed.sum()),
        "synthetic_not_employed_persons_sample": int((~employed).sum()),
        "synthetic_population_reference": "persons.csv; raw sample counts, not expanded",
    }
    _json(output / "synthetic_population_summary.json", result)
    return result


def _correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _compare_hts(observed, model, distances, output, asuncion_ids):
    # Unlike using an inner join on positive HTS rows, retain ALL modeled
    # destinations, including unobserved pairs (zero HTS counts).
    obs = _ids(observed[["origin_id", "destination_id", "weight"]]).dropna()
    obs["weight"] = pd.to_numeric(obs.weight, errors="raise")
    if (~np.isfinite(obs.weight)).any() or (obs.weight < 0).any():
        raise ValueError("Invalid HTS weights")
    obs = obs.groupby(["origin_id", "destination_id"], as_index=False).weight.sum()
    mod = _ids(model[["origin_id", "destination_id", "weight"]]).copy()
    mod["weight"] = pd.to_numeric(mod.weight, errors="raise")
    if mod.duplicated(["origin_id", "destination_id"]).any():
        raise ValueError("Repeated gravity OD pairs")
    if mod.weight.isna().any() or (mod.weight < 0).any():
        raise ValueError("Invalid modeled OD probabilities")
    row_total = mod.groupby("origin_id").weight.sum()
    if not np.allclose(row_total, 1, atol=1e-6):
        raise ValueError("Gravity OD probabilities do not sum to 1 by origin")
    distance = _ids(distances[["origin_id", "destination_id", "distance_km"]])
    if distance.duplicated(["origin_id", "destination_id"]).any():
        raise ValueError("Repeated OD distances")
    mod = mod.merge(distance, on=["origin_id", "destination_id"], how="left", validate="one_to_one")
    mod.to_csv(output / "model_probabilities_with_distance.csv", index=False)
    support = mod[["origin_id", "destination_id"]]
    tagged = obs.merge(support, on=["origin_id", "destination_id"],
                       how="left", indicator=True)
    excluded = tagged[tagged._merge == "left_only"].drop(columns="_merge")
    excluded.to_csv(output / "hts_excluded_od.csv", index=False)
    # Restrict observations by actual available model OD pairs.
    covered = tagged[tagged._merge == "both"].drop(columns="_merge")
    origins = covered.origin_id.unique()
    fit = mod[mod.origin_id.isin(origins)].merge(
        covered.rename(columns={"weight": "observed_flow"}),
        on=["origin_id", "destination_id"], how="left", validate="one_to_one")
    fit["observed_flow"] = fit.observed_flow.fillna(0)
    fit["hts_origin_total"] = fit.groupby("origin_id").observed_flow.transform("sum")
    fit["observed_probability"] = fit.observed_flow / fit.hts_origin_total.replace(0, np.nan)
    fit["fitted_hts_flow"] = fit.weight * fit.hts_origin_total
    fit.to_csv(output / "hts_vs_model_od.csv", index=False)

    n = float(fit.observed_flow.sum())
    measures = {
        "hts_total_weight": float(obs.weight.sum()),
        "hts_retained_weight": n,
        "hts_excluded_weight": float(excluded.weight.sum()),
        "hts_excluded_weight_share": (float(excluded.weight.sum() / obs.weight.sum())
                                       if obs.weight.sum() else None),
        "hts_excluded_od_rows": int(len(excluded)),
        "hts_retained_origins": int(len(origins)),
    }
    if n > 0:
        # Same definition as analysis_gravity.py: flatten every comparable
        # OD probability, including HTS zeros, and use Pearson correlation.
        a = fit.observed_probability.to_numpy(dtype=float)
        b = fit.weight.to_numpy(dtype=float)
        delta = b - a
        measures["hts_od_probability_correlation"] = _correlation(a, b)
        measures["hts_od_probability_rmse"] = float(np.sqrt(np.mean(delta ** 2)))
        measures["hts_od_probability_mae"] = float(np.mean(np.abs(delta)))
        different_zone = fit.origin_id.ne(fit.destination_id).to_numpy()
        if different_zone.any():
            measures["hts_interzonal_probability_correlation"] = _correlation(
                a[different_zone], b[different_zone])
        intra = fit.origin_id.eq(fit.destination_id)
        measures["hts_intrazonal_share"] = float(
            fit.loc[intra, "observed_flow"].sum() / n)
        measures["hts_fitted_intrazonal_share"] = float(
            fit.loc[intra, "fitted_hts_flow"].sum() / n)
        fit["hts_minus_gravity_probability"] = fit.observed_probability - fit.weight
        fit["gravity_minus_hts_probability"] = -fit.hts_minus_gravity_probability
        fit.to_csv(output / "hts_vs_model_od.csv", index=False)
        measures["hts_in_sample_nll"] = float(
            -(fit.observed_flow * np.log(fit.weight.clip(lower=1e-300))).sum() / n)
        measures["hts_in_sample_origin_tv"] = float(
            (0.5 * abs(fit.observed_probability - fit.weight) * fit.hts_origin_total).sum() / n)
        if fit.distance_km.notna().all():
            measures["hts_mean_distance_km"] = float(
                np.average(fit.distance_km, weights=fit.observed_flow))
            measures["hts_fitted_mean_distance_km"] = float(
                np.average(fit.distance_km, weights=fit.fitted_hts_flow))
        grouped = fit.groupby("destination_id", as_index=False).agg(
            hts_flow=("observed_flow", "sum"), model_flow=("fitted_hts_flow", "sum"))
        grouped["hts_share"] = grouped.hts_flow / n
        grouped["model_share"] = grouped.model_flow / n
        grouped.to_csv(output / "hts_destination_shares.csv", index=False)
        if asuncion_ids:
            local = grouped.destination_id.isin(asuncion_ids)
            measures["hts_asuncion_share"] = float(grouped.loc[local, "hts_share"].sum())
            measures["hts_fitted_asuncion_share"] = float(grouped.loc[local, "model_share"].sum())
            measures["hts_asuncion_share_gap_pp"] = 100 * (
                measures["hts_fitted_asuncion_share"] - measures["hts_asuncion_share"])
        distance_bins = [0, 1, 2, 5, 10, 15, 20, 30, 50, 100, np.inf]
        bands = ["0–1", "1–2", "2–5", "5–10", "10–15", "15–20", "20–30", "30–50", "50–100", "100+"]
        fit["distance_band_km"] = pd.cut(fit.distance_km, bins=distance_bins,
                                           labels=bands, include_lowest=True, right=False)
        histogram = fit.groupby("distance_band_km", observed=False).agg(
            hts_flow=("observed_flow", "sum"), model_flow=("fitted_hts_flow", "sum"))
        histogram.reindex(bands, fill_value=0).to_csv(output / "hts_distance_bands.csv")
    return measures, mod


def _final_od(synthesis_folder, prefix, zones, model, distances, output, asuncion_ids):
    """Compare sampled final work locations with the corresponding model probabilities.

    Final counts are persons with a simulated work activity on the modeled day;
    they are NOT the total number of employed persons in the census.
    """
    gpkg = synthesis_folder / f"{prefix}activities.gpkg"
    geoparquet = synthesis_folder / f"{prefix}activities.geoparquet"
    if not gpkg.is_file() and not geoparquet.is_file():
        return {"final_od_status": "not_available_no_spatial_activities"}
    try:
        import geopandas as gpd
        activities = gpd.read_file(gpkg) if gpkg.is_file() else gpd.read_parquet(geoparquet)
        geo = zones[["macrozone_id", "geometry"]].copy()
        if geo.crs is None or activities.crs is None:
            raise ValueError("Missing CRS for work activity spatial join")
        geo = geo.to_crs(activities.crs)
        geo = _ids(geo, ["macrozone_id"])
        if geo.macrozone_id.duplicated().any():
            geo = geo.dissolve(by="macrozone_id", as_index=False)

        def by_purpose(purpose):
            a = activities.loc[
                activities.purpose.astype(str).str.lower().eq(purpose),
                ["person_id", "geometry"]].dropna(subset=["geometry"])
            a = a.drop_duplicates("person_id")
            if a.empty:
                return pd.DataFrame(columns=["person_id", purpose + "_zone"])
            loc = gpd.sjoin(a, geo, how="left", predicate="within")
            loc = loc.loc[~loc.index.duplicated(keep="first")]
            return loc[["person_id", "macrozone_id"]].rename(
                columns={"macrozone_id": purpose + "_zone"})

        home = by_purpose("home")
        work = by_purpose("work")
        paired = home.merge(work, on="person_id", how="inner", validate="one_to_one")
        matched = paired.dropna(subset=["home_zone", "work_zone"]).copy()
        _ids(matched, ["home_zone", "work_zone"]).to_csv(
            output / "final_home_work_persons.csv", index=False)
        flows = matched.groupby(["home_zone", "work_zone"]).size().reset_index(name="persons_sample")
        flows = flows.rename(columns={"home_zone": "origin_id", "work_zone": "destination_id"})
        flows.to_csv(output / "final_home_work_od.csv", index=False)
        metrics = {
            "final_od_status": "ok", "final_work_people_sample": int(len(paired)),
            "final_work_people_geocoded": int(len(matched)),
            "final_work_people_unmapped": int(len(paired) - len(matched)),
        }
        if flows.empty:
            return metrics
        flows["final_origin_count"] = flows.groupby("origin_id").persons_sample.transform("sum")
        flows["final_probability"] = flows.persons_sample / flows.final_origin_count
        compared = flows.merge(model[["origin_id", "destination_id", "weight"]],
                               on=["origin_id", "destination_id"], how="left", validate="many_to_one")
        compared = compared.rename(columns={"weight": "model_probability"})
        compared.to_csv(output / "final_home_work_vs_model.csv", index=False)
        if compared.model_probability.notna().all():
            metrics["final_sampling_nll"] = float(
                -(compared.persons_sample * np.log(compared.model_probability.clip(lower=1e-300))).sum()
                / compared.persons_sample.sum())
        # Expected destination flows GIVEN the origins of people that actually
        # have a simulated work activity. This isolates sampling from the model.
        counts = flows.groupby("origin_id").persons_sample.sum().rename("persons_sample")
        expected = model.merge(counts, on="origin_id", how="inner")
        expected["expected_persons"] = expected.weight * expected.persons_sample
        expected.to_csv(output / "final_model_expected_od.csv", index=False)
        total = float(flows.persons_sample.sum())
        comparison = (flows.groupby("destination_id").persons_sample.sum()
                      .rename("realized_persons").to_frame()
                      .join(expected.groupby("destination_id").expected_persons.sum()
                            .rename("expected_persons"), how="outer").fillna(0).reset_index())
        comparison["realized_share"] = comparison.realized_persons / total
        comparison["expected_share"] = comparison.expected_persons / total
        comparison["difference_share_pp"] = 100 * (
            comparison.realized_share - comparison.expected_share)
        comparison.to_csv(output / "final_destination_shares.csv", index=False)
        metrics["final_vs_model_destination_tv"] = float(
            .5 * abs(comparison.realized_share - comparison.expected_share).sum())
        if asuncion_ids:
            local = comparison.destination_id.isin(asuncion_ids)
            metrics["final_asuncion_share"] = float(comparison.loc[local, "realized_share"].sum())
            metrics["expected_asuncion_share"] = float(comparison.loc[local, "expected_share"].sum())
            metrics["final_asuncion_share_gap_pp"] = 100 * (
                metrics["final_asuncion_share"] - metrics["expected_asuncion_share"])
        geom_dist = _ids(distances[["origin_id", "destination_id", "distance_km"]])
        length = flows.merge(geom_dist, on=["origin_id", "destination_id"], how="left")
        if length.distance_km.notna().all():
            metrics["final_mean_distance_km_centroid"] = float(
                np.average(length.distance_km, weights=length.persons_sample))
        return metrics
    except Exception as exc:
        return {"final_od_status": "failed", "final_od_error": str(exc)}




def execute(context):
    config = {key: context.config(key) for key in RUN_CONFIG_KEYS}
    config.update({
        key: context.config(key) for key in OPTIONAL_GRAVITY_CONFIG
    })
    name = config["gravity_run_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
        raise ValueError("gravity_run_name must be a safe directory name")

    root = (Path(config["analysis_path"]) / "gravity_runs").resolve()
    target = root / name
    if target.exists():
        raise FileExistsError(
            f"Run '{name}' already archived at {target}. Choose another gravity_run_name."
        )
    config_file = Path(config["config_file"]).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"config_file does not exist: {config_file}")

    context.stage("synthesis.output")
    calibration = context.stage(config["gravity_calibration_stage"])
    model = _ids(context.stage(config["gravity_od_stage"]))
    assignment_od = _ids(context.stage("data.od.weighted"))
    observed = context.stage("asuncion.gravity.observed")
    distances = context.stage("asuncion.gravity.distance_matrix")
    od_zones, origins, jobs, _ = context.stage("asuncion.gravity.od_zones")
    codes = context.stage("asuncion.data.codes")
    employees = context.stage("asuncion.data.census.employees")
    employment = context.stage("asuncion.data.census.employment")

    # Prepare everything in a temporary sibling directory. A failed extraction
    # must not leave a half-written archive that the report stage could mistake
    # for a complete run.
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.partial_", dir=root))
    try:
        analysis = temporary / "analysis"
        synthesis = temporary / "synthesis_output"
        analysis.mkdir()
        synthesis.mkdir()
        shutil.copy2(config_file, temporary / "config_as_used.yml")
        _json(temporary / "run_config.json", config)

        copied = _copy_synthesis_files(
            config["output_path"], config["output_prefix"], synthesis
        )
        model.to_csv(analysis / "model_probabilities.csv", index=False)
        # Separate snapshot of the actual SynPP OD alias for auditability.
        # No alias/stage consistency assertion is imposed on the run.
        assignment_od.to_csv(analysis / "assignment_od_probabilities.csv", index=False)
        if not isinstance(calibration, dict):
            raise TypeError(
                "Expected calibration stage to return a dictionary; "
                "cannot safely archive fitted parameters from another return type"
            )
        _json(analysis / "calibration_parameters.json", {
            k: v for k, v in calibration.items() if not isinstance(v, pd.DataFrame)
        })
        for key, value in calibration.items():
            if isinstance(value, pd.DataFrame):
                # Preserve all calibration diagnostics, not only a hardcoded
                # 'data' field used by the current calibrate.py.
                filename = "calibration_fit_od.csv" if key == "data" else (
                    "calibration_" + re.sub(r"[^A-Za-z0-9_-]", "_", str(key)) + ".csv"
                )
                value.to_csv(analysis / filename, index=False)

        _write_model_inputs(
            temporary, codes, od_zones, origins, jobs,
            employees, employment, distances
        )
        fingerprints = _reference_hashes(observed, od_zones, codes, distances)
        _json(analysis / "reference_fingerprints.json", fingerprints)

        asuncion_ids = _asuncion_zone_ids(codes, od_zones)
        measures, _ = _compare_hts(
            observed, model, distances, analysis, asuncion_ids
        )
        job_totals = _ids(jobs[["macrozone_id", "employees"]], ["macrozone_id"])
        job_totals = job_totals.groupby("macrozone_id", as_index=False).employees.sum()
        # Report the job-attraction totals actually passed to the gravity model.
        # Values may be 2011 counts or uniformly rescaled counts, depending on
        # the dataset and run: the archived census fields reveal which.
        job_totals.to_csv(analysis / "job_attraction_by_zone.csv", index=False)
        # Preserve this legacy name for existing compare_runs utilities.
        job_totals.to_csv(analysis / "jobs_2011_by_zone.csv", index=False)

        measures["job_attraction_total"] = float(
            pd.to_numeric(job_totals.employees, errors="raise").sum()
        )
        if {"employees", "weight"}.issubset(employees.columns):
            original = pd.to_numeric(employees["employees"], errors="coerce").sum()
            scaled = pd.to_numeric(employees["weight"], errors="coerce").sum()
            if original > 0:
                measures["cen2011_original_personnel_sum"] = float(original)
                measures["cen2011_to_model_weight_scale"] = float(scaled / original)
                measures["workplace_stage_attraction_total"] = float(scaled)
        if "weight" in employment.columns:
            measures["census_2022_employed_residents_total"] = float(
                pd.to_numeric(employment["weight"], errors="coerce").sum()
            )
        measures.update(_write_synthesis_summary(
            synthesis, config["output_prefix"], analysis
        ))
        final_measures = _final_od(
            synthesis, config["output_prefix"], od_zones,
            model, distances, analysis, asuncion_ids
        )
        # Report-grade archives must include final workplace locations.
        # Previously this failure was silently recorded only in metrics.json.
        if final_measures.get("final_od_status") != "ok":
            raise RuntimeError(
                "Could not archive realized synthetic OD: "
                f"{final_measures.get('final_od_status')}: "
                f"{final_measures.get('final_od_error', '')}"
            )
        measures.update(final_measures)
        # Distinct work-activity persons by assigned destination, not a count
        # of all employed residents and not a count of physical establishments.
        final_od = pd.read_csv(analysis / "final_home_work_od.csv",
                               dtype={"origin_id": "string", "destination_id": "string"})
        district_counts = final_od.groupby("destination_id", as_index=False).persons_sample.sum()
        district_counts = district_counts.rename(columns={"persons_sample": "work_activity_persons_sample"})
        district_counts.to_csv(analysis / "synthetic_workplace_counts_by_zone.csv", index=False)
        employed_count = measures.get("synthetic_employed_persons_sample", 0)
        if employed_count:
            measures["day_work_activity_people_over_employed_sample"] = (
                measures.get("final_work_people_geocoded", 0) / employed_count
            )
        measures.update({
            "gravity_run_name": name,
            "gravity_model_type": config["gravity_model_type"],
            "gravity_od_stage": config["gravity_od_stage"],
            "asuncion_zones_detected": len(asuncion_ids),
        })
        _json(analysis / "metrics.json", measures)

        # No plotting here: a later analysis stage will produce consistent,
        # publication-ready figures across archived runs.
        generated = [
            {"path": str(p.relative_to(temporary)), "bytes": p.stat().st_size}
            for p in sorted(temporary.rglob("*")) if p.is_file()
        ]
        _json(temporary / "manifest.json", {
            "schema_version": 3,
            "gravity_run_name": name,
            "run_description": config["run_description"],
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_config_sha256": _hash_file(config_file),
            "synthesis_files": copied,
            "files": generated,
            "shared_reference_fingerprints": fingerprints,
            "asuncion_zone_ids_inferred": sorted(asuncion_ids),
            "notes": [
                "HTS calibration and fit diagnostics are in-sample.",
                "Standalone HTS OD is not duplicated; model-aligned HTS diagnostics are kept.",
                "Zone geometry is not saved per run; zone IDs/attributes and geometry SHA256 are recorded.",
                "Origin and workplace census inputs AND distance matrix are archived as used.",
                "The configured gravity OD and the data.od.weighted assignment alias are saved separately.",
                "Realized OD counts reflect one work activity per person on the modeled day.",
                "The CEN 2011 employment stock is not target-year ground truth.",
                "jobs_2011_by_zone.csv is a legacy filename; inspect employees.stage weight for scaling.",
                "If the synthesis prefix is reused, clean stale prefix-matching outputs before running.",
                "Comparison plots are intentionally not generated by the archive stage.",
            ],
        })
        if target.exists():
            raise FileExistsError(f"Run archive appeared during execution: {target}")
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Archived '{name}' -> {target} ({len(copied)} synthesis files)")
    return str(target)
