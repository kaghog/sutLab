"""Spatial comparison of the FINAL Asuncion/AMA EQASIM population with census.

Residential geography: Asuncion barrios (boroughs), Central districts.
District geography: Asuncion and Central districts for district-only census data.
Outputs: PNG maps/plots and the tabulated values behind comparisons.

Dependencies: pandas, numpy, matplotlib, geopandas, openpyxl (for borough XLSX).
"""

from pathlib import Path
import re
import unicodedata

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd

METRIC_CRS = "EPSG:32721"  # UTM 21S, metric for the Asuncion metropolitan area
COLORS = {"final": "#147D92", "census": "#E39C42", "dark": "#22354D", "female": "#C15B87", "male": "#317FBA"}


def configure(context):
    context.config("output_path")
    context.config("output_prefix")
    context.config("analysis_path")
    context.config("data_path")
    context.config(
        "asuncion.population-ad-sex-bario",
        "census/population/Cuadro 1.1 Asunción. Población por sexo y edad mediana, según barrio, 2022.xlsx",
    )
    context.stage("synthesis.output")
    context.stage("asuncion.data.spatial.raw")
    context.stage("asuncion.data.codes")
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.data.census.households")
    context.stage("asuncion.data.census.employment")
    context.stage("asuncion.data.census.employees")


def _norm(values):
    """Normalize geographic names while keeping the original labels for maps."""
    def one(v):
        if pd.isna(v):
            return ""
        s = unicodedata.normalize("NFKD", str(v).upper().strip())
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", s)).strip()
    return values.map(one)


def _ids(values):
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _bool(values):
    return values.astype("string").str.strip().str.lower().map({
        "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
        "yes": 1.0, "no": 0.0, "si": 1.0, "sí": 1.0,
    })


def _sex(values):
    s = values.astype("string").str.lower().str.strip()
    observed = set(s.dropna().unique())
    if observed and observed <= {"0", "1"}:
        return s.replace({"0": "male", "1": "female"})
    if observed and observed <= {"1", "2"}:
        return s.replace({"1": "male", "2": "female"})
    return s.replace({"m": "male", "f": "female", "hombre": "male", "mujer": "female", "masculino": "male", "femenino": "female"})


def _numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _percent(num, den):
    d = pd.to_numeric(den, errors="coerce")
    if np.isscalar(d):
        d = np.nan if d == 0 else d
    else:
        d = d.replace(0, np.nan)
    return 100 * pd.to_numeric(num, errors="coerce") / d


def _make_zones(raw_spatial, codes):
    spatial = gpd.GeoDataFrame(raw_spatial.copy(), geometry="geometry", crs=raw_spatial.crs)
    if spatial.crs is None:
        spatial = spatial.set_crs("EPSG:4674")  # supplied spatial.raw stage output
    spatial = spatial.to_crs(METRIC_CRS)
    spatial = spatial[spatial.geometry.notna() & ~spatial.geometry.is_empty].copy()
    for col in ("departement", "district", "borough"):
        spatial[col] = spatial[col].astype("string").str.strip()
        spatial[f"_{col}"] = _norm(spatial[col])
    is_asuncion = spatial["_departement"].eq("ASUNCION")
    spatial["zone_name"] = np.where(is_asuncion, spatial["borough"], spatial["district"])
    spatial["_zone_name"] = np.where(is_asuncion, spatial["_borough"], spatial["_district"])
    spatial["zone_key"] = spatial["_departement"] + "|" + spatial["_district"] + "|" + spatial["_zone_name"]
    spatial["district_key"] = spatial["_departement"] + "|" + spatial["_district"]

    code = codes.copy()
    code["_departement"] = _norm(code["departement"])
    code["_district"] = _norm(code["district"])
    code["district_key"] = code["_departement"] + "|" + code["_district"]
    code["district_id"] = _ids(code["district_id"])
    district_lookup = code[["district_key", "district_id"]].drop_duplicates()
    # Name-based join to geographic polygons, because spatial.raw returns no IDs.
    spatial = spatial.merge(district_lookup, on="district_key", how="left")
    spatial = gpd.GeoDataFrame(spatial, geometry="geometry", crs=METRIC_CRS)
    # A district may consist of several parts. Dissolve repeated features.
    first = spatial.dissolve(by="zone_key", as_index=False, aggfunc="first")
    first["area_km2"] = first.geometry.area / 1e6
    first = first[["zone_key", "district_key", "district_id", "zone_name", "departement", "district", "area_km2", "geometry"]]
    district = first.dissolve(by="district_key", as_index=False, aggfunc="first")
    district["area_km2"] = district.geometry.area / 1e6
    district["zone_name"] = district["district"]
    return first, district


def _read_borough_census(context):
    """The population stage only returns district IDs, so the borough XLSX is essential."""
    path = Path(context.config("data_path")) / context.config("asuncion.population-ad-sex-bario")
    raw = pd.read_excel(path, sheet_name="C1.1_ASUNCIÓN", skiprows=7, skipfooter=1, dtype=str)
    raw = raw.iloc[:, [1, 2, 3, 4, 5]].copy()
    raw.columns = ["borough", "total", "male", "female", "age_median"]
    raw["_borough"] = _norm(raw["borough"])
    raw = raw[(raw["_borough"] != "") & (raw["_borough"] != "TOTAL")].copy()
    for c in ["total", "male", "female"]:
        raw[c] = _numeric(raw[c])
    # In the original reference stage the authoritative by-sex sum is used.
    raw["population"] = raw["male"] + raw["female"]
    out = raw.groupby("_borough", as_index=False).agg(
        census_population=("population", "sum"), census_female=("female", "sum")
    )
    out["zone_key"] = "ASUNCION|ASUNCION|" + out["_borough"]
    return out


def _references(context, mixed, district, output_dir):
    pop = context.stage("asuncion.data.census.population").copy()
    hh = context.stage("asuncion.data.census.households").copy()
    employment = context.stage("asuncion.data.census.employment").copy()
    employees = context.stage("asuncion.data.census.employees").copy()
    for d in (pop, hh, employment, employees):
        if "district_id" in d:
            d["district_id"] = _ids(d["district_id"])
        if "commune_id" in d:
            d["commune_id"] = _ids(d["commune_id"])
    # Census population: district, sex, age (2022)
    pop["sex_norm"] = _sex(pop["sex"])
    pop["age_class"] = _numeric(pop["age_class"])
    pop["weight"] = _numeric(pop["weight"])
    census_pop = pop.groupby("district_id", as_index=False).agg(census_population=("weight", "sum"))
    census_fem = pop[pop["sex_norm"].eq("female")].groupby("district_id")["weight"].sum().rename("census_female")
    under15 = pop[pop["age_class"] < 15].groupby("district_id")["weight"].sum().rename("census_under15")
    age65 = pop[pop["age_class"] >= 65].groupby("district_id")["weight"].sum().rename("census_65plus")
    pop15 = pop[pop["age_class"] >= 15].groupby("district_id")["weight"].sum().rename("census_population_15plus")
    for series in (census_fem, under15, age65, pop15):
        census_pop = census_pop.join(series, on="district_id")
    census_pop[["census_under15", "census_65plus"]] = census_pop[["census_under15", "census_65plus"]].fillna(0)
    census_pop["census_female_pct"] = _percent(census_pop["census_female"], census_pop["census_population"])
    census_pop["census_under15_pct"] = _percent(census_pop["census_under15"], census_pop["census_population"])
    census_pop["census_65plus_pct"] = _percent(census_pop["census_65plus"], census_pop["census_population"])

    hh["weight"] = _numeric(hh["weight"])
    hk = "commune_id" if "commune_id" in hh else "district_id"
    census_hh = hh.groupby(hk)["weight"].sum().rename("census_households")
    employment["weight"] = _numeric(employment["weight"])
    ek = "commune_id" if "commune_id" in employment else "district_id"
    census_emp = employment.groupby(ek)["weight"].sum().rename("census_employed")
    employees["weight"] = _numeric(employees["weight"])
    census_jobs = employees.groupby("district_id")["weight"].sum().rename("census_workplace_employees_2011")
    census_pop = census_pop.join(census_hh, on="district_id").join(census_emp, on="district_id").join(census_jobs, on="district_id")
    census_pop["census_employed_per_15plus_pct"] = _percent(census_pop["census_employed"], census_pop["census_population_15plus"])

    # Asuncion boroughs from original 2022 XLSX, Central from district census stage.
    borough = _read_borough_census(context)
    borough_data = borough[["zone_key", "census_population", "census_female"]]
    central = mixed[mixed["departement"].map(lambda x: _norm(pd.Series([x])).iloc[0]) != "ASUNCION"][ ["zone_key", "district_id"] ].copy()
    central = central.merge(census_pop[["district_id", "census_population", "census_female"]], on="district_id", how="left")
    mixed_ref = pd.concat([borough_data, central[["zone_key", "census_population", "census_female"]]], ignore_index=True)
    mixed_ref["census_female_pct"] = _percent(mixed_ref["census_female"], mixed_ref["census_population"])

    missing_borough = borough.loc[~borough["zone_key"].isin(mixed["zone_key"]), ["_borough", "census_population"]]
    missing_borough.to_csv(output_dir / "census_boroughs_unmatched_to_geometry.csv", index=False)
    spatial_matched = mixed[["departement", "district", "zone_name", "zone_key"]].merge(
        mixed_ref[["zone_key", "census_population"]], on="zone_key", how="left")
    spatial_unknown = spatial_matched.loc[spatial_matched["census_population"].isna(),
                                          ["departement", "district", "zone_name", "zone_key"]]
    spatial_unknown.to_csv(output_dir / "spatial_zones_unmatched_to_census.csv", index=False)
    mixed.loc[mixed["district_id"].isna(), ["departement", "district", "zone_name", "district_key"]].to_csv(
        output_dir / "spatial_zones_unmatched_to_codes.csv", index=False)
    return census_pop, mixed_ref


def _read_export(context):
    folder = Path(context.config("output_path"))
    prefix = context.config("output_prefix")
    def csv(name):
        return pd.read_csv(folder / f"{prefix}{name}.csv", sep=";", encoding="latin1")
    def geo(name):
        return gpd.read_file(folder / f"{prefix}{name}.gpkg")
    persons = csv("persons")
    households = csv("households")
    homes = geo("homes")
    acts = geo("activities")
    import json
    meta = folder / f"{prefix}meta.json"
    rate = float(json.loads(meta.read_text(encoding="utf-8")).get("sampling_rate", 1.0)) if meta.exists() else 1.0
    if not 0 < rate <= 1:
        raise ValueError(f"Invalid sampling rate: {rate}")
    return persons, households, homes, acts, rate


def _assign(points, polygons):
    """Point-in-polygon to zones. No duplication of points at polygon boundaries."""
    pts = points.copy()
    if pts.crs is None:
        raise ValueError("Point locations require a CRS (cannot spatially join without it)")
    pts = pts.to_crs(METRIC_CRS)
    poly = polygons[["zone_key", "district_key", "district_id", "geometry"]].copy()
    joined = gpd.sjoin(pts, poly, how="left", predicate="within")
    # A point in overlapping polygon features must not be double-counted.
    joined = joined[~joined.index.duplicated(keep="first")].copy()
    return joined.drop(columns="index_right", errors="ignore")


def _home_locations(homes, acts):
    h = homes[["household_id", "geometry"]].copy().dropna(subset=["geometry"]).drop_duplicates("household_id")
    # Some runs have a household without a homes.gpkg record. Use its home activity.
    fallback = acts.loc[acts["purpose"].astype(str).str.lower() == "home", ["household_id", "geometry"]].dropna(subset=["geometry"])
    fallback = fallback[~fallback["household_id"].isin(h["household_id"])].drop_duplicates("household_id")
    if len(fallback):
        fallback = fallback.to_crs(h.crs)
        h = gpd.GeoDataFrame(pd.concat([h, fallback], ignore_index=True), geometry="geometry", crs=homes.crs)
    return h


def _person_synthetic(persons, households, homes, acts, zones, sampling_rate):
    home_coords = _home_locations(homes, acts)
    located = _assign(home_coords, zones)
    hh = households.copy().merge(located[["household_id", "zone_key", "district_key", "district_id"]], on="household_id", how="left")
    p = persons.copy().merge(hh[["household_id", "zone_key", "district_key", "district_id"]], on="household_id", how="left")
    p["sex"] = _sex(p["sex"])
    p["age"] = _numeric(p["age"])
    p["is_employed"] = _bool(p["employed"])
    license_col = "has_driving_license" if "has_driving_license" in p else "has_license"
    p["has_license_norm"] = _bool(p[license_col])
    hh["expansion_weight"] = 1 / sampling_rate
    p["expansion_weight"] = 1 / sampling_rate
    return p, hh


def _aggregate_people(p, key, rate):
    q = p[p[key].notna()].copy()
    q["female"] = (q["sex"] == "female").astype(int)
    q["under15"] = (q["age"] < 15).astype(int)
    q["age65plus"] = (q["age"] >= 65).astype(int)
    q["adult"] = (q["age"] >= 18).astype(int)
    q["working_age"] = q["age"].between(15, 64).astype(int)
    q["employed_yes"] = q["is_employed"].eq(1).astype(int)
    q["employed_15plus"] = ((q["age"] >= 15) & q["is_employed"].eq(1)).astype(int)
    q["licensed_yes"] = q["has_license_norm"].eq(1).astype(int)
    q["female_working_age"] = ((q["sex"] == "female") & q["age"].between(15, 64)).astype(int)
    q["male_working_age"] = ((q["sex"] == "male") & q["age"].between(15, 64)).astype(int)
    q["female_wa_emp"] = ((q["sex"] == "female") & q["age"].between(15, 64) & q["is_employed"].eq(1)).astype(int)
    q["male_wa_emp"] = ((q["sex"] == "male") & q["age"].between(15, 64) & q["is_employed"].eq(1)).astype(int)
    q["licensed_adult"] = ((q["age"] >= 18) & q["has_license_norm"].eq(1)).astype(int)
    q["female_adult"] = ((q["sex"] == "female") & (q["age"] >= 18)).astype(int)
    q["male_adult"] = ((q["sex"] == "male") & (q["age"] >= 18)).astype(int)
    q["female_licensed_adult"] = ((q["sex"] == "female") & (q["age"] >= 18) & q["has_license_norm"].eq(1)).astype(int)
    q["male_licensed_adult"] = ((q["sex"] == "male") & (q["age"] >= 18) & q["has_license_norm"].eq(1)).astype(int)
    sums = {
        "population_sample": ("person_id", "size"),
        "female_sample": ("female", "sum"),
        "under15_sample": ("under15", "sum"),
        "age65plus_sample": ("age65plus", "sum"),
        "employed_sample": ("employed_yes", "sum"),
        "employed_15plus_sample": ("employed_15plus", "sum"),
        "licensed_sample": ("licensed_yes", "sum"),
        "adult_sample": ("adult", "sum"),
        "working_age_sample": ("working_age", "sum"),
        "female_wa_sample": ("female_working_age", "sum"),
        "male_wa_sample": ("male_working_age", "sum"),
        "female_emp_wa_sample": ("female_wa_emp", "sum"),
        "male_emp_wa_sample": ("male_wa_emp", "sum"),
        "licensed_adult_sample": ("licensed_adult", "sum"),
        "female_adult_sample": ("female_adult", "sum"),
        "male_adult_sample": ("male_adult", "sum"),
        "female_licensed_adult_sample": ("female_licensed_adult", "sum"),
        "male_licensed_adult_sample": ("male_licensed_adult", "sum"),
    }
    # Spatial differences in rate by age group (not share of all residents).
    age_groups = {
        "employment_15_24": (15, 24, "is_employed"),
        "employment_25_44": (25, 44, "is_employed"),
        "employment_45_64": (45, 64, "is_employed"),
        "license_18_24": (18, 24, "has_license_norm"),
        "license_25_44": (25, 44, "has_license_norm"),
        "license_45_64": (45, 64, "has_license_norm"),
        "license_65plus": (65, 150, "has_license_norm"),
    }
    for tag, (amin, amax, variable) in age_groups.items():
        valid = q["age"].between(amin, amax) & q[variable].notna()
        q[f"{tag}_den"] = valid.astype(int)
        q[f"{tag}_num"] = (valid & q[variable].eq(1)).astype(int)
        sums[f"{tag}_den"] = (f"{tag}_den", "sum")
        sums[f"{tag}_num"] = (f"{tag}_num", "sum")
    # One row per resident, not one row per activity or trip.
    out = q.groupby(key).agg(**sums)
    for tag in age_groups:
        out[f"{tag}_pct"] = _percent(out[f"{tag}_num"], out[f"{tag}_den"])
        out.loc[out[f"{tag}_den"] < 10, f"{tag}_pct"] = np.nan
    for col in ["population_sample", "female_sample", "under15_sample", "age65plus_sample", "employed_sample", "licensed_sample"]:
        out[col.replace("_sample", "_scaled")] = out[col] / rate
    out["female_pct"] = _percent(out["female_sample"], out["population_sample"])
    out["under15_pct"] = _percent(out["under15_sample"], out["population_sample"])
    out["age65plus_pct"] = _percent(out["age65plus_sample"], out["population_sample"])
    out["employment_15_64_pct"] = _percent(out["female_emp_wa_sample"] + out["male_emp_wa_sample"], out["working_age_sample"])
    out.loc[out["working_age_sample"] < 15, "employment_15_64_pct"] = np.nan
    out["license_18plus_pct"] = _percent(out["licensed_adult_sample"], out["adult_sample"])
    out.loc[out["adult_sample"] < 15, "license_18plus_pct"] = np.nan
    for name, num, den in [
        ("employment_female_15_64_pct", "female_emp_wa_sample", "female_wa_sample"),
        ("employment_male_15_64_pct", "male_emp_wa_sample", "male_wa_sample"),
        ("license_female_18plus_pct", "female_licensed_adult_sample", "female_adult_sample"),
        ("license_male_18plus_pct", "male_licensed_adult_sample", "male_adult_sample"),
    ]:
        out[name] = _percent(out[num], out[den])
        out.loc[out[den] < 10, name] = np.nan
    out["employment_gender_gap_pp"] = (_percent(out["male_emp_wa_sample"], out["male_wa_sample"])
                                         - _percent(out["female_emp_wa_sample"], out["female_wa_sample"]))
    out["license_gender_gap_pp"] = (_percent(out["male_licensed_adult_sample"], out["male_adult_sample"])
                                      - _percent(out["female_licensed_adult_sample"], out["female_adult_sample"]))
    out.loc[out[["female_wa_sample", "male_wa_sample"]].min(axis=1) < 10, "employment_gender_gap_pp"] = np.nan
    out.loc[out[["female_adult_sample", "male_adult_sample"]].min(axis=1) < 10, "license_gender_gap_pp"] = np.nan
    return out.reset_index()


def _plots_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
                         "axes.titleweight": "semibold", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})


def _save(fig, output_dir, name):
    fig.savefig(output_dir / name, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _map(gdf, col, title, name, output_dir, cbar_label, diverging=False, cmap=None,
         limits=None, subset=None, note=None):
    g = gdf[gdf["geometry"].notna()].copy()
    if subset == "asuncion":
        g = g[_norm(g["departement"]).eq("ASUNCION")].copy()
    if len(g) == 0:
        return
    values = _numeric(g[col])
    valid = values.dropna()
    if valid.empty:
        return
    cmap = cmap or ("RdBu_r" if diverging else "viridis")
    if limits is not None:
        vmin, vmax = limits
    elif diverging:
        cap = float(max(np.nanquantile(abs(valid), 0.97), 1.0))
        vmin, vmax = -cap, cap
    else:
        vmin = max(0., float(np.nanquantile(valid, 0.01))) if valid.min() >= 0 else float(valid.min())
        vmax = max(float(np.nanquantile(valid, 0.98)), vmin + 1e-5)
    # Non-negative ratio centered on 1 (e.g. final / census).
    if diverging:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(12, 10))
    g.plot(column=col, ax=ax, cmap=cmap, norm=norm, edgecolor="#65778C", linewidth=0.38,
           missing_kwds={"color": "#E6E7EA", "edgecolor": "#9CA3AF", "hatch": "///", "label": "No census / location data"})
    # Fixed common legend scale when plotting census vs final in paired charts.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.014, shrink=0.75)
    cb.set_label(cbar_label)
    ax.set_title(title, loc="left", pad=16)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if note:
        fig.text(0.10, 0.055, note, fontsize=9, color="#536276")
    _save(fig, output_dir, name)


def _paired_maps(g, col1, col2, title, filename, output_dir, label):
    one = _numeric(g[col1])
    two = _numeric(g[col2])
    vals = pd.concat([one, two]).dropna()
    if vals.empty:
        return
    upper = max(float(vals.quantile(0.99)), 1.0)
    norm = Normalize(vmin=0, vmax=upper)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, col, text in zip(axes, (col1, col2), ("Census reference", "Final synthetic, expanded")):
        g.plot(column=col, ax=ax, cmap="YlGnBu", norm=norm,
               edgecolor="#52667B", linewidth=0.36,
               missing_kwds={"color":"#E6E7EA", "hatch": "///"})
        ax.set_title(text, loc="left")
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap="YlGnBu")
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes), fraction=0.027, pad=0.012, shrink=0.65, label=label)
    fig.suptitle(title, x=0.12, ha="left", fontsize=16, fontweight="bold")
    _save(fig, output_dir, filename)


def _ranked_bars(df, name_col, diff_col, title, output_dir, filename, unit, threshold=None):
    data = df[[name_col, diff_col] + (["population_sample"] if "population_sample" in df else [])].copy()
    data[diff_col] = _numeric(data[diff_col])
    data = data.dropna(subset=[diff_col])
    if threshold is not None and "population_sample" in data:
        data = data[data["population_sample"] >= threshold]
    if data.empty:
        return
    # Show the largest absolute differences, with positive / negative signs preserved.
    data = data.loc[data[diff_col].abs().nlargest(min(22, len(data))).index].sort_values(diff_col)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.34 * len(data) + 2)))
    ax.barh(data[name_col].astype(str), data[diff_col], color=np.where(data[diff_col] < 0, "#D77566", "#148D94"))
    ax.axvline(0, color="#2D3748", linewidth=1)
    ax.set_title(title, loc="left", pad=14)
    ax.set_xlabel(unit)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    _save(fig, output_dir, filename)


def _scatter(df, xcol, ycol, label_col, title, output_dir, filename):
    d = df[[xcol, ycol, label_col]].copy().dropna()
    d = d[(d[xcol] > 0) & (d[ycol] >= 0)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(d[xcol], d[ycol], color=COLORS["final"], edgecolors="white", s=65, alpha=0.8)
    maxv = max(d[[xcol, ycol]].max()) * 1.07
    ax.plot([0, maxv], [0, maxv], "--", color=COLORS["census"], label="Perfect agreement")
    ax.set_xlim(0, maxv)
    ax.set_ylim(0, maxv)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Census residents")
    ax.set_ylabel("Final residents (expanded from sample)")
    ax.set_title(title, loc="left")
    d["error"] = abs(d[ycol] / d[xcol] - 1)
    for _, r in d.nlargest(min(10, len(d)), "error").iterrows():
        ax.annotate(str(r[label_col]), (r[xcol], r[ycol]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.16)
    _save(fig, output_dir, filename)


def _compare_metrics(geography, agg, ref, key, rate):
    g = geography.merge(agg, on=key, how="left").merge(ref, on=key if key == "zone_key" else "district_id", how="left", suffixes=("", "_ref"))
    # Unoccupied zones are genuinely zero, not missing. Missing references remain NaN.
    count_cols = [c for c in agg if c.endswith("_sample") or c.endswith("_scaled")]
    g[count_cols] = g[count_cols].fillna(0)
    g["population_density_final"] = g["population_scaled"] / g["area_km2"]
    g["population_density_census"] = g["census_population"] / g["area_km2"]
    g["density_difference"] = g["population_density_final"] - g["population_density_census"]
    g["population_error_pct"] = _percent(g["population_scaled"] - g["census_population"], g["census_population"])
    g["population_ratio"] = g["population_scaled"] / g["census_population"].replace(0, np.nan)
    matched = g["census_population"].notna()
    g["population_share_final_pct"] = _percent(g["population_scaled"], g.loc[matched, "population_scaled"].sum()).where(matched)
    g["population_share_census_pct"] = _percent(g["census_population"], g.loc[matched, "census_population"].sum())
    g["population_share_gap_pp"] = g["population_share_final_pct"] - g["population_share_census_pct"]
    g["female_share_gap_pp"] = g["female_pct"] - g["census_female_pct"]
    g["census_employed_density"] = g.get("census_employed", pd.Series(np.nan, index=g.index)) / g["area_km2"]
    g["final_employed_density"] = g["employed_scaled"] / g["area_km2"]
    return g


def _work_geographies(acts, persons, zones, district, rate):
    # Exactly one location per person with an observed 'work' activity that day.
    work = acts.loc[acts["purpose"].astype(str).str.lower().eq("work"), ["person_id", "geometry"]]
    work = work.dropna(subset=["geometry"]).drop_duplicates("person_id").copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    geo = _assign(work, zones)
    geo = geo.merge(persons[["person_id", "district_key"]].rename(columns={"district_key": "home_district"}), on="person_id", how="left")
    zone_counts = geo.groupby("zone_key").size().rename("work_people_sample").reset_index()
    zone_counts["work_people_scaled"] = zone_counts["work_people_sample"] / rate
    district_counts = geo.groupby("district_key").size().rename("work_people_sample").reset_index()
    district_counts["work_people_scaled"] = district_counts["work_people_sample"] / rate
    flows = geo.dropna(subset=["home_district", "district_key"]).groupby(["home_district", "district_key"]).size().rename("count_sample").reset_index()
    return zone_counts, district_counts, flows


def _workplace_plot(district_compare, output_dir):
    d = district_compare.dropna(subset=["census_workplace_employees_2011"]).copy()
    d = d[d["census_workplace_employees_2011"] > 0]
    if d.empty:
        return
    d["share_2011"] = _percent(d["census_workplace_employees_2011"], d["census_workplace_employees_2011"].sum())
    d["share_today"] = _percent(d["work_people_sample"], d["work_people_sample"].sum())
    d = d.sort_values("census_workplace_employees_2011", ascending=False).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(7, len(d) * 0.42)))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["share_2011"], height=.38, label="Census workplace employees (2011)", color=COLORS["census"])
    ax.barh(y + 0.2, d["share_today"], height=.38, label="Final persons at work activity (model day)", color=COLORS["final"])
    ax.set_yticks(y, d["zone_name"].astype(str))
    ax.set_xlabel("Share of recorded workplaces / work activities (%)")
    ax.set_title("Distribution of work locations by district", loc="left", pad=14)
    ax.text(0, -0.085, "Different vintages and concepts: 2011 job stock vs one-day modeled work activity", transform=ax.transAxes, color="#687482", fontsize=9)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=.16)
    _save(fig, output_dir, "41_workplace_share_2011_vs_final.png")


def _od_heatmap(flows, districts, output_dir):
    if flows.empty:
        return
    names = districts.set_index("district_key")["zone_name"]
    top = flows.groupby("home_district")["count_sample"].sum().nlargest(18).index
    selected = flows[flows["home_district"].isin(top) & flows["district_key"].isin(top)]
    matrix = selected.pivot_table(index="home_district", columns="district_key", values="count_sample", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(index=top, columns=top, fill_value=0)
    # Denominators include work destinations outside the top-18 districts.
    all_by_origin = flows.groupby("home_district")["count_sample"].sum().reindex(top).replace(0, np.nan)
    pct = matrix.div(all_by_origin, axis=0) * 100
    fig, ax = plt.subplots(figsize=(12, 10))
    plot = ax.imshow(pct.to_numpy(), cmap="YlOrRd", vmin=0, vmax=max(5, np.nanquantile(pct, 0.99)))
    ax.set_xticks(np.arange(len(top)), [names.get(v, v) for v in top], rotation=70, ha="right")
    ax.set_yticks(np.arange(len(top)), [names.get(v, v) for v in top])
    ax.set_xlabel("District of work activity")
    ax.set_ylabel("District of home")
    ax.set_title("Where residents work • district-to-district flows", loc="left", pad=15)
    fig.colorbar(plot, ax=ax, shrink=0.8, label="% of origin district's observed workers")
    _save(fig, output_dir, "42_home_to_work_district_matrix.png")


def execute(context):
    _plots_style()
    output_dir = Path(context.config("analysis_path")) / "spatial_population_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    persons, households, homes, acts, rate = _read_export(context)
    raw_spatial = context.stage("asuncion.data.spatial.raw")
    codes = context.stage("asuncion.data.codes")
    mixed, districts = _make_zones(raw_spatial, codes)
    census_district, census_mixed = _references(context, mixed, districts, output_dir)

    person_loc, house_loc = _person_synthetic(persons, households, homes, acts, mixed, rate)
    mixed_agg = _aggregate_people(person_loc, "zone_key", rate)
    district_agg = _aggregate_people(person_loc, "district_key", rate)
    # District geometry is keyed by district_key; join the reference by district_id.
    mixed_g = _compare_metrics(mixed, mixed_agg, census_mixed, "zone_key", rate)
    d = districts.merge(district_agg, on="district_key", how="left")
    d = d.merge(census_district, on="district_id", how="left")
    for c in [c for c in district_agg if c.endswith("_sample") or c.endswith("_scaled")]:
        d[c] = d[c].fillna(0)
    d["population_density_final"] = d["population_scaled"] / d["area_km2"]
    d["population_density_census"] = d["census_population"] / d["area_km2"]
    d["population_error_pct"] = _percent(d["population_scaled"] - d["census_population"], d["census_population"])
    matched = d["census_population"].notna()
    d["population_share_final_pct"] = _percent(d["population_scaled"], d.loc[matched, "population_scaled"].sum()).where(matched)
    d["population_share_census_pct"] = _percent(d["census_population"], d.loc[matched, "census_population"].sum())
    d["population_share_gap_pp"] = d["population_share_final_pct"] - d["population_share_census_pct"]
    d["under15_gap_pp"] = d["under15_pct"] - d["census_under15_pct"]
    d["age65plus_gap_pp"] = d["age65plus_pct"] - d["census_65plus_pct"]
    d["female_share_gap_pp"] = d["female_pct"] - d["census_female_pct"]
    # Comparisons of district employee rates are not strict like-for-like because
    # EPHC district employment is allocated using population shares.
    d["final_employment_15plus_pct"] = _percent(d["employed_15plus_sample"], d["population_sample"] - d["under15_sample"])
    d["employment_rate_gap_pp"] = d["final_employment_15plus_pct"] - d["census_employed_per_15plus_pct"]
    hh_counts = house_loc.groupby("district_key").size().rename("households_sample")
    d = d.join(hh_counts, on="district_key")
    d["households_sample"] = d["households_sample"].fillna(0)
    d["household_density_final"] = (d["households_sample"] / rate) / d["area_km2"]
    d["household_density_census"] = d["census_households"] / d["area_km2"]
    d["household_error_pct"] = _percent(d["households_sample"] / rate - d["census_households"], d["census_households"])
    d["mean_household_size"] = d["population_sample"] / d["households_sample"].replace(0, np.nan)

    # Work geography: assigned from activity points, not commuter line geometries.
    work_z, work_d, flows = _work_geographies(acts, person_loc, mixed, districts, rate)
    if not work_z.empty:
        mixed_g = mixed_g.merge(work_z, on="zone_key", how="left")
        d = d.merge(work_d, on="district_key", how="left")
    else:
        for frame in (mixed_g, d):
            frame["work_people_sample"] = 0
            frame["work_people_scaled"] = 0
    for frame in (mixed_g, d):
        frame["work_people_sample"] = frame["work_people_sample"].fillna(0)
        frame["work_people_scaled"] = frame["work_people_scaled"].fillna(0)
        frame["work_activity_density"] = frame["work_people_scaled"] / frame["area_km2"]
    d["residential_employed_scaled"] = d["employed_15plus_sample"] / rate
    d["work_vs_resident_employed_ratio"] = d["work_people_sample"] / d["employed_15plus_sample"].replace(0, np.nan)
    matched_jobs = d["census_workplace_employees_2011"].notna()
    d["work_activity_share_pct"] = _percent(d["work_people_sample"], d.loc[matched_jobs, "work_people_sample"].sum()).where(matched_jobs)
    d["census_jobs_share_pct"] = _percent(d["census_workplace_employees_2011"], d.loc[matched_jobs, "census_workplace_employees_2011"].sum())
    d["work_location_share_gap_pp"] = d["work_activity_share_pct"] - d["census_jobs_share_pct"]

    # Save numerical tables, necessary to inspect which zones cause a map difference.
    mixed_g.drop(columns="geometry").to_csv(output_dir / "residential_borough_and_district.csv", index=False)
    d.drop(columns="geometry").to_csv(output_dir / "district_population_employment_and_workplaces.csv", index=False)
    if not flows.empty:
        flows.to_csv(output_dir / "home_to_work_district_flows.csv", index=False)
    pd.DataFrame([{
        "sampling_rate": rate,
        "expansion_factor": 1 / rate,
        "total_exported_persons": len(persons),
        "residents_assigned_to_polygons": int(person_loc["zone_key"].notna().sum()),
        "residents_without_assigned_zone": int(person_loc["zone_key"].isna().sum()),
        "total_exported_households": len(households),
        "households_assigned_to_polygons": int(house_loc["zone_key"].notna().sum()),
        "polygons_without_district_id": int(mixed["district_id"].isna().sum()),
        "zones_without_matched_census_population": int(mixed_g["census_population"].isna().sum()),
        "districts_without_matched_census_population": int(d["census_population"].isna().sum()),
        "persons_with_geocoded_work_activity": int(acts.loc[
            acts["purpose"].astype(str).str.lower().eq("work") & acts.geometry.notna(), "person_id"
        ].nunique()),
        "work_activity_people_assigned_to_polygons": int(work_z["work_people_sample"].sum()) if not work_z.empty else 0,
        "reference_census_year": 2022,
        "workplace_census_year": 2011,
    }]).to_csv(output_dir / "spatial_assignment_summary.csv", index=False)

    fmt = f"Final expanded from {rate:.2%} export (x{1/rate:g}); census residents from 2022"
    # Residential geography: boroughs in Asuncion, districts in Central.
    _paired_maps(mixed_g, "population_density_census", "population_density_final",
                 "Residential population density • borough / district", "01_population_density_comparison.png",
                 output_dir, "Residents per km²")
    _map(mixed_g, "population_error_pct", "Residential population difference • final vs census",
         "02_population_percent_difference.png", output_dir, "Final minus census (% of census)", diverging=True, note=fmt)
    _map(mixed_g, "population_share_gap_pp", "Distribution of AMA residents • final − census",
         "03_population_share_gap.png", output_dir, "Percentage-point difference", diverging=True,
         note="Shares sum across the same geography; less sensitive to total-size differences")
    _map(mixed_g, "population_density_final", "Final residential population density • mixed geography",
         "04_final_population_density.png", output_dir, "Expanded residents / km²", note=fmt)
    _map(mixed_g, "population_error_pct", "Asunción boroughs • final − census population",
         "05_asuncion_borough_population_difference.png", output_dir, "Final minus census (%)", diverging=True,
         subset="asuncion", note=fmt)
    _map(mixed_g, "population_density_final", "Asunción boroughs • final residential density",
         "06_asuncion_borough_density.png", output_dir, "Expanded residents / km²", subset="asuncion", note=fmt)
    _ranked_bars(mixed_g, "zone_name", "population_error_pct", "Largest residential population discrepancies",
                 output_dir, "07_largest_population_differences.png", "Final − census (%)", threshold=20)
    _scatter(mixed_g, "census_population", "population_scaled", "zone_name",
             "Residents by area • census vs final", output_dir, "08_population_scatter.png")
    _map(mixed_g, "female_pct", "Share of women among residents • final",
         "09_female_share.png", output_dir, "Women (% of residents)", cmap="PuRd")
    _map(mixed_g, "female_share_gap_pp", "Female population share • final − census",
         "10_female_share_gap.png", output_dir, "Percentage-point difference", diverging=True)
    _map(mixed_g, "under15_pct", "Resident population younger than 15 • final",
         "11_under15_share.png", output_dir, "Residents under 15 (%)", cmap="YlGn")
    _map(mixed_g, "age65plus_pct", "Residents aged 65+ • final",
         "12_age65plus_share.png", output_dir, "Residents 65+ (%)", cmap="OrRd")
    _map(mixed_g, "employment_15_64_pct", "Employment flag among residents aged 15–64 • final",
         "13_employment_rate_residential.png", output_dir, "Employed among 15–64 (%)", cmap="YlGnBu")
    _map(mixed_g, "license_18plus_pct", "Driving licences among adult residents • final",
         "14_license_rate_residential.png", output_dir, "Licensed among 18+ (%)", cmap="PuBuGn")
    _map(mixed_g, "employment_gender_gap_pp", "Employment rate gap • men minus women, ages 15–64",
         "15_employment_gender_gap.png", output_dir, "Male − female (percentage points)", diverging=True)
    _map(mixed_g, "license_gender_gap_pp", "Driving licence rate gap • men minus women, 18+",
         "16_license_gender_gap.png", output_dir, "Male − female (percentage points)", diverging=True)

    # Age-specific and sex-specific rates among eligible residents (small groups masked).
    for filename, column, title in [
        ("29_employment_female_15_64.png", "employment_female_15_64_pct", "Women's employment • resident ages 15–64"),
        ("30_employment_male_15_64.png", "employment_male_15_64_pct", "Men's employment • resident ages 15–64"),
        ("31_license_female_18plus.png", "license_female_18plus_pct", "Women's driving licence ownership • 18+"),
        ("32_license_male_18plus.png", "license_male_18plus_pct", "Men's driving licence ownership • 18+"),
        ("33_employment_15_24.png", "employment_15_24_pct", "Employment among residents aged 15–24"),
        ("34_employment_25_44.png", "employment_25_44_pct", "Employment among residents aged 25–44"),
        ("35_employment_45_64.png", "employment_45_64_pct", "Employment among residents aged 45–64"),
        ("36_license_18_24.png", "license_18_24_pct", "Driving licence ownership among residents aged 18–24"),
        ("37_license_25_44.png", "license_25_44_pct", "Driving licence ownership among residents aged 25–44"),
        ("38_license_45_64.png", "license_45_64_pct", "Driving licence ownership among residents aged 45–64"),
        ("39_license_65plus.png", "license_65plus_pct", "Driving licence ownership among residents aged 65+"),
    ]:
        _map(mixed_g, column, title, filename, output_dir, "Share within group (%)",
             cmap="YlGnBu", limits=(0, 100),
             note="Areas with fewer than 10 persons in the target age/sex group are hatched")

    # District geography: supports population age/sex and employment census tables.
    _paired_maps(d, "population_density_census", "population_density_final",
                 "Population density at district level • 2022 vs final",
                 "17_district_population_density.png", output_dir, "Residents per km²")
    _map(d, "under15_gap_pp", "Share younger than 15 • final − 2022 census, districts",
         "18_district_under15_gap.png", output_dir, "Percentage-point difference", diverging=True)
    _map(d, "age65plus_gap_pp", "Share aged 65+ • final − 2022 census, districts",
         "19_district_age65plus_gap.png", output_dir, "Percentage-point difference", diverging=True)
    _map(d, "employment_rate_gap_pp", "Employment among residents 15+ • final − allocated 2022 reference",
         "20_district_employment_gap.png", output_dir, "Percentage-point difference", diverging=True,
         note="Reference district employment is model-allocated, not direct district survey estimates")
    _paired_maps(d, "household_density_census", "household_density_final",
                 "Household density • census 2022 vs final", "21_household_density_comparison.png",
                 output_dir, "Households per km²")
    _map(d, "household_error_pct", "Households per district • final − census",
         "22_household_error_pct.png", output_dir, "Final − census (% of census)", diverging=True)
    _map(d, "mean_household_size", "Residents per household • final, district level",
         "23_household_size.png", output_dir, "Mean persons per household", cmap="YlGnBu")

    # Household car access at the residence, not a count of entries in vehicles.csv.
    house_loc["has_car"] = (_numeric(house_loc["number_of_cars"]).fillna(0) > 0).astype(int)
    ownership = house_loc[house_loc["zone_key"].notna()].groupby("zone_key").agg(
        hh_with_car=("has_car", "sum"), hh_all=("household_id", "size")
    )
    mixed_g = mixed_g.join(ownership, on="zone_key")
    mixed_g["households_with_car_pct"] = _percent(mixed_g["hh_with_car"], mixed_g["hh_all"])
    mixed_g.loc[mixed_g["hh_all"] < 10, "households_with_car_pct"] = np.nan
    _map(mixed_g, "households_with_car_pct", "Households with at least one car • final residential",
         "40_car_owning_households.png", output_dir, "Households owning cars (%)", cmap="YlGnBu", limits=(0, 100))
    mixed_g.drop(columns="geometry").to_csv(output_dir / "residential_borough_and_district.csv", index=False)

    # Workplace geometry: observed one-day work activities, vs 2011 census job distribution.
    _map(mixed_g, "work_activity_density", "Work activity locations • Asunción boroughs / Central districts",
         "24_work_activity_density_mixed.png", output_dir, "Persons with work activity per km² (expanded)", cmap="YlOrRd",
         note="One location per person with a work activity; excludes workers not at work on model day")
    _map(mixed_g, "work_activity_density", "Asunción borough work-activity concentration",
         "25_asuncion_work_density.png", output_dir, "Persons with work activity per km² (expanded)", cmap="YlOrRd", subset="asuncion")
    _map(d, "work_activity_density", "Where people work • district work-activity density",
         "26_work_activity_density_districts.png", output_dir, "Observed persons at work per km², expanded", cmap="YlOrRd")
    _map(d, "work_location_share_gap_pp", "Spatial share of work • final day − 2011 workplace census",
         "27_workplace_share_difference.png", output_dir, "Percentage-point difference", diverging=True,
         note="Exploratory comparison only: one-day work activities vs 2011 job stock")
    _map(d, "work_vs_resident_employed_ratio", "Work location vs residential employment by district",
         "28_work_to_resident_employment.png", output_dir, "Work activity persons / employed residents", cmap="YlOrBr",
         note="Not a true jobs-to-resident-workers ratio: model-day attendance affects numerator")
    _workplace_plot(d, output_dir)
    _od_heatmap(flows, districts, output_dir)
    print(f"Spatial population analysis: plots and tables saved to {output_dir}")
