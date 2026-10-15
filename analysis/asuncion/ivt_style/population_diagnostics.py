"""EQASIM / Synpp stage: detailed population diagnostics.

Expected dependencies are the same as in the existing Asunción analysis stage.
Outputs: PNG figures, a browsable index.html, CSV diagnostic tables, and a manifest.
No seaborn/plotly dependency; uses only numpy, pandas, matplotlib.
"""

from pathlib import Path
from html import escape
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
COLORS = {"Census": "#364F6B", "IPU": "#008B8B", "HTS": "#F2A541"}
AGE_MAX = 115


def configure(context):
    context.config("analysis_path")
    context.config("diagnostics_age_bin", 5)
    context.config("diagnostics_min_effective_n", 30)
    for stage in (
        "asuncion.data.census.population", "asuncion.ipu.attributed",
        "asuncion.data.hts.entd.filtered", "asuncion.data.census.households",
        "asuncion.data.census.employment", "asuncion.data.census.income",
    ):
        context.stage(stage)


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B3C0C9", "axes.labelcolor": "#243447",
        "text.color": "#243447", "xtick.color": "#4A5968",
        "ytick.color": "#4A5968", "axes.titleweight": "bold",
        "axes.titlesize": 13, "font.size": 10,
        "grid.color": "#DCE4E9", "grid.linestyle": "--",
        "grid.alpha": 0.7, "savefig.dpi": 165,
    })


def _numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _binary(s):
    """Preserve missing/unknown instead of interpreting it as False."""
    lookup = {
        "true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
        "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0, "y": 1.0, "n": 0.0,
        "employed": 1.0, "unemployed": 0.0,
    }
    return s.astype("string").str.strip().str.lower().map(lookup).astype(float)


def _weights(df, column, source, qa):
    if column is None:
        return pd.Series(1.0, index=df.index)
    if column not in df:
        raise ValueError(f"{source}: required weight column '{column}' is missing")
    w = _numeric(df[column])
    invalid = (~np.isfinite(w.fillna(np.nan))) | (w < 0)
    if invalid.any():
        qa.append((source, f"Invalid {column} weights", int(invalid.sum()),
                   "Missing, infinite, or negative weights set to zero"))
    return w.where(np.isfinite(w) & (w >= 0), 0.0).astype(float)


def _people(df, source, weight_column, qa):
    p = df.copy()
    p["_w"] = _weights(p, weight_column, source, qa)
    p.attrs["original_columns"] = set(df.columns)
    for col in ("age", "sex", "employed", "has_license", "household_id", "income_class", "person_id"):
        if col not in p:
            p[col] = np.nan
    p["age"] = _numeric(p["age"])
    for col in ("employed", "has_license"):
        p[col] = _binary(p[col])
    if p["person_id"].notna().any():
        duplicates = int(p.loc[p["person_id"].notna(), "person_id"].duplicated().sum())
        if duplicates:
            qa.append((source, "Duplicate person_id", duplicates, "Investigate duplicated persons"))
    bad_age = p["age"].notna() & (~p["age"].between(0, AGE_MAX))
    if bad_age.any():
        qa.append((source, "Age outside 0–115", int(bad_age.sum()), "Excluded from age-based plots"))
    p.loc[bad_age, "age"] = np.nan
    return p


def _add_age(p, width):
    p = p.copy()
    p["age_start"] = np.floor(p["age"] / width) * width
    # All records age >= 100 enter open-ended 100+ group.
    p.loc[p["age_start"] >= 100, "age_start"] = 100
    return p


def _age_name(n, width):
    return "100+" if n >= 100 else f"{int(n)}–{int(n + width - 1)}"


def _weighted_categories(df, column, categories=None):
    if column not in df:
        return pd.Series(dtype=float)
    data = df.loc[df[column].notna() & (df["_w"] > 0)]
    v = data.groupby(column, observed=True)["_w"].sum()
    if categories is not None:
        v = v.reindex(categories, fill_value=0.0)
    return v


def _percent(s):
    return 100 * s / s.sum() if s.sum() > 0 else s * np.nan


def _rate_by_age(df, field, width, source, min_age=0):
    if field not in df or "age_start" not in df:
        return pd.DataFrame()
    d = df.loc[(df["age"].notna()) & (df["age"] >= min_age)
               & (df[field].isin([0, 1])) & (df["_w"] > 0)].copy()
    if d.empty:
        return pd.DataFrame()
    d["_positive"] = d[field] * d["_w"]
    d["_w2"] = d["_w"] ** 2
    result = d.groupby("age_start", observed=True).agg(
        weighted_yes=("_positive", "sum"), weighted_total=("_w", "sum"),
        weight_sq=("_w2", "sum"), n=(field, "size"),
    ).reset_index()
    result["rate"] = 100 * result["weighted_yes"] / result["weighted_total"]
    result["n_eff"] = result["weighted_total"] ** 2 / result["weight_sq"]
    # Kish effective N: approximate, not design-based survey confidence limits.
    result["se_pp"] = 100 * np.sqrt(
        (result["rate"] / 100) * (1 - result["rate"] / 100) / result["n_eff"])
    result["lo"] = (result["rate"] - 1.96 * result["se_pp"]).clip(lower=0)
    result["hi"] = (result["rate"] + 1.96 * result["se_pp"]).clip(upper=100)
    result["source"] = source
    result["metric"] = field
    result["age_label"] = result["age_start"].map(lambda x: _age_name(x, width))
    return result


def _as_sex(s):
    """Keep unrecognized sex codes distinct rather than guessing what 0/1 means."""
    mapping = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    raw = s.astype("string").str.strip()
    standard = raw.str.lower().map(mapping)
    return standard.fillna(raw.map(lambda x: f"Code: {x}" if pd.notna(x) and x else pd.NA))


def _sex_levels(data):
    labels = []
    for p in data.values():
        vc = _as_sex(p["sex"]).value_counts().head(4)
        labels.extend(vc.index.tolist())
    result = sorted(set(labels), key=lambda x: (x not in ["Female", "Male"], x))
    return result[:4]


def _save(fig, outdir, catalog, section, slug, title, note):
    name = f"{slug}.png"
    fig.savefig(outdir / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    catalog.append({"section": section, "filename": name, "title": title, "note": note})


def _axes(ax, ylabel=None, grid=True):
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(axis="y")
        ax.set_axisbelow(True)


def _empty(ax, message="Not enough comparable data"):
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, color="#7C8894")
    ax.set_xticks([])
    ax.set_yticks([])


def _category_plot(series_map, title, ylabel, outdir, catalog, section, slug,
                   note, max_categories=24, preferred=None):
    available = {k: v for k, v in series_map.items() if len(v) and v.sum() > 0}
    if len(available) < 2:
        return
    cats = list(dict.fromkeys(c for s in available.values() for c in s.index))
    if preferred is not None:
        cats = [c for c in preferred if c in cats] + [c for c in cats if c not in preferred]
    else:
        try:
            cats = sorted(cats)
        except TypeError:
            cats = sorted(cats, key=lambda z: str(z))
    cats = cats[:max_categories]
    fig, ax = plt.subplots(figsize=(max(9.5, len(cats) * .53), 5.2))
    x = np.arange(len(cats))
    bar_w = .82 / len(available)
    for i, (source, series) in enumerate(available.items()):
        values = _percent(series).reindex(cats, fill_value=0)
        ax.bar(x - .41 + (i + .5) * bar_w, values, width=bar_w,
               label=source, color=COLORS[source], alpha=.91)
    ax.set_xticks(x, [str(c) for c in cats], rotation=35 if len(cats) > 7 else 0,
                  ha="right" if len(cats) > 7 else "center")
    ax.set_title(title, loc="left", pad=15)
    _axes(ax, ylabel)
    ax.legend(frameon=False, ncol=3)
    _save(fig, outdir, catalog, section, slug, title, note)


def _plot_age_mix(census, ipu, hts, outdir, catalog, tables, qa):
    age_sources = {}
    raw = {}
    if "age_class" in census and "weight" in census:
        c = census.copy()
        c["age_start"] = np.floor(_numeric(c["age_class"]) / 5) * 5
        c.loc[c["age_start"] >= 100, "age_start"] = 100
        c["_w"] = _weights(c, "weight", "Census population", qa)
        age_sources["Census"] = _weighted_categories(c, "age_start")
    for label, p in (("IPU", ipu), ("HTS", hts)):
        age_sources[label] = _weighted_categories(_add_age(p, 5), "age_start")
    cats = sorted(set().union(*(s.index for s in age_sources.values())))
    if not cats:
        return
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(len(cats))
    for label in ("Census", "IPU", "HTS"):
        if label not in age_sources:
            continue
        s = _percent(age_sources[label]).reindex(cats, fill_value=0)
        ax.plot(x, s.values, lw=2.25, marker="o", markersize=4.5,
                color=COLORS[label], label=label)
        for c, val in zip(cats, s):
            raw[(label, c)] = float(val)
    ax.set_title("Age distribution · 5-year groups", loc="left", pad=15)
    ax.set_xticks(x, [_age_name(a, 5) for a in cats], rotation=45, ha="right")
    ax.legend(frameon=False, ncol=3)
    _axes(ax, "Share of persons (%)")
    _save(fig, outdir, catalog, "Population", "01_age_distribution",
          "Age distribution · 5-year groups", "Census and HTS use weights; IPU counts persons.")
    df = pd.DataFrame([{"source": src, "age_start": a, "share_pct": v}
                       for (src, a), v in raw.items()])
    tables["age_distribution.csv"] = df
    if "Census" in age_sources:
        s1 = _percent(age_sources["IPU"]).reindex(cats, fill_value=0)
        s0 = _percent(age_sources["Census"]).reindex(cats, fill_value=0)
        diff = (s1 - s0)
        fig, ax = plt.subplots(figsize=(12.5, 4.4))
        bars = ax.bar(np.arange(len(cats)), diff, color=np.where(diff >= 0, "#C87954", "#4B9D96"))
        ax.axhline(0, color="#566574", linewidth=1.2)
        ax.set_xticks(np.arange(len(cats)), [_age_name(c, 5) for c in cats],
                      rotation=45, ha="right")
        ax.set_title("IPU minus census · age-share difference", loc="left", pad=15)
        _axes(ax, "Difference (percentage points)")
        _save(fig, outdir, catalog, "Population", "02_age_share_delta",
              "Age-share deviations from census", "Positive bars mean overrepresentation in synthetic population.")
        tables["age_share_difference.csv"] = pd.DataFrame({
            "age_start": cats, "ipu_minus_census_pp": diff.to_numpy()})


def _plot_sex(ipu, hts, census, outdir, catalog, qa):
    src = {}
    for label, p in (("IPU", ipu), ("HTS", hts)):
        if p["sex"].notna().any():
            src[label] = _weighted_categories(p, "sex")
    if "sex" in census and "weight" in census:
        c = census.copy()
        c["_w"] = _weights(c, "weight", "Census population", qa)
        src["Census"] = _weighted_categories(c, "sex")
    _category_plot(src, "Sex distribution", "Share of persons (%)", outdir,
                   catalog, "Population", "03_sex_distribution",
                   "Compare sex codes carefully; numeric source encodings are not automatically recoded.")


def _plot_pyramid(data, outdir, catalog):
    # Only construct a male/female pyramid when sex codes are semantically known.
    prepared = {}
    for label, p in data.items():
        d = _add_age(p, 5)
        d["sex_named"] = _as_sex(d["sex"])
        if {"Male", "Female"}.issubset(set(d.sex_named.dropna().unique())):
            group = d.loc[d.age_start.notna() & d.sex_named.isin(["Male", "Female"])]
            by = group.pivot_table(index="age_start", columns="sex_named", values="_w",
                                   aggfunc="sum", fill_value=0)
            # Each sex/age bin's share is divided by TOTAL source population.
            if group._w.sum() > 0:
                prepared[label] = by / p._w.sum() * 100
    if not prepared:
        return
    all_ages = sorted(set().union(*(t.index for t in prepared.values())))
    fig, axes = plt.subplots(1, len(prepared), figsize=(6.4 * len(prepared), 6.4),
                             sharey=True, squeeze=False, layout="constrained")
    xmax = max(float(t.max().max()) for t in prepared.values()) * 1.22
    for ax, (source, df) in zip(axes[0], prepared.items()):
        df = df.reindex(all_ages, fill_value=0).reindex(columns=["Male", "Female"], fill_value=0)
        y = np.arange(len(all_ages))
        ax.barh(y, -df["Male"], height=.78, color="#4D819B", label="Male")
        ax.barh(y, df["Female"], height=.78, color="#DB8C86", label="Female")
        ax.set_xlim(-xmax, xmax)
        ax.set_xticks(np.linspace(-xmax, xmax, 7))
        ax.set_xticklabels([f"{abs(v):.1f}%" for v in np.linspace(-xmax, xmax, 7)])
        ax.set_title(source)
        ax.set_xlabel("Share of all persons (%)")
        ax.set_yticks(y, [_age_name(a, 5) for a in all_ages])
        ax.axvline(0, color="#8595A2", lw=1)
        ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Population pyramid · age × sex", x=.02, ha="left", weight="bold")
    _save(fig, outdir, catalog, "Population", "03b_population_pyramid",
          "Population pyramid · IPU vs HTS",
          "Male shown left, female right. Each bar is a share of total persons within its source.")


def _census_employment_rates(population, employment, age_width, qa):
    if not ({"age_class", "weight"} <= set(population)) or not ({"age_class", "weight"} <= set(employment)):
        return pd.DataFrame()
    p, e = population.copy(), employment.copy()
    p["age_start"] = np.floor(_numeric(p["age_class"]) / age_width) * age_width
    e["age_start"] = np.floor(_numeric(e["age_class"]) / age_width) * age_width
    p.loc[p["age_start"] >= 100, "age_start"] = 100
    e.loc[e["age_start"] >= 100, "age_start"] = 100
    p["_w"] = _weights(p, "weight", "Census population", qa)
    e["_w"] = _weights(e, "weight", "Census employment", qa)
    all_people = _weighted_categories(p, "age_start")
    employed = _weighted_categories(e, "age_start")
    combined = pd.DataFrame({"all": all_people, "yes": employed}).dropna(subset=["all"])
    combined = combined.loc[combined["all"] > 0]
    if combined.empty:
        return pd.DataFrame()
    combined["rate"] = 100 * combined["yes"].fillna(0) / combined["all"]
    if (combined["rate"] > 100.001).any():
        qa.append(("Census", "Employment age rate >100%",
                   int((combined["rate"] > 100.001).sum()),
                   "Population and employment denominators may not be compatible"))
        return pd.DataFrame()
    result = combined.reset_index()[["age_start", "rate"]]
    result["source"] = "Census"
    result["metric"] = "employed"
    return result


def _plot_rate(data, metric, title, min_age, width, min_eff, outdir,
               catalog, tables, census_rates=None, name=""):
    stats = [_rate_by_age(p, metric, width, source, min_age) for source, p in data.items()]
    stats = [d for d in stats if not d.empty]
    if census_rates is not None and not census_rates.empty:
        stats.append(census_rates)
    if not stats:
        return
    allstats = pd.concat(stats, ignore_index=True)
    tables[f"{name}_age_rates.csv"] = allstats
    # Only plot age groups observed in at least one source.
    ages = sorted(allstats.age_start.unique())
    fig, (ax, ax_diff) = plt.subplots(
        2, 1, figsize=(12.4, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1], "hspace": .12})
    xcoords = {a: i for i, a in enumerate(ages)}
    for source in ("Census", "IPU", "HTS"):
        d = allstats.loc[allstats.source.eq(source)].sort_values("age_start")
        if d.empty:
            continue
        xs = d.age_start.map(xcoords).to_numpy(dtype=float)
        ax.plot(xs, d.rate, "o-", label=source, color=COLORS[source], lw=2.2,
                markersize=5, zorder=3)
        if source == "HTS" and "lo" in d and "hi" in d:
            valid = d["n_eff"].ge(min_eff)
            # Intervals shown only for age groups meeting the reporting threshold.
            ax.vlines(xs[valid], d.loc[valid, "lo"], d.loc[valid, "hi"],
                      color=COLORS["HTS"], alpha=.6, lw=2.3, zorder=2)
            low = d.loc[~valid]
            if len(low):
                ax.scatter(low.age_start.map(xcoords), low.rate, marker="x",
                           color="#AF3D4E", s=58, zorder=6,
                           label=f"HTS low effective N (<{min_eff})")
    ax.set_ylim(-3, 103)
    ax.set_title(title, loc="left", pad=12)
    ax.set_ylabel("Within-age-group share (%)")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.set_axisbelow(True)
    ipu = allstats[allstats.source.eq("IPU")].set_index("age_start")
    hts = allstats[allstats.source.eq("HTS")].set_index("age_start")
    shared = sorted(set(ipu.index) & set(hts.index))
    if shared:
        diff = ipu.loc[shared, "rate"] - hts.loc[shared, "rate"]
        enough = hts.loc[shared, "n_eff"] >= min_eff
        colors = ["#C87954" if v > 0 else "#4B9D96" for v in diff]
        colors = [col if ok else "#BCC4CC" for col, ok in zip(colors, enough)]
        ax_diff.bar([xcoords[a] for a in shared], diff.values, color=colors, width=.65)
        ax_diff.axhline(0, color="#566574", lw=1)
        ax_diff.set_ylabel("IPU − HTS (pp)")
        ax_diff.grid(axis="y")
        ax_diff.set_axisbelow(True)
        delta = pd.DataFrame({"age_start": shared, "ipu_minus_hts_pp": diff.to_numpy(),
                              "hts_n_eff": hts.loc[shared, "n_eff"].to_numpy(),
                              "adequate_hts_n_eff": enough.to_numpy()})
        tables[f"{name}_age_differences.csv"] = delta
    else:
        _empty(ax_diff, "No shared age groups")
    ax_diff.set_xticks(np.arange(len(ages)), [_age_name(a, width) for a in ages],
                       rotation=30, ha="right")
    ax_diff.set_xlabel("Age group")
    _save(fig, outdir, catalog, "Age-specific outcomes", name, title,
          f"Rates exclude missing outcomes; HTS whiskers approximate 95% intervals using Kish N. "
          f"Grey differences and red × mark HTS cells below N_eff={min_eff}.")


def _plot_outcome_by_sex(data, metric, width, min_eff, outdir, catalog, slug, title):
    levels = _sex_levels(data)
    if not levels:
        return
    fig, axes = plt.subplots(1, len(levels), figsize=(max(7, 6.1 * len(levels)), 4.5),
                             sharey=True, layout="constrained", squeeze=False)
    any_plotted = False
    for ax, sex in zip(axes[0], levels):
        for label, p in data.items():
            mask = _as_sex(p["sex"]).eq(sex).fillna(False)
            stats = _rate_by_age(p.loc[mask], metric, width, label)
            if stats.empty:
                continue
            if label == "HTS":
                stats.loc[stats.n_eff < min_eff, "rate"] = np.nan
            ax.plot(stats.age_start, stats.rate, "o-", lw=2,
                    color=COLORS[label], label=label)
            any_plotted = True
        ax.set_title(str(sex))
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Age-group start (years)")
        _axes(ax, "Within-group share (%)" if ax is axes[0][0] else None)
        ax.legend(frameon=False)
    if any_plotted:
        fig.suptitle(title, weight="bold", x=.02, ha="left")
        _save(fig, outdir, catalog, "Age-specific outcomes", slug, title,
              f"HTS points suppressed where effective N < {min_eff}; unknown sex codes retain their original labels.")
    else:
        plt.close(fig)


def _plot_joint_state(data, width, outdir, catalog):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), sharey=True, layout="constrained")
    any_plotted = False
    for ax, (label, p) in zip(axes, data.items()):
        d = p.loc[p["age_start"].notna() & p["employed"].isin([0, 1])
                  & p["has_license"].isin([0, 1]) & (p["_w"] > 0)].copy()
        if d.empty:
            _empty(ax)
            continue
        d["status"] = (d["employed"].astype(int).astype(str) +
                       d["has_license"].astype(int).astype(str))
        tbl = d.pivot_table(index="age_start", columns="status", values="_w",
                            aggfunc="sum", fill_value=0)
        tbl = tbl.reindex(columns=["00", "01", "10", "11"], fill_value=0).sort_index()
        tbl = tbl.div(tbl.sum(axis=1), axis=0) * 100
        labels = ["Neither", "Licence only", "Employed only", "Both"]
        colors = ["#B9C4D0", "#86AACC", "#F2A541", "#008B8B"]
        x = np.arange(len(tbl))
        bottom = np.zeros(len(tbl))
        for key, lab, color in zip(tbl.columns, labels, colors):
            ax.bar(x, tbl[key], bottom=bottom, color=color, label=lab, width=.83)
            bottom += tbl[key].to_numpy()
        ax.set_title(label)
        ax.set_xticks(x, [_age_name(a, width) for a in tbl.index], rotation=45, ha="right")
        ax.set_xlabel("Age group")
        ax.set_ylim(0, 100)
        any_plotted = True
    if any_plotted:
        axes[0].set_ylabel("Within-age-group share (%)")
        axes[-1].legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, -0.45), ncol=2)
        fig.suptitle("Employment × driving licence by age", x=.03, ha="left", weight="bold")
        _save(fig, outdir, catalog, "Interactions", "08_employment_license_joint",
              "Employment and licence status · joint distribution",
              "Each bar sums to 100% within its age group; records missing either outcome are excluded.")
    else:
        plt.close(fig)


def _plot_sex_age_heatmaps(data, outcome, width, min_eff, outdir, catalog, slug, title):
    levels = _sex_levels(data)
    if not levels:
        return
    frames = []
    for label, p in data.items():
        sexes = _as_sex(p["sex"])
        for sex in levels:
            stats = _rate_by_age(p.loc[sexes.eq(sex).fillna(False)], outcome, width, label)
            if stats.empty:
                continue
            stats["sex"] = sex
            if label == "HTS":
                stats.loc[stats.n_eff < min_eff, "rate"] = np.nan
            frames.append(stats)
    if not frames:
        return
    frames = pd.concat(frames, ignore_index=True)
    ages = sorted(frames.age_start.unique())
    names = ["IPU", "HTS", "IPU − HTS"]
    maps = {}
    for source in ("IPU", "HTS"):
        d = frames[frames.source.eq(source)]
        maps[source] = d.pivot_table(index="sex", columns="age_start", values="rate",
                                     aggfunc="mean").reindex(index=levels, columns=ages)
    maps["IPU − HTS"] = maps["IPU"] - maps["HTS"]
    if maps["IPU"].isna().all().all() and maps["HTS"].isna().all().all():
        return
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(ages) * .68),
                                        6.5 + len(levels) * 1.3), layout="constrained")
    for ax, name in zip(axes, names):
        matrix = np.ma.masked_invalid(maps[name].to_numpy(dtype=float))
        if name == "IPU − HTS":
            valid = np.abs(matrix.compressed())
            scale = max(5, float(valid.max()) if len(valid) else 5)
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("#E7EBEF")
            im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
        else:
            cmap = plt.get_cmap("YlGnBu").copy()
            cmap.set_bad("#E7EBEF")
            im = ax.imshow(matrix, vmin=0, vmax=100, aspect="auto", cmap=cmap)
        ax.set_yticks(np.arange(len(levels)), levels)
        ax.set_xticks(range(len(ages)), [_age_name(a, width) for a in ages],
                      rotation=45, ha="right")
        ax.set_title(name, loc="left")
        for i in range(len(levels)):
            for j in range(len(ages)):
                val = maps[name].iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.0f}" if name == "IPU − HTS" else f"{val:.0f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if (name != "IPU − HTS" and val >= 57) else "#1E2933")
        fig.colorbar(im, ax=ax, shrink=.65,
                     label="Difference (pp)" if name == "IPU − HTS" else "Rate (%)")
    fig.suptitle(title, weight="bold", x=.03, ha="left")
    _save(fig, outdir, catalog, "Interactions", slug, title,
          f"Grey cells = unavailable or HTS effective N < {min_eff}; original sex codes are not guessed.")


def _households(ipu, hts_hh, qa):
    # IPU is one row per person. Household characteristics must be counted once.
    valid_id = ipu["household_id"].notna()
    if (~valid_id).any():
        qa.append(("IPU", "Persons missing household_id", int((~valid_id).sum()),
                   "Excluded from household comparisons"))
    d = ipu.loc[valid_id].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    grp = d.groupby("household_id", sort=False, observed=True)
    synthetic = pd.DataFrame({"household_size": grp.size()})
    synthetic["children"] = grp["age"].apply(lambda s: (s < 18).sum())
    if "income_class" in d:
        non_null = grp["income_class"].nunique(dropna=True)
        bad_income = (non_null > 1).sum()
        if bad_income:
            qa.append(("IPU", "Conflicting household income class", int(bad_income),
                       "Multiple income classes within the same household"))
        synthetic["income_class"] = grp["income_class"].first()
    synthetic["_w"] = 1.0
    hh = hts_hh.copy()
    hh["_w"] = _weights(hh, "household_weight", "HTS households", qa)
    if "household_id" in hh and hh["household_id"].notna().any():
        dup = int(hh.loc[hh["household_id"].notna(), "household_id"].duplicated().sum())
        if dup:
            qa.append(("HTS", "Duplicate household_id in household table", dup,
                       "First household row kept for household-weighted plots"))
            hh = hh.drop_duplicates(subset="household_id", keep="first")
    if "household_size" in hh:
        hh["household_size"] = _numeric(hh["household_size"])
        bad_size = hh["household_size"].notna() & (hh["household_size"] < 1)
        if bad_size.any():
            qa.append(("HTS", "Invalid household size <1", int(bad_size.sum()),
                       "Excluded from household-size plots"))
            hh.loc[bad_size, "household_size"] = np.nan
    return synthetic, hh


def _plot_hh_size(ipu_hh, hts_hh, census_hh, outdir, catalog, tables, qa):
    src = {}
    for label, d in (("IPU", ipu_hh), ("HTS", hts_hh)):
        if "household_size" in d:
            x = d.copy()
            x["size"] = _numeric(x.household_size).clip(upper=6)
            src[label] = _weighted_categories(x, "size")
    if "household_size" in census_hh and "weight" in census_hh:
        c = census_hh.copy()
        c["_w"] = _weights(c, "weight", "Census households", qa)
        c["size"] = _numeric(c.household_size).clip(upper=6)
        src["Census"] = _weighted_categories(c, "size")
    src = {name: v.rename(index={float(6): "6+"}).rename(index={float(i): str(i) for i in range(1, 6)})
           for name, v in src.items()}
    _category_plot(src, "Household size", "Share of households (%)", outdir,
                   catalog, "Households", "12_household_size", "Households, not persons; 6+ is an open group.",
                   preferred=["1", "2", "3", "4", "5", "6+"])
    tables["household_size.csv"] = pd.DataFrame([
        {"source": label, "size": str(size), "share_pct": float(value)}
        for label, series in src.items() for size, value in _percent(series).items()
    ])


def _plot_hh_composition(ipu_hh, outdir, catalog):
    if ipu_hh.empty:
        return
    df = ipu_hh.copy()
    df["size"] = df.household_size.clip(upper=6)
    df["children_group"] = df.children.clip(upper=4)
    mat = pd.crosstab(df["size"], df["children_group"], normalize="index") * 100
    mat = mat.reindex(index=range(1, 7), columns=range(5), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    im = ax.imshow(mat.to_numpy(), vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(5), ["0", "1", "2", "3", "4+"])
    ax.set_yticks(range(6), ["1", "2", "3", "4", "5", "6+"])
    ax.set_xlabel("Number of children (<18)")
    ax.set_ylabel("Household size")
    ax.set_title("IPU household composition", loc="left", pad=15)
    for i in range(6):
        for j in range(5):
            v = mat.iloc[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else "#253443")
    fig.colorbar(im, ax=ax, label="Share within household size (%)")
    _save(fig, outdir, catalog, "Households", "13_household_children_heatmap",
          "Number of children by household size", "IPU only: each row sums to 100%. Blank-looking categories have 0%.")


def _plot_income(ipu_hh, hts_hh, census_inc, outdir, catalog, tables, qa):
    sources = {}
    for label, df in (("IPU", ipu_hh), ("HTS", hts_hh)):
        if "income_class" in df:
            sources[label] = _weighted_categories(df, "income_class")
    _category_plot(sources, "Household income-class distribution",
                   "Share of households (%)", outdir, catalog, "Households",
                   "14_household_income", "Household-level weights; excludes unknown income classes.")
    if sources:
        tables["household_income.csv"] = pd.DataFrame([
            {"source": label, "income_class": str(cat), "share_pct": float(val)}
            for label, s in sources.items() for cat, val in _percent(s).items()
        ])
    if {"income_class", "weight"} <= set(census_inc.columns):
        c = census_inc.copy()
        c["_w"] = _weights(c, "weight", "Census income", qa)
        reference = _percent(_weighted_categories(c, "income_class"))
        tables["census_income_reference.csv"] = pd.DataFrame({
            "income_class": reference.index.astype(str), "reference_share_pct": reference.values})
        qa.append(("Census", "Income reference grain unverified", 1,
                   "Census income saved separately; verify person-vs-household units before comparison"))


def _plot_income_by_hh_size(ipu_hh, hts_hh, outdir, catalog):
    """Show household income-class composition within each household size."""
    frames = {}
    for label, df in (("IPU", ipu_hh), ("HTS", hts_hh)):
        if not {"income_class", "household_size"} <= set(df.columns):
            continue
        d = df.loc[df.income_class.notna() & df.household_size.notna()
                   & (df._w > 0)].copy()
        if d.empty:
            continue
        d["size"] = _numeric(d.household_size).clip(upper=6)
        d = d.loc[d["size"].between(1, 6)]
        if not d.empty:
            frames[label] = d
    if not frames:
        return
    all_classes = list(dict.fromkeys(c for d in frames.values() for c in d.income_class.unique()))
    if len(all_classes) > 12:
        # A very high-cardinality income label often means a continuous variable, not a class.
        LOG.warning("Skipping income-size heatmap: %s distinct income classes", len(all_classes))
        return
    try:
        all_classes = sorted(all_classes)
    except TypeError:
        all_classes = sorted(all_classes, key=str)
    fig, axes = plt.subplots(1, len(frames),
                             figsize=(max(7, len(all_classes) * 1.2) * len(frames), 5.3),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, (source, df) in zip(axes[0], frames.items()):
        pivot = df.pivot_table(index="size", columns="income_class", values="_w",
                               aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(index=range(1, 7), columns=all_classes, fill_value=0)
        denom = pivot.sum(axis=1).replace(0, np.nan)
        pivot = pivot.div(denom, axis=0) * 100
        vals = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        cmap = plt.get_cmap("BuGn").copy()
        cmap.set_bad("#E7EBEF")
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0, vmax=100)
        ax.set_yticks(range(6), ["1", "2", "3", "4", "5", "6+"])
        ax.set_xticks(range(len(all_classes)), [str(c) for c in all_classes],
                      rotation=30 if len(all_classes) > 4 else 0, ha="right" if len(all_classes) > 4 else "center")
        ax.set_ylabel("Household size" if ax is axes[0][0] else "")
        ax.set_xlabel("Income class")
        ax.set_title(source)
        for i in range(6):
            for j in range(len(all_classes)):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            color="white" if val >= 55 else "#253443", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=.8, label="Share of households (%)")
    fig.suptitle("Household income class × household size", x=.02, ha="left", weight="bold")
    _save(fig, outdir, catalog, "Households", "11_household_income_by_size",
          "Income class by household size",
          "Each household-size row sums to 100%; grey rows have no comparable records.")


def _plot_missingness(data, outdir, catalog, tables):
    cols = ["age", "sex", "employed", "has_license", "household_id", "income_class"]
    names = list(data)
    rates = pd.DataFrame(index=names, columns=cols, dtype=float)
    # Missingness is computed across rows (not weights): complementary to weighted plots.
    for label, p in data.items():
        for col in cols:
            rates.loc[label, col] = (float(p[col].isna().mean() * 100) if len(p)
                                     else np.nan) if col in p.attrs.get("original_columns", set(p)) else np.nan
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    im = ax.imshow(np.ma.masked_invalid(rates.to_numpy(dtype=float)), cmap="OrRd",
                   vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xticks(np.arange(len(cols)), ["Age", "Sex", "Employment", "Driving licence",
                                               "Household ID", "Income class"])
    for i in range(len(names)):
        for j in range(len(cols)):
            v = rates.iloc[i, j]
            ax.text(j, i, "N/A" if not np.isfinite(v) else f"{v:.1f}%",
                    ha="center", va="center", color="white" if v >= 60 else "#253443")
    ax.set_title("Missing / unrecognized values by source", loc="left", pad=15)
    fig.colorbar(im, ax=ax, label="Missing rows (%)")
    _save(fig, outdir, catalog, "Quality checks", "15_missingness",
          "Attribute completeness", "Missing or unrecognized binary values count as missing; row-based diagnostic.")
    tables["missingness.csv"] = rates.rename_axis("source").reset_index()


def _plot_coverage(hts, width, min_eff, outdir, catalog, tables):
    d = hts.loc[hts.age_start.notna() & (hts._w > 0)].copy()
    if d.empty:
        return
    d["_w2"] = d._w ** 2
    cov = d.groupby("age_start").agg(n=("_w", "size"),
                                     total_weight=("_w", "sum"), weight_sq=("_w2", "sum"))
    cov["n_eff"] = cov.total_weight ** 2 / cov.weight_sq
    cov.to_csv(outdir / "hts_age_coverage.csv")
    x = np.arange(len(cov))
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    ax.bar(x - .2, cov.n, width=.4, color="#9CB0C2", label="Raw respondents")
    ax.bar(x + .2, cov.n_eff, width=.4, color=COLORS["HTS"], label="Kish effective N")
    ax.axhline(min_eff, color="#B44755", linestyle="--", lw=1.5,
               label=f"Minimum N_eff = {min_eff}")
    ax.set_xticks(x, [_age_name(a, width) for a in cov.index], rotation=35, ha="right")
    ax.set_title("HTS age-bin sample coverage", loc="left", pad=15)
    _axes(ax, "Sample size / effective sample size")
    ax.legend(frameon=False, ncol=3)
    _save(fig, outdir, catalog, "Quality checks", "16_hts_coverage",
          "HTS effective sample size", "Kish N is not a substitute for a survey-design-based variance estimate.")
    tables["hts_age_coverage.csv"] = cov.reset_index()


def _plot_top_gaps(tables, min_eff, outdir, catalog):
    records = []
    for file, metric in (("employment_age_differences.csv", "Employment"),
                         ("license_age_differences.csv", "Driving licence")):
        if file not in tables:
            continue
        d = tables[file]
        for row in d.itertuples():
            if row.adequate_hts_n_eff:
                records.append((metric, int(row.age_start), float(row.ipu_minus_hts_pp)))
    if not records:
        return
    biggest = sorted(records, key=lambda z: abs(z[2]), reverse=True)[:16][::-1]
    fig, ax = plt.subplots(figsize=(10, max(4.8, len(biggest) * .39)))
    labels = [f"{name} · age {age}+" for name, age, _ in biggest]
    diffs = [v for _, _, v in biggest]
    ax.barh(np.arange(len(biggest)), diffs,
            color=["#C87954" if x >= 0 else "#4B9D96" for x in diffs])
    ax.set_yticks(np.arange(len(biggest)), labels)
    ax.axvline(0, color="#4A5968", lw=1)
    ax.set_title("Largest IPU–HTS differences", loc="left", pad=15)
    ax.set_xlabel("Difference in within-age-group rate (percentage points)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    _save(fig, outdir, catalog, "Quality checks", "17_largest_gaps",
          "Largest employment / licence differences",
          f"Only bins with HTS effective N ≥ {min_eff}; descriptive differences, not a statistical test.")


def _quality(data, qa):
    for source, df in data.items():
        if df.empty:
            qa.append((source, "Empty person table", 1, "No rows"))
            continue
        counts = {
            "Employed persons younger than 15": int(((df.age < 15) & df.employed.eq(1)).sum()),
            "Licensed persons younger than 18": int(((df.age < 18) & df.has_license.eq(1)).sum()),
            "Missing age": int(df.age.isna().sum()),
            "Missing employment": int(df.employed.isna().sum()),
            "Missing driving licence": int(df.has_license.isna().sum()),
        }
        for metric, count in counts.items():
            qa.append((source, metric, count,
                       "Potential anomaly; confirm local legal/eligibility rules" if "younger" in metric
                       else "Missing and unrecognized values can bias rates"))


def _write_dashboard(outdir, catalog, tables, qa):
    qa_df = pd.DataFrame(qa, columns=["source", "check", "count", "interpretation"])
    qa_df.to_csv(outdir / "quality_checks.csv", index=False)
    for file, table in tables.items():
        table.to_csv(outdir / file, index=False)
    sections = list(dict.fromkeys(c["section"] for c in catalog))
    nav = "".join(f'<a href="#{escape(s.lower().replace(" ", "-"))}">{escape(s)}</a>'
                  for s in sections)
    cards = []
    for section in sections:
        cards.append(f'<section id="{escape(section.lower().replace(" ", "-"))}"><h2>{escape(section)}</h2><div class="grid">')
        for entry in catalog:
            if entry["section"] != section:
                continue
            f = escape(entry["filename"])
            cards.append(f'<article class="card"><a href="{f}" target="_blank" rel="noopener">'
                         f'<img src="{f}" loading="lazy" alt="{escape(entry["title"])}"></a>'
                         f'<div class="body"><h3>{escape(entry["title"])}</h3>'
                         f'<p>{escape(entry["note"])}</p>'
                         f'<a href="{f}" target="_blank" rel="noopener">Open full resolution ↗</a></div></article>')
        cards.append('</div></section>')
    links = " ".join(f'<a href="{escape(t)}">{escape(t)}</a>'
                     for t in sorted(["quality_checks.csv", *tables]))
    significant = qa_df.loc[qa_df["count"].gt(0)].head(16)
    qa_rows = "".join(
        f'<tr><td>{escape(str(r.source))}</td><td>{escape(str(r.check))}</td>'
        f'<td>{escape(str(r.count))}</td><td>{escape(str(r.interpretation))}</td></tr>'
        for r in significant.itertuples(index=False)) or '<tr><td colspan="4">No flagged checks.</td></tr>'
    html_doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EQASIM · Population diagnostics</title>
<style>
:root{{--ink:#203245;--muted:#607386;--line:#DDE6EC;--teal:#008B8B;--bg:#F5F8FB}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);font-family:Inter,system-ui,-apple-system,sans-serif;color:var(--ink)}}
header{{background:linear-gradient(120deg,#1D354D,#29656D);color:white;padding:40px max(5vw,22px)}}
header h1{{font-size:clamp(26px,4vw,41px);margin:0 0 12px}}header p{{margin:0;max-width:800px;color:#E3F0F1;line-height:1.6}}
nav{{position:sticky;top:0;background:#fff;z-index:5;box-shadow:0 2px 9px #19324817;padding:15px 5vw;display:flex;gap:22px;flex-wrap:wrap}}
a{{color:#086D77;text-decoration:none}}nav a{{font-weight:650}}a:hover{{text-decoration:underline}}
main{{max-width:1500px;margin:auto;padding:22px max(2vw,16px) 60px}}section{{scroll-margin-top:85px;margin:34px 0 50px}}section h2{{font-size:23px;margin:15px 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:22px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 5px 19px #28455C10}}
.card img{{width:100%;aspect-ratio:1.85;object-fit:contain;padding:9px;background:#fff;display:block}}
.body{{padding:18px 22px 22px;border-top:1px solid var(--line)}}.body h3{{margin:0 0 8px;font-size:17px}}.body p{{color:var(--muted);line-height:1.5;font-size:13px;min-height:32px}}
.data{{background:white;border-radius:14px;padding:24px;overflow:auto;border:1px solid var(--line)}}.links a{{display:inline-block;padding:5px 10px;background:#E9F2F4;border-radius:7px;margin:4px;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left}}th{{background:#EEF4F7}}
</style></head><body><header><h1>EQASIM population diagnostics</h1>
<p>Census · IPU synthetic population · HTS. All charts are available as high-resolution PNGs. Age-specific rates are computed within the relevant age group; HTS is survey-weighted.</p></header>
<nav>{nav}<a href="#quality-table">Quality table</a><a href="#csv">CSV data</a></nav><main>
{''.join(cards)}
<section id="quality-table"><h2>Quality checks (non-zero counts)</h2><div class="data"><table><thead><tr><th>Source</th><th>Check</th><th>Count</th><th>Interpretation</th></tr></thead><tbody>{qa_rows}</tbody></table><p>Download the complete list from quality_checks.csv. This is a descriptive audit, not an automatic pass/fail test.</p></div></section>
<section id="csv"><h2>Download diagnostic data</h2><div class="data links">{links}</div></section>
</main></body></html>'''
    (outdir / "index.html").write_text(html_doc, encoding="utf-8")
    pd.DataFrame(catalog).to_csv(outdir / "plot_manifest.csv", index=False)
    return qa_df


def execute(context):
    """Create the report in analysis_path/population_diagnostics/ and return its manifest."""
    _style()
    outdir = Path(context.config("analysis_path")) / "population_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    width = int(context.config("diagnostics_age_bin"))
    min_eff = int(context.config("diagnostics_min_effective_n"))
    if width not in (5, 10, 20) or 100 % width:
        raise ValueError("diagnostics_age_bin must be 5, 10, or 20")
    if min_eff < 1:
        raise ValueError("diagnostics_min_effective_n must be >= 1")

    census_pop = context.stage("asuncion.data.census.population").copy()
    census_emp = context.stage("asuncion.data.census.employment").copy()
    census_hh = context.stage("asuncion.data.census.households").copy()
    census_inc = context.stage("asuncion.data.census.income").copy()
    ipu_raw = context.stage("asuncion.ipu.attributed")
    hts_hh_raw, hts_p_raw, _ = context.stage("asuncion.data.hts.entd.filtered")

    qa, tables, catalog = [], {}, []
    ipu = _add_age(_people(ipu_raw, "IPU", None, qa), width)
    hts = _add_age(_people(hts_p_raw, "HTS", "person_weight", qa), width)
    people = {"IPU": ipu, "HTS": hts}
    _quality(people, qa)

    _plot_age_mix(census_pop, ipu, hts, outdir, catalog, tables, qa)
    _plot_sex(ipu, hts, census_pop, outdir, catalog, qa)
    _plot_pyramid(people, outdir, catalog)
    census_rates = _census_employment_rates(census_pop, census_emp, width, qa)
    _plot_rate(people, "employed", "Employment rate by age", 0, width, min_eff,
               outdir, catalog, tables, census_rates, "04_employment_by_age")
    # Keep predictable CSV names for QA ranking.
    for old, new in (("04_employment_by_age_age_rates.csv", "employment_age_rates.csv"),
                     ("04_employment_by_age_age_differences.csv", "employment_age_differences.csv")):
        if old in tables:
            tables[new] = tables.pop(old)
    _plot_rate(people, "has_license", "Driving licence ownership by age", 0, width,
               min_eff, outdir, catalog, tables, None, "05_license_by_age")
    for old, new in (("05_license_by_age_age_rates.csv", "license_age_rates.csv"),
                     ("05_license_by_age_age_differences.csv", "license_age_differences.csv")):
        if old in tables:
            tables[new] = tables.pop(old)
    _plot_outcome_by_sex(people, "employed", width, min_eff, outdir, catalog,
                         "06_employment_by_sex_age", "Employment rate by age and sex")
    _plot_outcome_by_sex(people, "has_license", width, min_eff, outdir, catalog,
                         "07_license_by_sex_age", "Driving licence ownership by age and sex")
    _plot_joint_state(people, width, outdir, catalog)
    _plot_sex_age_heatmaps(people, "employed", width, min_eff, outdir, catalog,
                           "09_employment_heatmap", "Employment rate · age × sex heatmaps")
    _plot_sex_age_heatmaps(people, "has_license", width, min_eff, outdir, catalog,
                           "10_license_heatmap", "Driving licence rate · age × sex heatmaps")

    ipu_hh, hts_hh = _households(ipu, hts_hh_raw, qa)
    _plot_income_by_hh_size(ipu_hh, hts_hh, outdir, catalog)
    _plot_hh_size(ipu_hh, hts_hh, census_hh, outdir, catalog, tables, qa)
    _plot_hh_composition(ipu_hh, outdir, catalog)
    _plot_income(ipu_hh, hts_hh, census_inc, outdir, catalog, tables, qa)
    _plot_missingness(people, outdir, catalog, tables)
    _plot_coverage(hts, width, min_eff, outdir, catalog, tables)
    _plot_top_gaps(tables, min_eff, outdir, catalog)

    qa_df = _write_dashboard(outdir, catalog, tables, qa)
    LOG.info("Population diagnostics: %s charts saved to %s", len(catalog), outdir)
    return {"output_directory": str(outdir), "dashboard": str(outdir / "index.html"),
            "plots": [x["filename"] for x in catalog], "quality_checks": len(qa_df)}
