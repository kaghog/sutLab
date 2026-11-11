"""Common, Seville-specific helpers for population and spatial diagnostics.

No SynPP imports: the validation calculations can be exercised on data frames.
Census sections are 10-digit IDs; Seville districts 41091xx are seven digits,
while surrounding municipalities have five-digit OD macrozone codes.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

SCOPE_CITY = "41091"
COLORS = {"Census": "#358A78", "Final": "#D08332", "IPU": "#28689B", "HTS": "#28689B"}  # Asunción population-validation palette


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def binary(s):
    """Unknowns stay missing; never treat 'nan' or empty as false."""
    return s.astype("string").str.strip().str.lower().map({
        "true": 1., "false": 0., "1": 1., "0": 0.,
        "1.0": 1., "0.0": 0., "yes": 1., "no": 0.,
        "t": 1., "f": 0., "y": 1., "n": 0.,
    }).astype(float)


def sex(s):
    # The Seville HTS cleaner maps its raw 1/2 to male/female; synthetic exports
    # generally use 0/1. Do not reinterpret ambiguous numeric '1' as a sex.
    m = {"male": "male", "m": "male", "males": "male", "hombre": "male",
         "female": "female", "f": "female", "females": "female", "mujer": "female"}
    return s.astype("string").str.strip().str.lower().map(m)


def ids(s):
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    x = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return x.mask(x.isin(["", "nan", "None", "<NA>"]))


def section_ids(df):
    for column in ("census_section_id", "commune_id", "iris_id"):
        if column in df:
            result = ids(df[column]).str[:10]
            return result.where(result.str.fullmatch(r"\d{10}").fillna(False))
    return pd.Series(pd.NA, index=df.index, dtype="string")


def residence_macrozone(df, city_code=SCOPE_CITY):
    sec = section_ids(df)
    return sec.str[:5].where(~sec.str.startswith(city_code).fillna(False), sec.str[:7])


def municipality(df):
    for column in ("municipality_id", "census_section_id", "commune_id", "iris_id"):
        if column in df:
            return ids(df[column]).str[:5]
    return pd.Series(pd.NA, index=df.index, dtype="string")


def check_weight(df, column, source):
    if column is None:
        return pd.Series(1., index=df.index)
    if column not in df:
        raise ValueError(f"{source}: missing weight {column}")
    w = numeric(df[column])
    if (~np.isfinite(w.to_numpy(float))).any() or (w < 0).any():
        raise ValueError(f"{source}: invalid or negative weights in {column}")
    return w.astype(float)


def prepare_persons(df, source, sampling_rate=1., weight_col=None):
    """Explicitly expand sampled IPU/final persons, retain HTS expansion weights."""
    required = {"person_id", "age", "sex"}
    if not required.issubset(df):
        raise ValueError(f"{source}: missing {sorted(required-set(df.columns))}")
    if df.person_id.duplicated().any():
        raise ValueError(f"{source}: duplicate person_id")
    if not 0 < sampling_rate <= 1:
        raise ValueError("sampling_rate must be in (0,1]")
    p = df.copy()
    p["_w"] = (1 / sampling_rate if weight_col is None else check_weight(p, weight_col, source))
    p["age"] = numeric(p.age)
    p.loc[~p.age.between(0, 115), "age"] = np.nan
    p["sex_norm"] = sex(p.sex)
    for out, cols in {"employed_n": ("employed",), "license_n": ("has_license", "has_driving_license"),
                      "studies_n": ("studies",)}.items():
        candidate = next((k for k in cols if k in p), None)
        p[out] = binary(p[candidate]) if candidate else np.nan
    # Exported final persons have no census-section code. Their validated
    # location is supplied via household home geometry. Preserve that spatial
    # assignment rather than overwriting it with NA in this common formatter.
    derived_municipality=municipality(p)
    derived_macrozone=residence_macrozone(p)
    if "municipality_id_validation" in p:
        p["municipality_id_validation"]=ids(p["municipality_id_validation"]).fillna(derived_municipality)
    else:
        p["municipality_id_validation"]=derived_municipality
    if "macrozone_id_validation" in p:
        p["macrozone_id_validation"]=ids(p["macrozone_id_validation"]).fillna(derived_macrozone)
    else:
        p["macrozone_id_validation"]=derived_macrozone
    return p


def rate_groups(df, breaks, outcome, group_sex=False, min_age=0):
    """Per-person weighted probabilities with approximate Kish effective sample N."""
    d = df.loc[df.age.ge(min_age) & df.age.notna() & df[outcome].isin([0., 1.]) & df._w.gt(0)].copy()
    if d.empty:
        return pd.DataFrame(columns=["age_start", "sex", "weighted_yes", "weighted_total", "n", "n_eff", "rate_pct"])
    d["age_start"] = pd.cut(d.age, breaks, labels=breaks[:-1], right=False, include_lowest=True).astype(float)
    d = d.dropna(subset=["age_start"])
    d["_yes"] = d[outcome] * d._w
    d["_w2"] = d._w ** 2
    keys = ["age_start"] + (["sex_norm"] if group_sex else [])
    result = (d.groupby(keys, observed=True, dropna=False)
              .agg(weighted_yes=("_yes", "sum"), weighted_total=("_w", "sum"),
                   n=(outcome, "size"), w2=("_w2", "sum")).reset_index())
    result["rate_pct"] = 100 * result.weighted_yes / result.weighted_total
    result["n_eff"] = result.weighted_total ** 2 / result.w2
    result = result.rename(columns={"sex_norm": "sex"})
    if not group_sex:
        result["sex"] = "all"
    return result.drop(columns="w2")


def census_population_in_bins(population, breaks, by_sex=True):
    """Fractionally allocate official five-year Census bins where boundaries differ.

    This is an explicit UNIFORM-WITHIN-5-YEAR-BIN approximation, required for
    16–19 employment and 18–20, 21–24 driving-license denominators. The 100+
    open-ended population bin is assigned only to an open-ended target group.
    """
    required = {"age_class", "sex", "weight"}
    if not required.issubset(population):
        raise ValueError(f"census population missing {required-set(population.columns)}")
    p = population.copy()
    p["age_start"] = numeric(p.age_class)
    p["sex"] = sex(p.sex)
    p["_w"] = check_weight(p, "weight", "census population")
    if p.age_start.isna().any() or (~p.age_start.mod(5).eq(0)).any():
        raise ValueError("Census age_class should be the lower bounds of 5-year bins")
    parts = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        fraction = (np.minimum(p.age_start + 5, hi) - np.maximum(p.age_start, lo)).clip(0, 5) / 5
        fraction = fraction.mask(p.age_start.eq(100), float(np.isinf(hi) and lo <= 100))
        select = fraction > 0
        if select.any():
            block = p.loc[select, ["sex", "_w"]].copy()
            block["age_start"] = lo
            block["_w"] *= fraction[select]
            parts.append(block)
    if not parts:
        return pd.DataFrame(columns=["age_start", "sex", "population"])
    expanded = pd.concat(parts, ignore_index=True)
    cols = ["age_start", "sex"] if by_sex else ["age_start"]
    res = expanded.groupby(cols, dropna=False, observed=True)["_w"].sum().reset_index(name="population")
    if not by_sex:
        res["sex"] = "all"
    return res


def census_outcome_rates(population, outcome, breaks, by_sex=False):
    """Census employed/driver count vs census-population age/sex denominator."""
    o = outcome.copy()
    o["age_start"] = numeric(o.age_class)
    o["sex"] = sex(o.sex)
    o["weight"] = check_weight(o, "weight", "census outcome")
    valid = set(breaks[:-1])
    unexpected = set(o.age_start.dropna().unique()) - valid
    if unexpected:
        raise ValueError(f"Census outcome age groups {sorted(unexpected)} don't match age breaks")
    keys = ["age_start", "sex"] if by_sex else ["age_start"]
    num = o.groupby(keys, dropna=False, observed=True).weight.sum().reset_index(name="weighted_yes")
    if not by_sex:
        num["sex"] = "all"
    den = census_population_in_bins(population, breaks, by_sex=by_sex)
    merged = den.merge(num, on=["age_start", "sex"], how="left", validate="one_to_one")
    merged["weighted_yes"] = merged.weighted_yes.fillna(0)
    merged["weighted_total"] = merged.population
    merged["rate_pct"] = 100 * merged.weighted_yes / merged.weighted_total.replace(0, np.nan)
    merged["n"] = np.nan
    merged["n_eff"] = np.nan
    return merged[["age_start", "sex", "weighted_yes", "weighted_total", "n", "n_eff", "rate_pct"]]


def write_json(path, data):
    def safe(x):
        if isinstance(x, dict): return {str(k):safe(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [safe(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return safe(float(x))
        if isinstance(x, float) and not np.isfinite(x): return None
        if x is pd.NA: return None
        return x
    Path(path).write_text(json.dumps(safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_index(folder, title, manifest, notes=()):
    """Small offline report index, like the Asuncion diagnostic dashboard."""
    output = Path(folder)
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<style>body{font:15px/1.6 system-ui,sans-serif;max-width:1080px;margin:auto;padding:24px;color:#243447}'
             'figure{border-bottom:1px solid #e2e5e9;padding:20px 0;margin:0}figure img{width:100%;max-width:1000px}'
             'figcaption{font-weight:bold;font-size:18px}small{color:#58697d}a{color:#116b78}</style></head><body>',
             f'<h1>{escape(title)}</h1>']
    if notes:
        parts += ['<h2>Interpretation & source notes</h2><ul>'] + [f'<li>{escape(str(n))}</li>' for n in notes] + ['</ul>']
    parts.append('<h2>Figures</h2>')
    for row in manifest:
        file, heading, note = row['file'], row['title'], row.get('note', '')
        parts.append(f'<figure><figcaption>{escape(heading)}</figcaption>'
                     f'<img loading="lazy" src="{escape(file, quote=True)}" alt="{escape(heading)}">'
                     f'<p><small>{escape(note)}</small></p></figure>')
    parts.append('</body></html>')
    (output/'index.html').write_text(''.join(parts), encoding='utf-8')


def source_summary(data, cols=('age', 'sex_norm', 'employed_n', 'license_n', 'studies_n')):
    return pd.DataFrame([{"source": name, "n": len(df), "weight_total": float(df._w.sum()),
                          **{f"{col}_missing_pct": round(df[col].isna().mean() * 100, 2) if col in df and len(df) else np.nan for col in cols}}
                         for name, df in data.items()])


def metric_error(obs, syn):
    return np.where(obs > 0, (syn - obs) / obs * 100., np.nan)


def write_home_exclusion_report(folder, quality, affected_plots, report_name="Seville validation",
                                additional_outputs=()):
    """Report skipped *geographic* households; never drop them globally.

    ``affected_plots`` contains already-generated manifest entries and is only
    populated for figures that require residential coordinates. The report is
    produced even if there are zero missing homes so runs are comparable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    total = int(quality.get("households", 0))
    missing = int(quality.get("households_missing_home", 0))
    people = int(quality.get("persons_missing_home", 0))
    mapped = total - missing
    data = pd.DataFrame([{
        "total_exported_households":total,
        "households_with_mapped_home":mapped,
        "households_excluded_missing_home":missing,
        "persons_in_excluded_households":people,
        "excluded_household_ids":";".join(str(x) for x in quality.get("missing_home_household_ids", [])),
    }])
    data.to_csv(folder / "home_location_exclusions.csv", index=False)

    rows = []
    for item in affected_plots:
        rows.append({"file":item["file"], "scope":item.get("scope", ""),
                     "type":"plot", "note":"Final residential geography requires an exported home point."})
    for item in additional_outputs:
        rows.append({"file":item["file"], "scope":item.get("scope", ""),
                     "type":"data output", "note":item.get("note", "Residential location required")})
    pd.DataFrame(rows, columns=["file", "scope", "type", "note"]).drop_duplicates().to_csv(
        folder / "home_location_affected_plots.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(["No exported home location"], [missing], color="#b54c52", width=.4)
    ax.text(0, missing + max(0.05, missing * .04), f"{missing:,}", ha="center",
            va="bottom", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(1.3, missing*1.2))
    ax.set_ylabel("Households excluded from location-dependent checks")
    ax.set_title(report_name + " · missing home locations")
    ax.text(.5,-.17, f"{missing:,} of {total:,} households have no exported home "
            f"({people:,} people). Non-spatial totals retain them.",
            transform=ax.transAxes, ha="center", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "home_location_exclusions.png", dpi=165, bbox_inches="tight")
    plt.close(fig)
    return data
