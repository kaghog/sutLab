"""PNG-only analysis of final EQASIM/Synpp exported population.

Place at: analysis/asuncion/final_population_plots.py (matching your example imports).
Add `analysis.asuncion.final_population_plots` to your Synpp run list.
If your repository uses a different Python package layout, adjust the stage path.

Reads semicolon-delimited <output_path>/<output_prefix>{persons,households,trips,
activities,vehicles}.csv. All figures go to <analysis_path>/final_population_plots/.
Optional GeoPackage maps use the corresponding <prefix>*.gpkg files.

Reference datasets are requested to compare final export with IPU, HTS and
census. IPU and final results are compared as distributions, NOT matched by ID.
"""

from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

COLORS = {
    "Final": "#087e8b", "IPU": "#4567a9", "HTS": "#e18d32",
    "Census": "#828b99", "Male": "#4567a9", "Female": "#dd7c9f",
}
OTHER = "#66788a"
AGE_BREAKS = list(range(0, 95, 5)) + [np.inf]
AGE_LABELS = [f"{lo}–{lo+4}" for lo in range(0, 90, 5)] + ["90+"]
HH_LABELS = ["1", "2", "3", "4", "5", "6+"]


def configure(context):
    context.stage("synthesis.output")
    context.stage("asuncion.ipu.attributed")
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.data.hts.entd.filtered")
    context.config("output_path")
    context.config("output_prefix")
    context.config("analysis_path")


def _style(ax, ylabel=None, xlabel=None, title=None, grid="y"):
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    if grid:
        ax.grid(axis=grid, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("right", "top"):
        ax.spines[spine].set_visible(False)


def _save(fig, folder, name):
    fig.tight_layout(pad=1.5)
    fig.savefig(folder / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sex(series):
    """Normalize output strings and likely input census/HTS numeric encodings."""
    as_text = series.astype("string").str.strip().str.lower()
    nums = set(as_text.dropna().unique()) & {"0", "1", "2"}
    mapping = {"male": "Male", "m": "Male", "homme": "Male",
               "female": "Female", "f": "Female", "femme": "Female"}
    if "2" in nums:            # Common survey convention: 1=Male, 2=Female.
        mapping.update({"1": "Male", "2": "Female", "0": "Male"})
    else:                      # Pipeline gender selector: 0=Male, 1=Female.
        mapping.update({"0": "Male", "1": "Female"})
    return as_text.map(mapping)


def _binary(series):
    """Return numeric 0/1 with missing/other values as NaN."""
    x = series.astype("string").str.strip().str.lower()
    return pd.to_numeric(x.map({"true": "1", "false": "0", "1": "1", "0": "0",
                                "yes": "1", "no": "0", "t": "1", "f": "0"}),
                         errors="coerce")


def _prepare_persons(df):
    out = df.copy()
    out["sex_plot"] = _sex(out["sex"])
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["age_group"] = pd.cut(out["age"], AGE_BREAKS, labels=AGE_LABELS,
                              right=False, include_lowest=True)
    if "employed" in out:
        out["emp_plot"] = _binary(out["employed"])
    if "has_driving_license" in out:
        out["license_plot"] = _binary(out["has_driving_license"])
    elif "has_license" in out:
        out["license_plot"] = _binary(out["has_license"])
    if "has_pt_subscription" in out:
        out["pt_plot"] = _binary(out["has_pt_subscription"])
    return out


def _w(df, col=None):
    if col is not None and col in df:
        return pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    return pd.Series(1.0, index=df.index)


def _distribution(df, category, weight=None, labels=None):
    valid = df[category].notna()
    counts = _w(df, weight).loc[valid].groupby(df.loc[valid, category], observed=True).sum()
    if labels is not None:
        counts = counts.reindex(labels, fill_value=0)
    total = counts.sum()
    return counts * 100.0 / total if total else counts.astype(float) * np.nan


def _rate(df, target, group, weight=None):
    if target not in df or group not in df:
        return pd.Series(dtype=float)
    idx = df[target].notna() & df[group].notna()
    sub = df.loc[idx, [target, group]].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["_w"] = _w(df, weight).loc[idx].to_numpy()
    sub["_yes"] = sub[target].to_numpy() * sub["_w"].to_numpy()
    den = sub.groupby(group, observed=True)["_w"].sum()
    return 100.0 * sub.groupby(group, observed=True)["_yes"].sum().div(den.replace(0, np.nan))


def _grouped_bar(series_by_source, folder, filename, title, xlabel, ylabel="Share (%)",
                 categories=None, ticklabels=None, ylim=None, horizontal=False):
    if categories is None:
        categories = sorted(set().union(*(s.index.tolist() for s in series_by_source.values())), key=str)
    cats = list(categories)
    if not cats:
        return
    fig, ax = plt.subplots(figsize=(max(8.8, 0.62 * len(cats) + 3), 5.5))
    xs = np.arange(len(cats), dtype=float)
    width = 0.82 / len(series_by_source)
    for i, (source, data) in enumerate(series_by_source.items()):
        y = data.reindex(cats).to_numpy(dtype=float)
        offset = (i - (len(series_by_source) - 1) / 2.0) * width
        kw = dict(label=source, color=COLORS.get(source, OTHER), alpha=0.92)
        if horizontal:
            ax.barh(xs + offset, y, width, **kw)
        else:
            ax.bar(xs + offset, y, width, **kw)
    labels = ticklabels if ticklabels is not None else [str(x) for x in cats]
    if horizontal:
        ax.set_yticks(xs, labels)
        _style(ax, xlabel=ylabel, ylabel=xlabel, title=title, grid="x")
    else:
        ax.set_xticks(xs, labels, rotation=35 if len(cats) > 6 else 0, ha="right" if len(cats) > 6 else "center")
        _style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    if ylim is not None:
        (ax.set_xlim if horizontal else ax.set_ylim)(*ylim)
    ax.legend(frameon=False, ncol=min(4, len(series_by_source)), loc="best")
    _save(fig, folder, filename)


def _line_rates(sources, field, folder, filename, title):
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    for source, (frame, weight) in sources.items():
        r = _rate(frame, field, "age_group", weight).reindex(AGE_LABELS)
        ax.plot(np.arange(len(AGE_LABELS)), r, marker="o", markersize=3.5,
                linewidth=2.3, color=COLORS.get(source, OTHER), label=source)
    ax.set_xticks(np.arange(len(AGE_LABELS)), AGE_LABELS, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    _style(ax, xlabel="Age group (years)", ylabel="People in age group (%)", title=title)
    ax.legend(frameon=False, ncol=3)
    _save(fig, folder, filename)


def _age_sex_rates(sources, field, folder, filename, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, sex in zip(axes, ("Male", "Female")):
        for source, (frame, weight) in sources.items():
            sample = frame.loc[frame["sex_plot"] == sex]
            r = _rate(sample, field, "age_group", weight).reindex(AGE_LABELS)
            ax.plot(np.arange(len(AGE_LABELS)), r, marker="o", markersize=3,
                    linewidth=2.0, color=COLORS.get(source, OTHER), label=source)
        ax.set_xticks(np.arange(len(AGE_LABELS)), AGE_LABELS, rotation=55, ha="right")
        ax.set_ylim(0, 100)
        _style(ax, xlabel="Age group (years)", ylabel="Within-group rate (%)", title=sex)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.52, 0.985))
    fig.suptitle(title, x=0.13, ha="left", fontsize=14, weight="bold", y=1.035)
    _save(fig, folder, filename)


def _heatmap(table, folder, filename, title, xlabel, ylabel, cbar, cmap="YlGnBu", annot=False):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8.4, 0.7 * table.shape[1] + 3), max(4.5, 0.6 * table.shape[0] + 2)))
    arr = table.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, aspect="auto", interpolation="nearest",
                   vmin=0 if np.nanmin(arr) >= 0 else None)
    ax.set_xticks(range(table.shape[1]), [str(v) for v in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), [str(v) for v in table.index])
    _style(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=None)
    if annot and arr.size <= 125:
        for (i, j), value in np.ndenumerate(arr):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if value > np.nanmax(arr) * 0.65 else "#172c3a")
    fig.colorbar(im, ax=ax, label=cbar, shrink=0.86)
    _save(fig, folder, filename)


def _difference_by_age(final, ipu, hts, field, folder, filename, title):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True, sharey=True)
    final_rate = _rate(final, field, "age_group").reindex(AGE_LABELS)
    for ax, (other, frame, weight) in zip(axes, (("IPU", ipu, None), ("HTS", hts, "person_weight"))):
        other_rate = _rate(frame, field, "age_group", weight).reindex(AGE_LABELS)
        gap = final_rate - other_rate
        ax.bar(np.arange(len(AGE_LABELS)), gap, color=np.where(gap >= 0, "#168276", "#d77555"), alpha=0.9)
        ax.axhline(0, color="black", linewidth=1)
        _style(ax, ylabel="Percentage points", title=f"Final − {other}")
    axes[-1].set_xticks(np.arange(len(AGE_LABELS)), AGE_LABELS, rotation=45, ha="right")
    axes[-1].set_xlabel("Age group (years)")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.08, ha="left")
    _save(fig, folder, filename)


def _pop_plots(folder, final, ipu, hts, census):
    sources = {"Final": (final, None), "IPU": (ipu, None), "HTS": (hts, "person_weight")}
    age_sources = {s: _distribution(d, "age_group", w, AGE_LABELS) for s, (d, w) in sources.items()}
    if {"age_class", "weight"}.issubset(census.columns):
        cn = census.copy()
        cn["age_group"] = pd.cut(pd.to_numeric(cn["age_class"], errors="coerce"), AGE_BREAKS,
                                 labels=AGE_LABELS, right=False)
        age_sources["Census"] = _distribution(cn, "age_group", "weight", AGE_LABELS)
    _grouped_bar(age_sources, folder, "01_age_distribution.png", "Age distribution | final vs references",
                 "Age group (years)", categories=AGE_LABELS)

    sex_sources = {s: _distribution(d, "sex_plot", w, ["Male", "Female"])
                   for s, (d, w) in sources.items()}
    if {"sex", "weight"}.issubset(census.columns):
        cs = census.copy()
        cs["sex_plot"] = _sex(cs["sex"])
        sex_sources["Census"] = _distribution(cs, "sex_plot", "weight", ["Male", "Female"])
    _grouped_bar(sex_sources, folder, "02_sex_distribution.png",
                 "Sex distribution | final vs references", "Sex", categories=["Male", "Female"])

    fig, ax = plt.subplots(figsize=(9, 8))
    ages = AGE_LABELS
    male = final.loc[final.sex_plot == "Male", "age_group"].value_counts().reindex(ages, fill_value=0) / len(final) * 100
    female = final.loc[final.sex_plot == "Female", "age_group"].value_counts().reindex(ages, fill_value=0) / len(final) * 100
    y = np.arange(len(ages))
    ax.barh(y, -male, label="Male", color=COLORS["Male"])
    ax.barh(y, female, label="Female", color=COLORS["Female"])
    ax.set_yticks(y, ages)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{abs(x):g}"))
    _style(ax, xlabel="Share of all persons (%)", ylabel="Age group", title="Final population pyramid", grid="x")
    ax.legend(frameon=False)
    _save(fig, folder, "03_population_pyramid.png")

    table = pd.crosstab(final.sex_plot, final.age_group).reindex(index=["Male", "Female"], columns=AGE_LABELS, fill_value=0)
    table = table * 100 / len(final)
    _heatmap(table, folder, "04_age_sex_heatmap.png", "Final population | age × sex", "Age group",
             "Sex", "Share of all persons (%)")

    for field, stem, text in (("emp_plot", "employment", "Employment"),
                              ("license_plot", "license", "Driving licence ownership")):
        _line_rates(sources, field, folder, f"05_{stem}_by_age.png", f"{text} rate by age")
        _grouped_bar({name: _rate(df, field, "sex_plot", w) for name, (df, w) in sources.items()},
                     folder, f"06_{stem}_by_sex.png", f"{text} rate by sex", "Sex",
                     "People within sex (%)", categories=["Male", "Female"], ylim=(0, 100))
        _age_sex_rates(sources, field, folder, f"07_{stem}_by_age_and_sex.png",
                       f"{text} rate by age and sex")
        _difference_by_age(final, ipu, hts, field, folder, f"08_{stem}_age_difference.png",
                           f"{text}: how the final population differs from IPU and HTS")

    rate_by_employment = {s: _rate(d, "license_plot", "emp_plot", w)
                          for s, (d, w) in sources.items()}
    _grouped_bar(rate_by_employment, folder, "09_license_by_employment.png",
                 "Driving licence ownership by employment", "Employment status",
                 "Licensed within employment group (%)", categories=[0, 1],
                 ticklabels=["Not employed", "Employed"], ylim=(0, 100))

    # Employment/age especially helpful for catching unexpected employment among children.
    minor = final.loc[final.age < 25].copy()
    minor["early_age"] = pd.cut(minor.age, [0, 5, 10, 15, 18, 21, 25], right=False,
                                labels=["0–4", "5–9", "10–14", "15–17", "18–20", "21–24"])
    rates = _rate(minor, "emp_plot", "early_age")
    _grouped_bar({"Final": rates}, folder, "10_employment_early_ages.png",
                 "Employment rate among children and young adults", "Age group (years)",
                 "Employed within age group (%)", categories=list(minor.early_age.cat.categories), ylim=(0, 100))

    # Cross-property heatmaps use denominator from persons in the age × sex cell.
    for field, title, filename in (("emp_plot", "Employment", "11_employment_age_sex_heatmap.png"),
                                    ("license_plot", "Driving licence", "12_license_age_sex_heatmap.png")):
        data = pd.concat({sex: _rate(final.loc[final.sex_plot == sex], field, "age_group")
                          for sex in ("Male", "Female")}, axis=1)
        data = data.reindex(AGE_LABELS).T
        _heatmap(data, folder, filename, f"Final | {title} rate by age and sex",
                 "Age group", "Sex", "Within-cell rate (%)")


def _household_plots(folder, final, households, ipu, hts_hh):
    sizes = final.groupby("household_id").size().rename("hh_size")
    hh = households.drop_duplicates("household_id").copy().set_index("household_id").join(sizes)
    hh["hh_size"] = hh.hh_size.fillna(0)
    hh["size_group"] = hh.hh_size.clip(upper=6).astype(int)
    ipu_sizes = ipu.groupby("household_id").size().clip(upper=6)
    hh_series = {"Final": _distribution(hh, "size_group"),
                 "IPU": ipu_sizes.value_counts(normalize=True).mul(100)}
    if "household_size" in hts_hh and "household_weight" in hts_hh:
        ht = hts_hh.copy()
        ht["size_group"] = pd.to_numeric(ht.household_size, errors="coerce").clip(upper=6)
        hh_series["HTS"] = _distribution(ht, "size_group", "household_weight")
    _grouped_bar(hh_series, folder, "13_household_size.png", "Household size distribution",
                 "People per household", categories=[1, 2, 3, 4, 5, 6], ticklabels=HH_LABELS)

    metrics = {
        "children": final.assign(is_child=(final.age < 18).astype(int)).groupby("household_id").is_child.sum(),
        "employed": final.groupby("household_id").emp_plot.sum(),
        "licensed": final.groupby("household_id").license_plot.sum(),
    }
    for number, (key, values) in enumerate(metrics.items(), start=14):
        clipped = values.clip(upper=5).value_counts(normalize=True).mul(100)
        _grouped_bar({"Final": clipped}, folder, f"{number:02d}_{key}_per_household.png",
                     f"Final | number of {key} persons per household", "People per household",
                     categories=[0, 1, 2, 3, 4, 5], ticklabels=["0", "1", "2", "3", "4", "5+"])

    for number, key, title in ((17, "number_of_cars", "Cars"),
                                (18, "number_of_motorbikes", "Motorcycles"),
                                (19, "number_of_bikes", "Bicycles")):
        vals = pd.to_numeric(hh[key], errors="coerce").clip(upper=6)
        _grouped_bar({"Final": vals.value_counts(normalize=True).mul(100)},
                     folder, f"{number:02d}_{key}_distribution.png",
                     f"Final | {title} reported per household", "Vehicles per household",
                     categories=range(7), ticklabels=["0", "1", "2", "3", "4", "5", "6+"])

    for key, title, fname in (("number_of_cars", "Cars", "20_cars_by_household_size.png"),
                              ("number_of_bikes", "Bicycles", "21_bikes_by_household_size.png")):
        means = hh.groupby("size_group")[key].mean().reindex(range(1, 7))
        _grouped_bar({"Final": means}, folder, fname,
                     f"Mean number of {title.lower()} by household size", "People per household",
                     "Mean vehicles per household", categories=range(1, 7), ticklabels=HH_LABELS)

    car_group = pd.cut(pd.to_numeric(hh["number_of_cars"], errors="coerce"),
                       [-0.1, 0.5, 1.5, 2.5, np.inf],
                       labels=["0", "1", "2", "3+"])
    hh["car_group"] = car_group
    hh["licensed_count"] = metrics["licensed"]
    hh["has_licensed_person"] = (hh.licensed_count > 0).astype(float)
    _grouped_bar({"Final": _rate(hh.reset_index(), "has_licensed_person", "car_group")},
                 folder, "22_license_presence_by_household_cars.png",
                 "Households with a licence holder, by cars reported", "Reported cars per household",
                 "Households with ≥1 licensed person (%)", categories=list(car_group.cat.categories), ylim=(0, 100))

    # Missing fields are shown honestly rather than assigning zero to unknown values.
    percent = pd.Series({
        "Age": final.age.notna().mean() * 100,
        "Employment": final.emp_plot.notna().mean() * 100,
        "Driving licence": final.license_plot.notna().mean() * 100,
        "PT subscription": final.pt_plot.notna().mean() * 100 if "pt_plot" in final else 0,
        "Socioprofessional class": final.socioprofessional_class.notna().mean() * 100
            if "socioprofessional_class" in final else 0,
        "Household income": hh.income.notna().mean() * 100 if "income" in hh else 0,
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    ys = np.arange(len(percent))
    ax.barh(ys, percent.to_numpy(), color=COLORS["Final"])
    ax.set_yticks(ys, percent.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    for i, val in enumerate(percent):
        ax.text(min(101, val + 1), i, f"{val:.0f}%", va="center", fontsize=9)
    _style(ax, xlabel="Non-missing records (%)", title="Availability of final output attributes", grid="x")
    _save(fig, folder, "23_attribute_availability.png")

    if "income" in hh:
        values = pd.to_numeric(hh.income, errors="coerce")
        if values.nunique(dropna=True) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(values.dropna(), bins=45, color=COLORS["Final"], alpha=0.9)
            _style(ax, xlabel="Exported household income (source units)", ylabel="Households",
                   title="Final | household income")
            _save(fig, folder, "24_household_income.png")
        else:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis("off")
            ax.text(0.5, 0.63, "Household income", ha="center", fontsize=17, weight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.37, f"Only one exported value: {values.dropna().iloc[0]:g}" if values.notna().any() else "All entries missing",
                    ha="center", fontsize=13, transform=ax.transAxes)
            ax.text(0.5, 0.15, "No meaningful income distribution can be drawn from this export.",
                    ha="center", fontsize=10, color=OTHER, transform=ax.transAxes)
            _save(fig, folder, "24_household_income.png")


def _travel_plots(folder, final, trips, activities):
    t = trips.copy()
    a = activities.copy()
    num_trips = t.groupby("person_id").size()
    p = final.copy()
    p["trip_count"] = p.person_id.map(num_trips).fillna(0).astype(int)
    p["travels"] = (p.trip_count > 0).astype(float)
    p["trips_cap"] = p.trip_count.clip(upper=8)

    _grouped_bar({"Final": _distribution(p, "trips_cap")}, folder,
                 "25_trips_per_person.png", "Trips per person | including non-travellers",
                 "Trips per person", categories=range(9), ticklabels=[str(x) for x in range(8)] + ["8+"])
    _line_rates({"Final": (p, None)}, "travels", folder,
                "26_trip_participation_by_age.png", "Share of persons with ≥1 trip, by age")
    _age_sex_rates({"Final": (p, None)}, "travels", folder,
                    "27_trip_participation_by_age_sex.png", "Trip participation by age and sex")
    _grouped_bar({"Final": _rate(p, "travels", "emp_plot")}, folder,
                 "28_trip_participation_by_employment.png", "Trip participation by employment",
                 "Employment status", "Persons with ≥1 trip (%)", categories=[0, 1],
                 ticklabels=["Not employed", "Employed"], ylim=(0, 100))

    mean_trips = p.groupby("age_group", observed=True).trip_count.mean().reindex(AGE_LABELS)
    _grouped_bar({"Final": mean_trips}, folder, "29_mean_trips_by_age.png",
                 "Mean trips per person (including zero trips)", "Age group (years)",
                 "Mean trips per person", categories=AGE_LABELS)

    _grouped_bar({"Final": _distribution(t, "following_purpose")}, folder,
                 "30_trip_destination_purposes.png", "Trip destination purposes", "Destination purpose")
    _grouped_bar({"Final": _distribution(a, "purpose")}, folder,
                 "31_activity_purposes.png", "Activities by purpose (home included)", "Activity purpose")

    t["departure_hour"] = pd.to_numeric(t.departure_time, errors="coerce") / 3600.0
    end_hour = int(np.ceil(t.departure_hour.max()))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(t.departure_hour.dropna(), bins=np.arange(0, max(24, end_hour) + 2, 1),
            color=COLORS["Final"], alpha=0.88)
    _style(ax, xlabel="Hours from simulation start (can exceed 24)", ylabel="Trips",
           title="Trip departures over simulated time")
    _save(fig, folder, "32_trip_departure_hour.png")

    purposes = t.following_purpose.value_counts().head(6).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for purpose, sub in t.loc[t.following_purpose.isin(purposes)].groupby("following_purpose"):
        ax.hist(sub.departure_hour, bins=np.arange(0, max(24, end_hour) + 2, 2),
                histtype="step", density=True, linewidth=2, label=str(purpose))
    _style(ax, xlabel="Hours from simulation start", ylabel="Probability density (per hour)",
           title="Departure time profiles by destination purpose")
    ax.legend(frameon=False, ncol=3)
    _save(fig, folder, "33_departure_profiles_by_purpose.png")

    t["distance_km"] = pd.to_numeric(t.euclidean_distance, errors="coerce") / 1000.0
    t["duration_min"] = (pd.to_numeric(t.arrival_time, errors="coerce") -
                          pd.to_numeric(t.departure_time, errors="coerce")) / 60.0
    ordered = t.following_purpose.value_counts().head(7).index.tolist()
    for num, field, label, filename in ((34, "distance_km", "Euclidean distance (km)", "34_trip_distance_by_purpose.png"),
                                        (35, "duration_min", "Trip duration (min)", "35_trip_duration_by_purpose.png")):
        fig, ax = plt.subplots(figsize=(11, 5.2))
        groups = [t.loc[(t.following_purpose == cat) & t[field].notna(), field].to_numpy() for cat in ordered]
        parts = ax.boxplot(groups, tick_labels=ordered, showfliers=False, patch_artist=True)
        for patch in parts["boxes"]:
            patch.set_facecolor(COLORS["Final"])
            patch.set_alpha(0.65)
        ax.tick_params(axis="x", rotation=25)
        _style(ax, xlabel="Destination purpose", ylabel=label,
               title=("Trip distance" if field == "distance_km" else "Trip duration") + " by purpose | box = IQR, whiskers = 1.5×IQR")
        _save(fig, folder, filename)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    hexb = ax.hexbin(t.distance_km, t.duration_min, gridsize=45, mincnt=1, bins="log", cmap="YlGnBu")
    _style(ax, xlabel="Euclidean distance (km)", ylabel="Trip duration (min)",
           title="Trip distance vs duration | density")
    fig.colorbar(hexb, ax=ax, label="Trips (log color scale)")
    _save(fig, folder, "36_trip_distance_duration_density.png")

    matrix = pd.crosstab(t.preceding_purpose, t.following_purpose)
    rowsum = matrix.sum(axis=1).replace(0, np.nan)
    matrix = matrix.div(rowsum, axis=0) * 100.0
    _heatmap(matrix, folder, "37_origin_destination_purpose_matrix.png",
             "Trip purpose transitions | row-normalized", "Following purpose", "Preceding purpose",
             "Trips from origin purpose (%)", annot=True)

    a["start_hour"] = pd.to_numeric(a.start_time, errors="coerce") / 3600.0
    major = a.purpose.value_counts().head(6).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for purpose in major:
        s = a.loc[(a.purpose == purpose) & a.start_hour.notna(), "start_hour"]
        if len(s):
            ax.hist(s, bins=np.arange(0, max(24, int(np.ceil(a.start_hour.max()))) + 2, 2),
                    histtype="step", linewidth=2, density=True, label=purpose)
    _style(ax, xlabel="Hours from simulation start", ylabel="Probability density (per hour)",
           title="Activity start time profiles | defined start times only")
    ax.legend(frameon=False, ncol=3)
    _save(fig, folder, "38_activity_start_time_by_purpose.png")

    # Trips are not mode-labelled in the provided output: no mode-share chart is fabricated.
    return p


def _vehicles_plot(folder, vehicles):
    _grouped_bar({"Final": _distribution(vehicles, "mode")}, folder,
                 "39_generated_vehicle_records_by_mode.png",
                 "Generated vehicle records by role (not household vehicle ownership)",
                 "Vehicle mode")


def _diverging_heatmap(table, folder, filename, title, units):
    fig, ax = plt.subplots(figsize=(13, 4.7))
    values = table.to_numpy(dtype=float)
    upper = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    upper = max(float(upper), 0.1)
    image = ax.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-upper, vmax=upper,
                      aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(table.index)), [str(x) for x in table.index])
    ax.set_xticks(range(len(table.columns)), [str(x) for x in table.columns],
                  rotation=45, ha="right")
    _style(ax, xlabel="Age group (years)", ylabel="Sex", title=title, grid=None)
    fig.colorbar(image, ax=ax, label=units, shrink=0.85)
    _save(fig, folder, filename)


def _change_plots(folder, final, ipu):
    def composition(d):
        counts = pd.crosstab(d.sex_plot, d.age_group)
        return counts.reindex(index=["Male", "Female"], columns=AGE_LABELS, fill_value=0) * 100 / len(d)
    delta = composition(final) - composition(ipu)
    _diverging_heatmap(delta, folder, "44_population_age_sex_change_vs_ipu.png",
                       "Population composition change | Final − IPU", "Share of all persons (percentage points)")
    for number, key, text in ((45, "emp_plot", "Employment"),
                              (46, "license_plot", "Driving licence")):
        f = pd.DataFrame({sex: _rate(final[final.sex_plot == sex], key, "age_group")
                          for sex in ("Male", "Female")}).reindex(AGE_LABELS).T
        i = pd.DataFrame({sex: _rate(ipu[ipu.sex_plot == sex], key, "age_group")
                          for sex in ("Male", "Female")}).reindex(AGE_LABELS).T
        _diverging_heatmap(f - i, folder, f"{number:02d}_{key}_age_sex_change_vs_ipu.png",
                           f"{text} rate change | Final − IPU", "Within-group rate (percentage points)")


def _spatial_plots(folder, output_path, prefix):
    """Optional static maps. Requires geopandas; no basemap/network access."""
    try:
        import geopandas as gpd
    except ImportError:
        return
    names = {x: output_path / f"{prefix}{x}.gpkg" for x in ("homes", "activities", "trips", "commutes")}

    def read_spatial(path, maximum=100000):
        # Stride-sample in SQLite BEFORE decoding potentially millions of geometries.
        # Sampling is systematic over feature IDs rather than geographically filtered.
        with sqlite3.connect(path) as conn:
            table = conn.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features' LIMIT 1").fetchone()[0]
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        step = max(1, int(np.ceil(count / maximum)))
        return gpd.read_file(path, where=f"fid % {step} = 0" if step > 1 else None)

    if names["homes"].exists():
        homes = read_spatial(names["homes"])
        if not homes.empty:
            fig, ax = plt.subplots(figsize=(9, 7))
            hb = ax.hexbin(homes.geometry.x, homes.geometry.y, gridsize=55,
                           bins="log", mincnt=1, cmap="YlGnBu")
            ax.set_aspect("equal", adjustable="datalim")
            _style(ax, xlabel="Coordinate X", ylabel="Coordinate Y",
                   title="Final household locations | spatial concentration (sample if large)", grid=None)
            fig.colorbar(hb, ax=ax, label="Plotted homes per hexagon (log color scale)")
            _save(fig, folder, "40_home_spatial_density.png")

    if names["activities"].exists():
        acts = read_spatial(names["activities"])
        if not acts.empty:
            sample = acts.loc[acts.purpose != "home"]
            if len(sample) > 8000:
                sample = sample.sample(8000, random_state=24)
            fig, ax = plt.subplots(figsize=(9, 7))
            for purpose, g in sample.groupby("purpose"):
                ax.scatter(g.geometry.x, g.geometry.y, s=5, alpha=0.35, label=str(purpose), rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            _style(ax, xlabel="Coordinate X", ylabel="Coordinate Y",
                   title="Non-home activity locations | random sample", grid=None)
            ax.legend(frameon=False, markerscale=2, ncol=2, loc="best")
            _save(fig, folder, "41_activity_locations.png")

    for index, label, title in ((42, "trips", "Trip geometry"), (43, "commutes", "Commute geometry")):
        if names[label].exists():
            geo = read_spatial(names[label])
            if not geo.empty:
                if len(geo) > 2500:
                    geo = geo.sample(2500, random_state=24)
                fig, ax = plt.subplots(figsize=(9, 7))
                geo.plot(ax=ax, color=COLORS["Final"], alpha=0.08, linewidth=0.5)
                ax.set_aspect("equal", adjustable="datalim")
                _style(ax, xlabel="Coordinate X", ylabel="Coordinate Y",
                       title=f"{title} | sampled linework (not a traffic-volume map)", grid=None)
                _save(fig, folder, f"{index:02d}_{label}_geometry.png")


def execute(context):
    output = Path(context.config("output_path"))
    prefix = context.config("output_prefix")
    folder = Path(context.config("analysis_path")) / "final_population_plots"
    folder.mkdir(parents=True, exist_ok=True)

    def load(name):
        return pd.read_csv(output / f"{prefix}{name}.csv", sep=";", encoding="latin1")

    final = _prepare_persons(load("persons"))
    households = load("households")
    trips = load("trips")
    activities = load("activities")
    vehicles = load("vehicles")

    ipu = _prepare_persons(context.stage("asuncion.ipu.attributed").copy())
    hts_households, hts_persons, _ = context.stage("asuncion.data.hts.entd.filtered")
    hts = _prepare_persons(hts_persons.copy())
    census = context.stage("asuncion.data.census.population").copy()

    _pop_plots(folder, final, ipu, hts, census)
    _change_plots(folder, final, ipu)
    _household_plots(folder, final, households, ipu, hts_households.copy())
    _travel_plots(folder, final, trips, activities)
    _vehicles_plot(folder, vehicles)
    _spatial_plots(folder, output, prefix)
    print(f"Final population charts: {len(list(folder.glob('*.png')))} PNGs in {folder}")
