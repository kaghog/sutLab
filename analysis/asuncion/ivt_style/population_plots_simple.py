"""Simple population plots for the asuncion EQASIM / Synpp pipeline.

Creates only PNG figures in <analysis_path>/population_plots/.
Assumes employed and has_license are boolean or numeric 0/1.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Run on machines without a graphical display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Census": "#64748b", "IPU": "#2563eb", "HTS": "#ea580c"}


def configure(context):
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.data.census.income")
    context.stage("asuncion.ipu.attributed")
    context.stage("asuncion.data.hts.entd.filtered")
    context.config("analysis_path")


def save(fig, folder, filename):
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def style(ax, ylabel="Share (%)"):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def shares(df, column, weight=None):
    d = df.dropna(subset=[column])
    counts = d.groupby(column)[weight].sum() if weight else d[column].value_counts()
    return counts / counts.sum() * 100


def rates(df, column, group, weight=None):
    """Weighted percentage where a 0/1 or boolean column equals 1."""
    d = df.dropna(subset=[column, group]).copy()
    d["_weight"] = d[weight] if weight else 1.0
    d["_positive"] = d[column].astype(float) * d["_weight"]
    totals = d.groupby(group)["_weight"].sum()
    return d.groupby(group)["_positive"].sum() / totals * 100


def grouped_bars(data, title, xlabel, folder, filename, categories=None, labels=None):
    if categories is None:
        categories = sorted(set().union(*(s.index for s in data.values())), key=str)
    x = np.arange(len(categories))
    width = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(max(9, len(categories) * 0.65), 5))
    for i, (name, values) in enumerate(data.items()):
        offset = (i - (len(data) - 1) / 2) * width
        ax.bar(x + offset, values.reindex(categories, fill_value=0),
               width, label=name, color=COLORS[name])
    ax.set_xticks(x, labels if labels is not None else [str(v) for v in categories])
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    style(ax)
    ax.legend(frameon=False)
    save(fig, folder, filename)


def age_rate_plot(ipu, hts, field, title, folder, filename):
    a = rates(ipu, field, "age_10")
    b = rates(hts, field, "age_10", "person_weight")
    ages = sorted(set(a.index) | set(b.index))
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, series in [("IPU", a), ("HTS", b)]:
        ax.plot(ages, series.reindex(ages), marker="o", markersize=5,
                linewidth=2.4, label=name, color=COLORS[name])
    ax.set_xticks(ages, [f"{i}–{i+9}" for i in ages], rotation=45)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Age group (years)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    style(ax, "People in age group (%)")
    ax.legend(frameon=False)
    save(fig, folder, filename)


def age_sex_rate_plot(ipu, hts, field, title, folder, filename):
    sexes = sorted(set(ipu["sex"].dropna()) | set(hts["sex"].dropna()), key=str)
    ages = sorted(set(ipu["age_10"]) | set(hts["age_10"]))
    fig, axes = plt.subplots(1, len(sexes), figsize=(max(11, 5.8 * len(sexes)), 5), sharey=True, squeeze=False)
    for ax, sex in zip(axes[0], sexes):
        for name, df, weight in [("IPU", ipu, None), ("HTS", hts, "person_weight")]:
            s = rates(df.loc[df["sex"] == sex], field, "age_10", weight)
            ax.plot(ages, s.reindex(ages), marker="o", markersize=4,
                    linewidth=2.2, label=name, color=COLORS[name])
        ax.set_title(f"Sex: {sex}")
        ax.set_xticks(ages, [f"{i}–{i+9}" for i in ages], rotation=45)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Age group (years)")
        style(ax, "People in age group (%)")
        ax.legend(frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    save(fig, folder, filename)


def sex_rate_plot(ipu, hts, field, title, folder, filename):
    ipu_rates = rates(ipu, field, "sex")
    hts_rates = rates(hts, field, "sex", "person_weight")
    sexes = sorted(set(ipu_rates.index) | set(hts_rates.index), key=str)
    x = np.arange(len(sexes))
    fig, ax = plt.subplots(figsize=(8, 5))
    for shift, (name, values) in zip([-0.19, 0.19], [("IPU", ipu_rates), ("HTS", hts_rates)]):
        ax.bar(x + shift, values.reindex(sexes, fill_value=0), 0.36,
               label=name, color=COLORS[name])
    ax.set_xticks(x, [str(v) for v in sexes])
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    style(ax, "People in sex group (%)")
    ax.legend(frameon=False)
    save(fig, folder, filename)


def execute(context):
    folder = Path(context.config("analysis_path")) / "population_plots"
    folder.mkdir(parents=True, exist_ok=True)

    census = context.stage("asuncion.data.census.population").copy()
    census_income = context.stage("asuncion.data.census.income").copy()
    ipu = context.stage("asuncion.ipu.attributed").copy()
    hts_hh, hts, _ = context.stage("asuncion.data.hts.entd.filtered")
    hts_hh = hts_hh.copy()
    hts = hts.copy()

    for df in (ipu, hts):
        df["age_5"] = (df["age"] // 5) * 5
        df["age_10"] = (df["age"] // 10) * 10

    # 01: Age distribution (population-weighted HTS and census).
    grouped_bars(
        {"Census": shares(census, "age_class", "weight"),
         "IPU": shares(ipu, "age_5"),
         "HTS": shares(hts, "age_5", "person_weight")},
        "Population age distribution", "Age group (5-year start)", folder,
        "01_age_distribution.png",
        categories=sorted(set(census["age_class"].dropna()) | set(ipu["age_5"]) | set(hts["age_5"])),
    )

    # 02: Sex distribution.
    grouped_bars(
        {"Census": shares(census, "sex", "weight"),
         "IPU": shares(ipu, "sex"),
         "HTS": shares(hts, "sex", "person_weight")},
        "Sex distribution", "Sex", folder, "02_sex_distribution.png",
    )

    # 03: IPU population age x sex heatmap.
    age_sex = pd.crosstab(ipu["sex"], ipu["age_5"])
    age_sex = age_sex / age_sex.to_numpy().sum() * 100
    fig, ax = plt.subplots(figsize=(12, 4.4))
    im = ax.imshow(age_sex, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(age_sex.index)), age_sex.index.astype(str))
    ax.set_xticks(range(len(age_sex.columns)), [str(i) for i in age_sex.columns], rotation=45)
    ax.set_xlabel("Age group (5-year start)")
    ax.set_ylabel("Sex")
    ax.set_title("IPU population: age × sex", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Share of all IPU persons (%)")
    save(fig, folder, "03_age_sex_heatmap.png")

    # 04–07: Employment / driving licence by age, overall and by sex.
    age_rate_plot(ipu, hts, "employed", "Employment rate by age", folder, "04_employment_by_age.png")
    age_sex_rate_plot(ipu, hts, "employed", "Employment rate by age and sex", folder, "05_employment_by_age_sex.png")
    age_rate_plot(ipu, hts, "has_license", "Driving licence ownership by age", folder, "06_license_by_age.png")
    age_sex_rate_plot(ipu, hts, "has_license", "Driving licence ownership by age and sex", folder, "07_license_by_age_sex.png")

    # 08–09: Employment and driving licence rates by sex.
    sex_rate_plot(ipu, hts, "employed", "Employment rate by sex", folder, "08_employment_by_sex.png")
    sex_rate_plot(ipu, hts, "has_license", "Driving licence ownership by sex", folder, "09_license_by_sex.png")

    # 10: Driving licence rate depending on employment status.
    licence_by_employment = {
        "IPU": rates(ipu, "has_license", "employed"),
        "HTS": rates(hts, "has_license", "employed", "person_weight"),
    }
    grouped_bars(licence_by_employment,
                 "Driving licence ownership by employment status", "Employment status",
                 folder, "10_license_by_employment.png",
                 categories=[0, 1], labels=["Not employed", "Employed"])

    # One record per synthetic household for household-level statistics.
    ipu_hh = ipu.drop_duplicates("household_id")

    # 11: Household size, with 5+ as the last category.
    ipu_size = ipu.groupby("household_id").size().clip(upper=5)
    hts_size = hts_hh.assign(household_size=hts_hh["household_size"].clip(upper=5))
    grouped_bars(
        {"IPU": ipu_size.value_counts(normalize=True).mul(100),
         "HTS": shares(hts_size, "household_size", "household_weight")},
        "Household size distribution", "People per household", folder,
        "11_household_size.png", categories=[1, 2, 3, 4, 5],
        labels=["1", "2", "3", "4", "5+"],
    )

    # 12: Household income class (one IPU row per household).
    grouped_bars(
        {"Census": shares(census_income, "income_class", "weight"),
         "IPU": shares(ipu_hh, "income_class"),
         "HTS": shares(hts_hh, "income_class", "household_weight")},
        "Household income class", "Income class", folder,
        "12_household_income.png",
    )

    # 13: Number of employed people in IPU households (4+ pooled).
    employed_per_hh = ipu.groupby("household_id")["employed"].sum().clip(upper=4)
    grouped_bars(
        {"IPU": employed_per_hh.value_counts(normalize=True).mul(100)},
        "Employed persons per synthetic household", "Employed persons", folder,
        "13_employed_per_household.png", categories=[0, 1, 2, 3, 4],
        labels=["0", "1", "2", "3", "4+"],
    )

    # 14: Number of children younger than 18 in IPU households (4+ pooled).
    children_per_hh = ipu.assign(child=ipu["age"] < 18).groupby("household_id")["child"].sum().clip(upper=4)
    grouped_bars(
        {"IPU": children_per_hh.value_counts(normalize=True).mul(100)},
        "Children per synthetic household", "Children younger than 18", folder,
        "14_children_per_household.png", categories=[0, 1, 2, 3, 4],
        labels=["0", "1", "2", "3", "4+"],
    )
