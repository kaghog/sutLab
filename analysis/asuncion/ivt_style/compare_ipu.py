import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def configure(context):
    context.stage("asuncion.data.census.population")
    context.stage("asuncion.ipu.attributed")
    context.config("analysis_path")
    context.stage("asuncion.data.hts.entd.filtered")
    context.stage("asuncion.data.census.households")
    context.stage("asuncion.data.census.employment")


def plot_distribution(context, df1, df2, column, weight1, label1, label2, title, filename):
    d1 = df1.groupby(column)[weight1].sum() if weight1 else df1[column].value_counts()
    d2 = df2[column].value_counts()

    d1 = d1 / d1.sum() * 100
    d2 = d2 / d2.sum() * 100

    cats = sorted(set(d1.index) | set(d2.index), key=str)
    d1 = d1.reindex(cats, fill_value=0)
    d2 = d2.reindex(cats, fill_value=0)

    x = np.arange(len(cats))
    w = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(x - w / 2, d1, width=w, label=label1)
    plt.bar(x + w / 2, d2, width=w, label=label2)
    plt.xticks(x, cats, rotation=45, ha="right")
    plt.xlabel(column)
    plt.ylabel("Population share (%)")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{context.config('analysis_path')}/{filename}")
    plt.close()


def plot_population(context):
    census_population = context.stage("asuncion.data.census.population").copy()
    census_employment = context.stage("asuncion.data.census.employment").copy()

    ipu = context.stage("asuncion.ipu.attributed").copy()
    _, hts, _ = context.stage("asuncion.data.hts.entd.filtered")

    assert ipu["person_id"].is_unique

    # Age
    census_population_age = census_population.groupby("age_class", as_index=False)["weight"].sum()
    census_population_age["weight"] = census_population_age["weight"] / census_population_age["weight"].sum() * 100

    ipu["age_class"] = (ipu["age"] // 5) * 5
    hts["age_class"] = (hts["age"] // 5) * 5

    ipu_age = ipu.groupby("age_class").size()
    ipu_age = ipu_age / ipu_age.sum() * 100

    hts_age = hts.groupby("age_class")["person_weight"].sum()
    hts_age = hts_age / hts_age.sum() * 100

    bins = sorted(set(census_population_age["age_class"]) | set(ipu_age.index) | set(hts_age.index))
    x = np.arange(len(bins))
    w = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x - w, census_population_age.set_index("age_class")["weight"].reindex(bins, fill_value=0), width=w, label="Census")
    plt.bar(x, ipu_age.reindex(bins, fill_value=0), width=w, label="IPU synthetic")
    plt.bar(x + w, hts_age.reindex(bins, fill_value=0), width=w, label="HTS output")
    plt.xticks(x, bins, rotation=45)
    plt.xlabel("Age class (5-year bins)")
    plt.ylabel("Population share (%)")
    plt.title("Age distribution comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{context.config('analysis_path')}/census_vs_synthesis_ages.png")
    plt.close()

    # Employment: Census vs IPU
    census_employment["employed"] = True
    census_unemployment = pd.DataFrame({"employed": [False], "weight": [census_population["weight"].sum() - census_employment["weight"].sum()]})
    census_employment = pd.concat([census_employment, census_unemployment])

    plot_distribution(
        context, census_employment, ipu, "employed", "weight",
        "Census", "IPU synthetic",
        "Employment comparison",
        "census_vs_synthesis_employment.png",
    )
    plot_distribution(
        context, census_employment, hts, "employed", "weight",
        "Census", "HTS",
        "Employment comparison",
        "census_vs_hts_employment.png",
    )


    # Sex: Census vs IPU
    plot_distribution(
        context, census_population, ipu, "sex", "weight",
        "Census", "IPU synthetic",
        "Sex distribution comparison",
        "census_vs_synthesis_sex.png",
    )

    # Driving license: HTS vs IPU
    plot_distribution(
        context, hts, ipu, "has_license", "person_weight",
        "HTS", "IPU synthetic",
        "Driving license comparison",
        "hts_vs_synthesis_driving_license.png",
    )


def plot_household(context):
    census = context.stage("asuncion.data.census.households").copy()
    ipu = context.stage("asuncion.ipu.attributed").copy()
    hts, _, _ = context.stage("asuncion.data.hts.entd.filtered")

    ipu_size = ipu.groupby("household_id").size().clip(upper=5)
    ipu_dist = ipu_size.value_counts().sort_index()
    ipu_dist = ipu_dist / ipu_dist.sum() * 100

    hts["household_size"] = hts["household_size"].clip(upper=5)
    hts_dist = hts.groupby("household_size")["household_weight"].sum().sort_index()
    hts_dist = hts_dist / hts_dist.sum() * 100

    bins = [1, 2, 3, 4, 5]
    labels = ["1", "2", "3", "4", "5+"]

    x = np.arange(len(bins))
    w = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x, ipu_dist.reindex(bins, fill_value=0), width=w, label="IPU synthetic")
    plt.bar(x + w, hts_dist.reindex(bins, fill_value=0), width=w, label="HTS")
    plt.xticks(x, labels)
    plt.xlabel("Household size (persons)")
    plt.ylabel("Household share (%)")
    plt.title("Household size distribution comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{context.config('analysis_path')}/census_vs_synthesis_households.png")
    plt.close()


def execute(context):
    plot_population(context)
    plot_household(context)
