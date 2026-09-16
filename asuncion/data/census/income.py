import pandas as pd
import numpy as np
import os



def configure(context):
    context.config("data_path")
    context.config("asuncion.income_survey", "census/income/INGREFAM_EPHC_ANUAL_2022.csv")
    context.stage("asuncion.data.census.households")

def weighted_quantile(values, weights, quantiles):
    x, w = np.asarray(values), np.asarray(weights)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    o = np.argsort(x)
    x, w = x[o], w[o] 
    return np.interp(quantiles, np.cumsum(w) / np.sum(w), x) 
def weighted_mean(values, weights): 
    return np.average(values, weights=weights)


#ASUNCION_INCOME_CLASS_BINS = [-np.inf, np.inf]
#ASUNCION_INCOME_CLASS_LABELS = ["all"]

ASUNCION_INCOME_CLASS_BINS = [-np.inf, 2000, 4000, 6000, 8000, np.inf]
ASUNCION_INCOME_CLASS_LABELS = ["0-2m", "2-4m", "4-6m", "6-8m", "8m+"]


def execute(context):
    # ------------------------------------------------------------------
    # 1. EPHC: load household income data
    # ------------------------------------------------------------------
    path = f"{context.config('data_path')}/{context.config('asuncion.income_survey')}"
    cols = {
        "dpto":"departement_id", 
        "ingrem":"income", 
        "fex.2022":"weight", 
        "totpers":"person_count", 
        "v19ede":"rent",
        "area": "urban_type",
        "e01aimde": "primary_job_income",
        "e01bimde": "secondary_job_income",
        "e01cimde": "other_job_income"
    }
    
    df = pd.read_csv(path, sep=";", dtype=str)[list(cols)].rename(columns=cols)

    df["departement_id"] = df["departement_id"].replace({"0":"00"})
    df = df[df["departement_id"].isin(["00", "11"])].copy()
    df = df[df["urban_type"] == "1"].copy()  # only urban households, because AMA area covers mostly only urban areas

    for c in ["weight", "person_count", "primary_job_income", "secondary_job_income", "other_job_income"]:
        df[c] = df[c].str.replace(",", ".", regex=False).astype(float)

    df["income"] = (df["primary_job_income"] + df["secondary_job_income"] + df["other_job_income"]) / 1000
    df = df[(df["weight"] > 0) & (df["person_count"] > 0)].copy()



    # ------------------------------------------------------------------
    # 2. EPHC: define income classes
    # ------------------------------------------------------------------
    bins = ASUNCION_INCOME_CLASS_BINS
    labels = ASUNCION_INCOME_CLASS_LABELS
    df["income_class"] = pd.cut(df["income"], bins=bins, labels=labels, right=False)

    # ------------------------------------------------------------------
    # 3. EPHC: weighted income shares, separately by department
    # ------------------------------------------------------------------
    weighted = df.groupby(["departement_id", "income_class"], observed=False)["weight"].sum()
    totals = df.groupby("departement_id")["weight"].sum()
    shares = (weighted / totals).reset_index(name="share")

    # ------------------------------------------------------------------
    # 4. Census: total number of households by department
    # ------------------------------------------------------------------
    census = context.stage("asuncion.data.census.households").copy()
    census_totals = census.groupby(["commune_id", "departement_id"], as_index=False)["weight"].sum()
    census_totals = census_totals.rename(columns={"weight":"census_households"})

    # ------------------------------------------------------------------
    # 5. Combine EPHC shares with Census household totals
    # ------------------------------------------------------------------
    targets = shares.merge(census_totals[["departement_id", "commune_id", "census_households"]], on="departement_id", how="inner")
    targets["weight"] = targets["share"] * targets["census_households"]



    targets["commune_id"] = targets["commune_id"]

    print(targets[["income_class", "weight", "commune_id"]])

    return targets[["departement_id", "commune_id", "income_class", "share", "census_households", "weight"]]

def validate(context):
    path = f"{context.config('data_path')}/{context.config('asuncion.income_survey')}"
    if not os.path.exists(path):
        raise RuntimeError(f"Data is not available at location {path}")
    return [os.path.getsize(path)]
