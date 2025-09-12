import synthesis.population.enriched as delegate

import pandas as pd
import numpy as np


def configure(context):
    # Base population with donor attributes
    delegate.configure(context)
    # PT constraints derived from Hannover HTS
    context.stage("hannover.data.pt.constraints")
    # License constraints derived from Hannover HTS
    context.stage("hannover.data.license.constraints")
    context.config("random_seed")


def execute(context):
    # Start from base enriched population (donor copy)
    df_persons = delegate.execute(context)

    # Load Hannover-specific constraints
    data_pt = context.stage("hannover.data.pt.constraints")
    constraints_pt = data_pt["pt_subscription_constraints"]
    data_lic = context.stage("hannover.data.license.constraints")
    constraints_lic = data_lic["license_constraints"]

    # Prepare IPF-style proportional fitting using numpy arrays
    df_persons = df_persons.copy()
    n = len(df_persons)
    prob = np.ones(n, dtype=float)
    sex_arr = df_persons["sex"].astype(str).to_numpy()
    age_arr = df_persons["age"].to_numpy()

    # Build integer index arrays and targets for PT calibration
    pt_index_groups = []
    targets = []

    for c in constraints_pt:
        f = np.ones(n, dtype=bool)
        if "sex" in c:
            f &= (sex_arr == c["sex"])  # category-safe
        if "age" in c:
            low, high = c["age"]
            f &= (age_arr >= low) & (age_arr <= high)
        idx = np.flatnonzero(f)
        pt_index_groups.append(idx)
        targets.append(float(c["target"]) * int(idx.size))

    iterations = 500
    for _ in context.progress(range(iterations), label="imputing pt subscription (Hannover)"):
        for idx, target in zip(pt_index_groups, targets):
            current = float(prob[idx].sum())
            if current == 0 or target == 0:
                factor = 0.0
            else:
                factor = target / current
            prob[idx] = prob[idx] * float(factor)

    # Sample booleans using random seed for determinism
    rng = np.random.RandomState(context.config("random_seed") + 92341)
    u = rng.random_sample(n)
    sel = u < np.clip(prob, 0.0, 1.0)
    # Enforce 0% PT subscription for children <6
    sel[age_arr < 6] = False
    df_persons["has_pt_subscription"] = sel

    # Optional deterministic micro-adjustment for the 15–17 age×sex bins
    # to better match tight targets without affecting other groups.
    # We only adjust if such constraints exist.
    # Identify targets
    targets_1517 = {}
    for c in constraints_pt:
        if c.get("age") == (15, 17) and "sex" in c:
            targets_1517[c["sex"]] = float(c["target"])

    if targets_1517:
        # Work on a copy to avoid chained assignment issues
        for sex, target_share in targets_1517.items():
            mask = (df_persons["age"].between(15, 17)) & (df_persons["sex"].astype(str) == sex)
            subgroup = df_persons.loc[mask, ["has_pt_subscription"]].copy()
            n_sub = len(subgroup)
            if n_sub == 0:
                continue
            desired = int(round(target_share * n_sub))
            current = int(subgroup["has_pt_subscription"].sum())
            delta = desired - current
            if delta == 0:
                continue
            # Use RNG with a stable seed offset for reproducibility
            rng2 = np.random.RandomState(context.config("random_seed") + (1123 if sex == "male" else 2246))
            idx = df_persons.loc[mask].index.to_numpy()
            rng2.shuffle(idx)
            if delta > 0:
                # Flip up some False to True
                candidates = df_persons.loc[idx, :]
                to_flip = candidates.index[~candidates["has_pt_subscription"]][:delta]
                df_persons.loc[to_flip, "has_pt_subscription"] = True
            else:
                # Flip down some True to False
                candidates = df_persons.loc[idx, :]
                to_flip = candidates.index[candidates["has_pt_subscription"]][: -delta]
                df_persons.loc[to_flip, "has_pt_subscription"] = False

    # --- Driving license calibration (overall + sex + age + teens-sex) ---
    lic_prob = np.ones(n, dtype=float)
    lic_index_groups = []
    lic_targets = []
    for c in constraints_lic:
        f = np.ones(n, dtype=bool)
        if "sex" in c:
            f &= (sex_arr == c["sex"])  # category-safe
        if "age" in c:
            low, high = c["age"]
            f &= (age_arr >= low) & (age_arr <= high)
        idx = np.flatnonzero(f)
        lic_index_groups.append(idx)
        lic_targets.append(float(c["target"]) * int(idx.size))

    for _ in range(200):
        for idx, target in zip(lic_index_groups, lic_targets):
            current = float(lic_prob[idx].sum())
            if current == 0 or target == 0:
                factor = 0.0
            else:
                factor = target / current
            lic_prob[idx] = lic_prob[idx] * float(factor)

    rng3 = np.random.RandomState(context.config("random_seed") + 45421)
    u3 = rng3.random_sample(n)
    has_license = u3 < np.clip(lic_prob, 0.0, 1.0)

    # Deterministic micro-adjustment for 15–17 by sex
    teen_targets = {}
    for c in constraints_lic:
        if c.get("age") == (15, 17) and "sex" in c:
            teen_targets[c["sex"]] = float(c["target"])
    for sex in teen_targets:
        mask = (age_arr >= 15) & (age_arr <= 17) & (sex_arr == sex)
        pos = np.flatnonzero(mask)
        if pos.size == 0:
            continue
        desired = int(round(teen_targets[sex] * int(pos.size)))
        current = int(has_license[pos].sum())
        delta = desired - current
        if delta != 0:
            rngx = np.random.RandomState(context.config("random_seed") + (6611 if sex == "male" else 7722))
            perm = pos.copy()
            rngx.shuffle(perm)
            if delta > 0:
                to_flip = [i for i in perm if not has_license[i]][:delta]
                if to_flip:
                    has_license[np.array(to_flip, dtype=int)] = True
            else:
                to_flip = [i for i in perm if has_license[i]][:-delta]
                if to_flip:
                    has_license[np.array(to_flip, dtype=int)] = False

    # Write back calibrated license
    df_persons["has_license"] = has_license

    return df_persons
