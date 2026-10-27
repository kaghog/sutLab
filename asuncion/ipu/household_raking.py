"""One-weight-per-household generalized raking against person and household margins.

Each margin is a household contribution count. The solution minimizes deviation
from starting household expansion factors, while approximately matching margins.
Infeasible targets are reported rather than silently skipped.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix


def fit_household_weights(df, census_targets, household_id_col="household_id",
                          household_vars=("household_category",),
                          initial_weight_col="household_weight", max_iterations=1000,
                          tolerance=1e-5, verbose=False):
    if df.empty:
        raise ValueError("Cannot calibrate an empty population")
    if household_id_col not in df or initial_weight_col not in df:
        raise ValueError("Missing household identifier or initial household weight")
    if df[household_id_col].isna().any():
        raise ValueError("Null household identifiers")

    df = df.copy()
    group_indices, households = pd.factorize(df[household_id_col], sort=False)
    H = len(households)
    n_people = len(df)
    seed = pd.to_numeric(df[initial_weight_col], errors="coerce").to_numpy(float)
    if not np.isfinite(seed).all() or (seed <= 0).any():
        raise ValueError("Initial household expansion weights must be finite and positive")
    hh_weight_min = (np.bincount(group_indices, weights=seed, minlength=H) /
                     np.bincount(group_indices, minlength=H))
    # Ensure each household has only one initial survey weight. Do not choose
    # the first person and silently ignore inconsistent records.
    if np.any(abs(seed - hh_weight_min[group_indices]) > 1e-6 * np.maximum(1, seed)):
        raise ValueError("Person records from one household have differing seed weights")
    w0 = hh_weight_min
    # Normalize the survey expansion factors to the target sampled household count.
    hh_targets = [census_targets[v] for v in household_vars if v in census_targets]
    if hh_targets:
        total_hh = float(sum(next(iter(hh_targets)).values()))
        if total_hh > 0:
            w0 = w0 * (total_hh / w0.sum())

    rows, targets, descriptions = [], [], []
    hh_sizes = np.bincount(group_indices, minlength=H)

    for var, categories in census_targets.items():
        if var in ("total_households", "total_population"):
            vals = categories.values() if isinstance(categories, dict) else [categories]
            values = list(vals)
            if len(values) != 1:
                raise ValueError(f"{var} must contain exactly one target")
            row = np.ones(H) if var == "total_households" else hh_sizes.astype(float)
            target = float(values[0])
            if np.isfinite(target) and target >= 0:
                rows.append(row)
                targets.append(target)
                descriptions.append((var, "total"))
            continue
        if var not in df:
            raise ValueError(f"Margin variable {var!r} is missing")
        if not isinstance(categories, dict):
            raise ValueError(f"Margin {var!r} must map categories to target counts")
        values = df[var].tolist()
        if var in household_vars:
            # Require household-level categories to be constant across members.
            first_by_hh = {}
            for h, cat in zip(group_indices, values):
                if h in first_by_hh and first_by_hh[h] != cat:
                    raise ValueError(f"Household-level variable {var} differs within a household")
                first_by_hh[h] = cat
            values_by_hh = [first_by_hh[h] for h in range(H)]
        for category, value in categories.items():
            target = float(value)
            if not np.isfinite(target) or target < 0:
                raise ValueError(f"Invalid target for {var}={category}: {value}")
            if var in household_vars:
                row = np.fromiter((v == category for v in values_by_hh), dtype=float, count=H)
            else:
                match = np.fromiter((v == category for v in values), dtype=bool, count=n_people)
                row = np.bincount(group_indices[match], minlength=H).astype(float)
            if row.sum() == 0:
                if target > 0:
                    raise ValueError(f"Positive census target has zero seed coverage: {var}={category}: {target}")
                continue
            rows.append(row)
            targets.append(target)
            descriptions.append((var, category))

    if not rows:
        raise ValueError("No valid margins to calibrate")
    A = csr_matrix(np.stack(rows, axis=0))
    t = np.asarray(targets, dtype=float)
    # Soft-margin entropy projection. The >0 penalty makes infeasible systems
    # well-posed while keeping the main goal close to exact census margins.
    # A penalty with units of people is added to the dual objective.
    slack = np.maximum(0.10, 0.0005 * t)
    penalty = slack ** 2

    def f_and_g(lam):
        log_ratio = A.T @ lam
        # Clip only for numerical safety; reasonable solutions stay far away.
        weight = w0 * np.exp(np.clip(log_ratio, -30, 30))
        obj = float(weight.sum() - np.dot(t, lam) + .5 * np.dot(penalty, lam * lam))
        grad = np.asarray(A @ weight - t + penalty * lam).reshape(-1)
        return obj, grad

    result = minimize(f_and_g, np.zeros(len(t)), method="L-BFGS-B", jac=True,
                      options={"maxiter":int(max_iterations),"ftol":1e-13,
                               "gtol": min(float(tolerance), 1e-6),"maxls":40})
    w = w0 * np.exp(np.clip(A.T @ result.x, -30, 30))
    predicted = np.asarray(A @ w).reshape(-1)
    abs_error = np.abs(predicted-t)
    # Scale by true target, with floor to avoid huge relative percentages at zero.
    rel_error = abs_error / np.maximum(1, t)
    if verbose:
        print(f"Household raking: success={result.success}, optimizer={result.message}")
        for order in np.argsort(-abs_error)[:10]:
            print(f"  {descriptions[order]} target={t[order]:.2f} "
                  f"fit={predicted[order]:.2f} error={abs_error[order]:.2f}")
        print(f"Max abs residual: {abs_error.max():.2f}, "
              f"mean abs residual: {abs_error.mean():.2f}")
    if not result.success:
        warnings.warn(f"Household calibration optimizer did not converge: {result.message}")
    if np.any(rel_error > 0.05):
        bad = np.argsort(-rel_error)[:5]
        warnings.warn("Household calibration residual >5% for some margins: "
                      + "; ".join(f"{descriptions[k]} target={t[k]:.2f} fit={predicted[k]:.2f}"
                                  for k in bad if rel_error[k] > 0.05))
    df["weight"] = w[group_indices]
    return df
