"""
income validation and comparison for Asunción HTS and synthetic population

  - 2x3 income-shape grids (ALL / Asunción / Central) for job-only and total
  - compact comparison tables rendered both as CSV and figure tables
  - income-class barplots with embedded comparison tables
  - synthetic-vs-HTS and Asunción-vs-Central summary tables
  - missing-income diagnostics
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from asuncion.data.census.income import (
    ASUNCION_INCOME_CLASS_BINS,
    ASUNCION_INCOME_CLASS_LABELS,
)
from asuncion.data.hts.entd.fix_income import JOB_COLUMNS, INCOME_COLUMNS, MISSING_CODES

DEPARTMENTS = {"00": "Asunción", "11": "Central"}
GROUPS = ("ALL", "00", "11")
INCOME_TYPES = ("job_only", "total_income")
SOURCE_ORDER = ("EPHC 2022", "HTS raw", "HTS complete cases", "HTS repaired", "Synthetic")
COLORS = {
    "EPHC 2022": "#1F385A",
    "HTS raw": "#C36B36",
    "HTS complete cases": "#BC9E2C",
    "HTS repaired": "#19898D",
    "Synthetic": "#A53D65",
}
CLASS_BINS_M = np.asarray(ASUNCION_INCOME_CLASS_BINS, dtype=float) / 1000.0
CLASS_LABELS = list(ASUNCION_INCOME_CLASS_LABELS)


def configure(context):
    context.stage("synthesis.output")
    context.stage("data.hts.entd.raw")
    context.stage("asuncion.data.hts.entd.cleaned")
    for key in ("data_path", "output_path", "output_prefix", "analysis_path"):
        context.config(key)
    context.config("asuncion.income_survey", "census/income/INGREFAM_EPHC_ANUAL_2022.csv")
    context.config("income_type", "job_only")
    context.config("asuncion.income_analysis_synthetic_scale_pyg", 1000.0)


def _require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _number(col):
    return pd.to_numeric(
        col.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).astype(float)


def _dept(col):
    return col.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(2)


def _safe_weighted_quantile(values, weights, probs):
    x = np.asarray(values, float)
    w = np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[good], w[good]
    if not len(x):
        return np.repeat(np.nan, len(probs))
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cutoff = np.cumsum(w) / w.sum()
    return x[np.minimum(np.searchsorted(cutoff, probs, side="left"), len(x) - 1)]


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "pdf.fonttype": 42,
    })
    return plt


def _save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=230, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _frame(source, household, job, total, weight, department, active=None):
    result = pd.DataFrame({
        "source": source,
        "departement_id": _dept(household[department]).to_numpy(),
        "weight": _number(household[weight]).to_numpy(),
        "job_only": _number(household[job]).to_numpy() / 1e6,
        "total_income": _number(household[total]).to_numpy() / 1e6,
    })
    if source == "Synthetic":
        inactive = "total_income" if active == "job_only" else "job_only"
        result[inactive] = np.nan
    return result


def _observed_hts(raw, cleaned):
    pr, hr, *_ = raw
    hc, pc, *_ = cleaned
    _require(hr, ["household_id", "departement_id", "household_weight", "household_income"], "raw HTS households")
    _require(pr, ["household_id"] + list(INCOME_COLUMNS), "raw HTS persons")
    _require(hc, ["household_id", "job_income_pyg", "total_income_pyg", "is_total_income_complete"], "cleaned HTS households")
    hr = hr.copy(); pr = pr.copy(); hc = hc.copy()
    for frame in (hr, pr, hc):
        frame["household_id"] = frame["household_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if hr.household_id.duplicated().any() or hc.household_id.duplicated().any():
        raise ValueError("Raw or cleaned HTS contains duplicate household IDs")
    if len(hr) != len(hc) or set(hr.household_id) != set(hc.household_id):
        raise ValueError("Raw HTS households and cleaned HTS households differ; income audit requires the full cleaned sample")

    mon = pr[INCOME_COLUMNS].apply(_number)
    special = mon.isin(MISSING_CODES)
    nonjob_cols = [c for c in INCOME_COLUMNS if c not in JOB_COLUMNS]
    flags = pd.DataFrame({
        "household_id": pr["household_id"],
        "any_unknown": special.any(axis=1),
        "job_unknown": special[JOB_COLUMNS].any(axis=1),
        "nonjob_unknown": special[nonjob_cols].any(axis=1),
    }).groupby("household_id", sort=False).any()

    raw_job = mon[JOB_COLUMNS].mask(special[JOB_COLUMNS]).fillna(0).sum(axis=1).groupby(pr.household_id, sort=False).sum()
    hr["raw_job_pyg"] = hr.household_id.map(raw_job)
    hr["raw_total_pyg"] = _number(hr.household_income)
    hr["any_unknown"] = hr.household_id.map(flags.any_unknown).fillna(False).astype(bool)
    hr["job_unknown"] = hr.household_id.map(flags.job_unknown).fillna(False).astype(bool)
    hr["nonjob_unknown"] = hr.household_id.map(flags.nonjob_unknown).fillna(False).astype(bool)

    merged = hr.merge(
        hc[["household_id", "job_income_pyg", "total_income_pyg", "is_total_income_complete"]],
        on="household_id", how="left", validate="one_to_one",
    )
    merged["fixed_job_pyg"] = _number(merged.job_income_pyg)
    merged["fixed_total_pyg"] = _number(merged.total_income_pyg)
    # Estimated total is now numeric: unresolved components contributed zero.
    # Preserve the pre-zero-fill completeness flag, rather than inferring
    # completeness from total_income_pyg.notna().
    flag = merged["is_total_income_complete"].astype("string").str.strip().str.lower()
    if not flag.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Cleaned is_total_income_complete must contain nonmissing booleans")
    merged["is_total_income_complete"] = flag.isin(["true", "1"])
    if (merged.nonjob_unknown & merged.is_total_income_complete).any():
        raise ValueError("Cleaned total marked complete despite a source unknown non-job component")
    if merged.fixed_job_pyg.isna().any():
        warnings.warn(f"Repaired job income unresolved in {merged.fixed_job_pyg.isna().sum()} HTS households", stacklevel=2)

    frames = [
        _frame("HTS raw", merged, "raw_job_pyg", "raw_total_pyg", "household_weight", "departement_id"),
        _frame("HTS complete cases", merged.loc[~merged.any_unknown], "raw_job_pyg", "raw_total_pyg", "household_weight", "departement_id"),
        _frame("HTS repaired", merged, "fixed_job_pyg", "fixed_total_pyg", "household_weight", "departement_id"),
    ]
    missing = {
        "raw_households": int(len(merged)),
        "complete_households": int((~merged.any_unknown).sum()),
        "unknown_any_households": int(merged.any_unknown.sum()),
        "unknown_job_households": int(merged.job_unknown.sum()),
        "unknown_nonjob_households": int(merged.nonjob_unknown.sum()),
        "unknown_both_households": int((merged.job_unknown & merged.nonjob_unknown).sum()),
        "weighted_unknown_any_share_pct": float(100 * _number(merged.loc[merged.any_unknown, "household_weight"]).sum() / _number(merged.household_weight).sum()),
        "fixed_job_missing_households": int(merged.fixed_job_pyg.isna().sum()),
        "fixed_total_missing_households": int(merged.fixed_total_pyg.isna().sum()),
        "fixed_total_incomplete_households": int((~merged.is_total_income_complete).sum()),
        "weighted_fixed_total_incomplete_share_pct": float(100 * _number(merged.loc[~merged.is_total_income_complete, "household_weight"]).sum() / _number(merged.household_weight).sum()),
    }
    diag = merged[["household_id", "departement_id", "household_weight", "any_unknown", "job_unknown", "nonjob_unknown", "raw_job_pyg", "fixed_job_pyg", "raw_total_pyg", "fixed_total_pyg", "is_total_income_complete"]]
    return frames, missing, diag


def _ephc(path):
    data = pd.read_csv(path, sep=";", dtype="string")
    fields = ["dpto", "area", "fex.2022", "ingrem", "e01aimde", "e01bimde", "e01cimde"]
    _require(data, fields, "EPHC INGREFAM")
    data["dpto"] = _dept(data.dpto)
    data = data[data.dpto.isin(DEPARTMENTS) & data.area.eq("1")].copy()
    if data.empty:
        raise ValueError("EPHC has no Asunción/Central urban households")
    for field in fields[2:]:
        data[field] = _number(data[field])
    data["ephc_job_pyg"] = data[["e01aimde", "e01bimde", "e01cimde"]].fillna(0).sum(axis=1)
    data["ephc_total_pyg"] = data.ingrem
    return _frame("EPHC 2022", data, "ephc_job_pyg", "ephc_total_pyg", "fex.2022", "dpto")


def _synthetic(output_path, prefix, active, scale):
    path = Path(output_path) / f"{prefix}households.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing synthesis.output household CSV: {path}")
    data = pd.read_csv(path, sep=";", dtype="string", low_memory=False)
    _require(data, ["household_id", "departement_id", "income"], "synthetic households")
    income = _number(data.income) * float(scale)
    data["__selected_income_pyg"] = income
    data["__other_income_pyg"] = np.nan
    data["__unit_weight"] = 1.0
    frame = _frame(
        "Synthetic", data,
        "__selected_income_pyg" if active == "job_only" else "__other_income_pyg",
        "__selected_income_pyg" if active == "total_income" else "__other_income_pyg",
        "__unit_weight", "departement_id", active=active,
    )
    return frame, path


def _stats(data):
    rows, classes = [], []
    for mode in INCOME_TYPES:
        for dep in GROUPS:
            for source in SOURCE_ORDER:
                rowsrc = data.loc[data.source.eq(source)]
                if dep != "ALL":
                    rowsrc = rowsrc.loc[rowsrc.departement_id.eq(dep)]
                if source == "Synthetic" and rowsrc[mode].isna().all():
                    continue
                v = rowsrc[mode].to_numpy(dtype=float)
                w = rowsrc.weight.to_numpy(dtype=float)
                eligible = np.isfinite(v) & (v >= 0) & np.isfinite(w) & (w > 0)
                x, wt = v[eligible], w[eligible]
                counts = len(rowsrc)
                if not len(x):
                    rows.append(dict(income_type=mode, department=dep, source=source, households=counts, valid_households=0))
                    continue
                q05, q25, q50, q75, q90, q95, q99 = _safe_weighted_quantile(x, wt, [.05, .25, .5, .75, .9, .95, .99])
                rows.append(dict(
                    income_type=mode, department=dep, source=source,
                    households=counts, valid_households=len(x),
                    coverage_household_pct=100 * len(x) / counts if counts else np.nan,
                    coverage_weight_pct=100 * wt.sum() / w[np.isfinite(w) & (w > 0)].sum(),
                    mean_mpyg=float(np.average(x, weights=wt)),
                    median_mpyg=q50, p05_mpyg=q05, p25_mpyg=q25, p75_mpyg=q75,
                    p90_mpyg=q90, p95_mpyg=q95, p99_mpyg=q99,
                    zero_pct=float(100 * wt[x == 0].sum() / wt.sum()),
                    over_8m_pct=float(100 * wt[x >= 8].sum() / wt.sum()),
                    over_20m_pct=float(100 * wt[x >= 20].sum() / wt.sum()),
                    weight_sum=float(wt.sum()),
                ))
                for i, label in enumerate(CLASS_LABELS):
                    share = wt[(x >= CLASS_BINS_M[i]) & (x < CLASS_BINS_M[i + 1])].sum() / wt.sum() * 100
                    classes.append(dict(income_type=mode, department=dep, source=source, income_class=label, share_pct=share))
    return pd.DataFrame(rows), pd.DataFrame(classes)


def _display_sources(mode, active):
    return [s for s in SOURCE_ORDER if s != "Synthetic" or mode == active]


def _arrays(data, source, mode, dep):
    d = data.loc[data.source.eq(source)]
    if dep != "ALL":
        d = d.loc[d.departement_id.eq(dep)]
    x = d[mode].to_numpy(float)
    w = d.weight.to_numpy(float)
    good = np.isfinite(x) & (x >= 0) & np.isfinite(w) & (w > 0)
    return x[good], w[good]


def _cdf(x, w, grid):
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    idx = np.searchsorted(xs, grid, side="right") - 1
    cum = np.cumsum(ws) / ws.sum()
    return np.where(idx >= 0, cum[np.maximum(idx, 0)], 0.0)


def _subtitle(mode, active):
    if mode == active:
        return f"ACTIVE PIPELINE SCENARIO: {mode} · synthetic population included"
    return f"REFERENCE-ONLY SCENARIO: {mode} · synthetic population not shown"


def _figure_table(df, title, outstem, font_size=10, col_width=1.0):
    plt = _plotting()
    rows = len(df) + 1
    cols = len(df.columns)
    fig_h = max(2.2, 0.55 * rows)
    fig_w = max(8.5, 2.1 * cols * col_width)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E9ECEF')
        if c == 0 and r > 0:
            cell.set_text_props(weight='bold')
    ax.set_title(title, loc='left', weight='bold', fontsize=12)
    _save(fig, outstem)


def _shape_grid(data, mode, active, out):
    plt = _plotting()
    fig, axs = plt.subplots(2, 3, figsize=(17.5, 8.8), layout="constrained")
    cols = [("ALL", "Asunción + Central"), ("00", "Asunción"), ("11", "Central")]
    sources = _display_sources(mode, active)
    hist_edges = np.r_[np.arange(0, 21, 1), np.inf]
    for j, (dep, dep_title) in enumerate(cols):
        for s in sources:
            x, w = _arrays(data, s, mode, dep)
            if not len(x):
                continue
            color = COLORS[s]
            lw = 2.6 if s == "Synthetic" else 2.0
            label = s + (" [ACTIVE]" if s == "Synthetic" else "")
            counts, _ = np.histogram(x, bins=hist_edges, weights=w)
            hist_centers = np.r_[np.arange(0.5, 20.5, 1), 20.5]
            axs[0, j].step(hist_centers, 100 * counts / w.sum(), where='mid', color=color, lw=lw, label=label)
            xs = np.linspace(0, 20, 401)
            axs[1, j].step(xs, _cdf(x, w, xs), where='post', color=color, lw=lw, label=label)
        axs[0, j].set_title(dep_title, fontsize=16)
        axs[0, j].set_xlim(0, 21.2)
        axs[0, j].set_xticks([0, 5, 10, 15, 20.5], ["0", "5", "10", "15", "20+"])
        axs[0, j].set_xlabel("Household income (million PYG/month; 20+ open-ended)")
        axs[0, j].set_ylabel("Households per 1M bin (%)")
        axs[0, j].grid(alpha=.15)
        axs[1, j].set_xlim(0, 20)
        axs[1, j].set_ylim(0, 1)
        axs[1, j].set_xlabel("Household income (million PYG/month)")
        axs[1, j].set_ylabel("Cumulative share")
        axs[1, j].grid(alpha=.15)
    axs[0, 0].legend(loc='upper right', fontsize=8.5, frameon=False)
    fig.suptitle(
        f"{'Job-only' if mode == 'job_only' else 'Total household'} income: EPHC vs HTS vs synthetic\n{_subtitle(mode, active)}",
        fontsize=20, weight='bold', color='#A53D65' if mode == active else '#536474'
    )
    _save(fig, out / f"01_shape_grid_{mode}")


def _summary_table(stats, mode, active, out):
    sub = stats[(stats.income_type == mode) & (stats.department == "ALL") & (stats.source.isin(_display_sources(mode, active)))]
    sub = sub.set_index('source').reindex(_display_sources(mode, active)).reset_index()
    nice = pd.DataFrame({
        "Dataset": sub['source'],
        "Mean": sub['mean_mpyg'].map(lambda x: f"{x:.2f}"),
        "Median": sub['median_mpyg'].map(lambda x: f"{x:.2f}"),
        "Income ≥8M": sub['over_8m_pct'].map(lambda x: f"{x:.1f}%"),
        "Zero income": sub['zero_pct'].map(lambda x: f"{x:.1f}%"),
    })
    nice.to_csv(out / f"02_summary_table_{mode}.csv", index=False)
    nice.to_markdown(out / f"02_summary_table_{mode}.md", index=False)
    _figure_table(nice, f"Weighted mean monthly household {'job-only' if mode == 'job_only' else 'total'} income, million PYG\n{_subtitle(mode, active)}", out / f"02_summary_table_{mode}")


def _dept_means_table(stats, active, out):
    plt = _plotting()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), layout='constrained')
    for ax, mode in zip(axes, INCOME_TYPES):
        sub = stats[(stats.income_type == mode) & (stats.department.isin(GROUPS))]
        sources = [s for s in ["HTS raw", "HTS repaired", "EPHC 2022"] if s in set(sub.source)]
        rows = []
        for dep in ["00", "11", "ALL"]:
            row = {"Department": "Asunción" if dep == "00" else ("Central" if dep == "11" else "Combined")}
            for src in sources:
                q = sub[(sub.department == dep) & (sub.source == src)]
                row[src] = f"{q.iloc[0].mean_mpyg:.2f}" if not q.empty else ""
            rows.append(row)
        df = pd.DataFrame(rows)
        ax.axis('off')
        tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.35)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight='bold'); cell.set_facecolor('#E9ECEF')
            if c == 0 and r > 0:
                cell.set_text_props(weight='bold')
        ax.set_title(f"Weighted mean monthly household {'job-only' if mode == 'job_only' else 'total'} income, million PYG", loc='left', fontsize=12, weight='bold')
        df.to_csv(out / f"03_department_mean_table_{mode}.csv", index=False)
        df.to_markdown(out / f"03_department_mean_table_{mode}.md", index=False)
    fig.suptitle("Results of the repair: HTS raw, HTS repaired and EPHC 2022\nJob-only and total household income (million PYG)", fontsize=15, weight="bold")
    _save(fig, out / "03_department_mean_tables")


def _income_class_plot_with_table(classes, mode, active, out):
    plt = _plotting()
    sources = _display_sources(mode, active)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), gridspec_kw={"height_ratios": [3.1, 1.35]}, layout='constrained')
    dep_order = ["00", "11"]
    for j, dep in enumerate(dep_order):
        ax = axes[0, j]
        sub = classes[(classes.income_type == mode) & (classes.department == dep)]
        x = np.arange(len(CLASS_LABELS))
        width = 0.86 / len(sources)
        table_rows = []
        for i, src in enumerate(sources):
            part = sub[sub.source == src].set_index('income_class').reindex(CLASS_LABELS)
            vals = part['share_pct'].to_numpy(float)
            ax.bar(x + (i - (len(sources) - 1) / 2) * width, vals, width * .92, color=COLORS[src], label=src + (" [ACTIVE]" if src == "Synthetic" else ""))
            table_rows.append([src] + [f"{v:.1f}%" for v in vals])
        ax.set_xticks(x, CLASS_LABELS)
        ax.set_ylabel("Share of households (%)")
        ax.set_xlabel("Monthly income, million PYG")
        ax.set_title(f"{DEPARTMENTS[dep]} ({dep}) — income brackets")
        ax.grid(axis='y', alpha=.18)
        if j == 0:
            ax.legend(frameon=False, fontsize=8.5)
        ta = axes[1, j]
        ta.axis('off')
        tdf = pd.DataFrame(table_rows, columns=["Source"] + CLASS_LABELS)
        tdf["Source"] = tdf["Source"].replace({"HTS complete cases": "HTS complete", "HTS repaired": "HTS repaired"})
        tbl = ta.table(cellText=tdf.values, colLabels=tdf.columns,
                       colWidths=[0.25] + [0.15] * len(CLASS_LABELS),
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.35)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight='bold'); cell.set_facecolor('#E9ECEF')
    fig.suptitle(f"Income classes by department\n{_subtitle(mode, active)}", fontsize=18, weight='bold', color='#A53D65' if mode == active else '#536474')
    _save(fig, out / f"04_income_classes_with_tables_{mode}")


def _shape_grid_synthetic_vs_ephc(data, active, out):
    plt = _plotting()
    fig, axs = plt.subplots(2, 3, figsize=(17.5, 8.8), layout="constrained")
    cols = [("ALL", "Asunción + Central"), ("00", "Asunción"), ("11", "Central")]
    sources = ["EPHC 2022", "Synthetic"]
    hist_edges = np.r_[np.arange(0, 21, 1), np.inf]
    for j, (dep, dep_title) in enumerate(cols):
        for s in sources:
            x, w = _arrays(data, s, active, dep)
            if not len(x):
                continue
            color = COLORS[s]
            lw = 2.6 if s == "Synthetic" else 2.0
            label = s + (" [ACTIVE]" if s == "Synthetic" else "")
            counts, _ = np.histogram(x, bins=hist_edges, weights=w)
            hist_centers = np.r_[np.arange(0.5, 20.5, 1), 20.5]
            axs[0, j].step(hist_centers, 100 * counts / w.sum(), where='mid', color=color, lw=lw, label=label)
            xs = np.linspace(0, 20, 401)
            axs[1, j].step(xs, _cdf(x, w, xs), where='post', color=color, lw=lw, label=label)
        axs[0, j].set_title(dep_title, fontsize=16)
        axs[0, j].set_xlim(0, 21.2)
        axs[0, j].set_xticks([0, 5, 10, 15, 20.5], ["0", "5", "10", "15", "20+"])
        axs[0, j].set_xlabel("Household income (million PYG/month; 20+ open-ended)")
        axs[0, j].set_ylabel("Households per 1M bin (%)")
        axs[0, j].grid(alpha=.15)
        axs[1, j].set_xlim(0, 20)
        axs[1, j].set_ylim(0, 1)
        axs[1, j].set_xlabel("Household income (million PYG/month)")
        axs[1, j].set_ylabel("Cumulative share")
        axs[1, j].grid(alpha=.15)
    axs[0, 0].legend(loc='upper right', fontsize=8.5, frameon=False)
    fig.suptitle(
        f"{'Job-only' if active == 'job_only' else 'Total household'} income: EPHC vs synthetic population\nACTIVE PIPELINE SCENARIO: {active}",
        fontsize=20, weight='bold', color='#A53D65'
    )
    _save(fig, out / f"01b_shape_grid_ephc_vs_synthetic_{active}")

def _hts_vs_synthetic_table(stats, active, out):
    sub = stats[(stats.income_type == active) & (stats.department.isin(["00", "11"])) & (stats.source.isin(["HTS raw", "Synthetic"]))]
    # Build explicit table in requested style.
    h0 = sub[(sub.department == "00") & (sub.source == "HTS raw")].iloc[0]
    s0 = sub[(sub.department == "00") & (sub.source == "Synthetic")].iloc[0]
    h1 = sub[(sub.department == "11") & (sub.source == "HTS raw")].iloc[0]
    s1 = sub[(sub.department == "11") & (sub.source == "Synthetic")].iloc[0]
    df = pd.DataFrame([
        ["Median income", f"{h0.median_mpyg:.2f} M", f"{s0.median_mpyg:.2f} M", f"{h1.median_mpyg:.2f} M", f"{s1.median_mpyg:.2f} M"],
        ["Zero-income households", f"{h0.zero_pct:.1f}%", f"{s0.zero_pct:.1f}%", f"{h1.zero_pct:.1f}%", f"{s1.zero_pct:.1f}%"],
        ["Income ≥8 M", f"{h0.over_8m_pct:.1f}%", f"{s0.over_8m_pct:.1f}%", f"{h1.over_8m_pct:.1f}%", f"{s1.over_8m_pct:.1f}%"],
    ], columns=["Indicator", "Asunción HTS", "Asunción synthetic", "Central HTS", "Central synthetic"])
    df.to_csv(out / f"05_hts_vs_synthetic_table_{active}.csv", index=False)
    df.to_markdown(out / f"05_hts_vs_synthetic_table_{active}.md", index=False)
    _figure_table(df, f"Where the differences occur\nConfigured scenario: {active}", out / f"05_hts_vs_synthetic_table_{active}")


def _synthetic_department_table(stats, active, out):
    sub = stats[(stats.income_type == active) & (stats.department.isin(["00", "11"])) & (stats.source == "Synthetic")].set_index('department')
    if sub.empty:
        return
    a, c = sub.loc['00'], sub.loc['11']
    df = pd.DataFrame([
        ["Synthetic households", f"{int(a.households):,}", f"{int(c.households):,}"],
        ["Mean income", f"{a.mean_mpyg:.2f} M", f"{c.mean_mpyg:.2f} M"],
        ["Median income", f"{a.median_mpyg:.2f} M", f"{c.median_mpyg:.2f} M"],
        ["75th percentile", f"{a.p75_mpyg:.2f} M", f"{c.p75_mpyg:.2f} M"],
        ["90th percentile", f"{a.p90_mpyg:.2f} M", f"{c.p90_mpyg:.2f} M"],
        ["Households with income ≥8 M", f"{a.over_8m_pct:.1f}%", f"{c.over_8m_pct:.1f}%"],
        ["Households with zero income", f"{a.zero_pct:.1f}%", f"{c.zero_pct:.1f}%"],
    ], columns=["Indicator", "Asunción (00)", "Central (11)"])
    df.to_csv(out / f"06_synthetic_department_table_{active}.csv", index=False)
    df.to_markdown(out / f"06_synthetic_department_table_{active}.md", index=False)
    _figure_table(df, f"Synthetic population comparison by department\nConfigured scenario: {active}", out / f"06_synthetic_department_table_{active}")


def _missing_table_and_plot(diag, out):
    plt = _plotting()
    h = diag.copy()
    h['department'] = _dept(h.departement_id)
    rows = []
    plot_rows = []
    for dep in ['ALL', '00', '11']:
        d = h if dep == 'ALL' else h[h.department == dep]
        w = _number(d.household_weight)
        if w.sum() <= 0:
            continue
        before_job = 100 * w[d.job_unknown].sum() / w.sum()
        before_nonjob = 100 * w[d.nonjob_unknown].sum() / w.sum()
        before_any = 100 * w[d.any_unknown].sum() / w.sum()
        after_job_na = 100 * w[d.fixed_job_pyg.isna()].sum() / w.sum()
        after_total_na = 100 * w[d.fixed_total_pyg.isna()].sum() / w.sum()
        after_total_incomplete = 100 * w[~d.is_total_income_complete].sum() / w.sum()
        label = 'Asunción + Central' if dep == 'ALL' else DEPARTMENTS[dep]
        rows.append([label, f"{before_job:.1f}%", f"{before_nonjob:.1f}%", f"{before_any:.1f}%", f"{after_job_na:.1f}%", f"{after_total_na:.1f}%", f"{after_total_incomplete:.1f}%"])
        plot_rows.append((label, [before_job, before_nonjob, before_any, after_job_na, after_total_na, after_total_incomplete]))
    df = pd.DataFrame(rows, columns=["Department", "Before repair: missing job", "Before repair: missing non-job", "Before repair: any missing component", "After repair: NA job income", "After repair: NA total income", "After repair: estimated incomplete total"])
    df.to_csv(out / "07_missing_income_table.csv", index=False)
    df.to_markdown(out / "07_missing_income_table.md", index=False)
    table_view = df.rename(columns={
        "Department": "Department",
        "Before repair: missing job": "Before: job\nunknown",
        "Before repair: missing non-job": "Before: non-job\nunknown",
        "Before repair: any missing component": "Before: any\nunknown",
        "After repair: NA job income": "After: job\nNA",
        "After repair: NA total income": "After: total\nNA",
        "After repair: estimated incomplete total": "After: total\nincomplete*",
    })
    _figure_table(table_view, "Missing-income exposure before and after repair · weighted HTS household share\n*Incomplete total = best-guess estimate with unresolved income components treated as zero", out / "07_missing_income_table", font_size=9, col_width=1.0)

    fig, ax = plt.subplots(figsize=(13.6, 5.6), layout='constrained')
    kinds = ["Before: missing job", "Before: missing non-job", "Before: any missing", "After: NA job", "After: NA total", "After: incomplete total*"]
    xs = np.arange(len(kinds)); wd = .22
    palette = ['#4C78A8', '#F28E2B', '#9C755F']
    for i, (label, vals) in enumerate(plot_rows):
        ax.bar(xs + (i - (len(plot_rows)-1)/2) * wd, vals, width=wd * .9, label=label, color=palette[i])
    ax.set_xticks(xs, kinds, rotation=12, ha='right')
    ax.set_ylabel("Weighted share of HTS households (%)")
    ax.set_title("Missing-income exposure: remaining NA versus best-guess incomplete totals", weight='bold', loc='left')
    ax.text(0, -0.22, "*Incomplete total is numeric; remaining unknown components were treated as zero.", transform=ax.transAxes, fontsize=9, color="#536474")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis='y', alpha=.15)
    _save(fig, out / "07_missing_income_plot")


def _write_methods(folder, active, missing, ephc_path, synthetic_path, synth_scale):
    notes = f"""# Asunción income validation – methods and limitations

**Configured scenario:** `{active}`. Synthetic results are shown only for the active income scenario.

**Sources:** raw HTS, repaired cleaned HTS, urban EPHC 2022 `{ephc_path}`, and synthesis.output `{synthetic_path}`.
`HTS complete cases` removes entire households if any resident has a special-coded unknown income field.

**Units:** calculations use full PYG internally and display million PYG. Synthetic `income` is converted using
`asuncion.income_analysis_synthetic_scale_pyg = {synth_scale:g}`.

**Definitions:** job-only in raw HTS = sum of reported primary and secondary earnings, ignoring special-coded unknown amounts.
Raw total = original HTS household total. Repaired values come directly from `asuncion.data.hts.entd.cleaned`; this stage does not re-run imputation.
EPHC job-only = `e01aimde + e01bimde + e01cimde`. EPHC total = `ingrem`.

**Missingness:** sentinel codes {list(MISSING_CODES)} are treated as unknown. Fixed job-income missing households: {missing['fixed_job_missing_households']}.
Fixed total-income missing households: {missing['fixed_total_missing_households']}. Households with estimated but incomplete repaired total income: {missing['fixed_total_incomplete_households']} ({missing['weighted_fixed_total_incomplete_share_pct']:.2f}% weighted). Unresolved non-job components remain `NA` individually, but are treated as zero when computing the numeric best-guess total. `is_total_income_complete` is `False` for households with unresolved components. The flag measures component completeness, not the reliability of hot-deck-imputed job income.

**Important limitation:** this stage does **not** include quantile-mapped HTS income. It compares raw HTS, complete-case HTS,
repaired HTS, EPHC, and the synthetic population only.
"""
    (folder / "methods.md").write_text(notes, encoding='utf-8')


def _summary_report(out, stats, active):
    lines = [f"# Income validation summary — configured scenario: `{active}`", ""]
    for mode in INCOME_TYPES:
        lines.append(f"## {'Job-only' if mode == 'job_only' else 'Total'} income")
        lines.append("")
        lines.append("| Dataset | Mean | Median | Income ≥8M | Zero income |")
        lines.append("|---|---:|---:|---:|---:|")
        sub = stats[(stats.income_type == mode) & (stats.department == 'ALL') & (stats.source.isin(_display_sources(mode, active)))]
        sub = sub.set_index('source').reindex(_display_sources(mode, active)).reset_index()
        for _, r in sub.iterrows():
            lines.append(f"| {r['source']} | {r['mean_mpyg']:.2f} | {r['median_mpyg']:.2f} | {r['over_8m_pct']:.1f}% | {r['zero_pct']:.1f}% |")
        lines.append("")
    (out / "summary_report.md").write_text("\n".join(lines) + "\n", encoding='utf-8')


def create_income_report(raw, cleaned, synthetic_root, prefix, analysis_path,
                         ephc_path, income_type="job_only", synthetic_scale_pyg=1000.0):
    if income_type not in INCOME_TYPES:
        raise ValueError(f"Unsupported income_type={income_type!r}; expected {INCOME_TYPES}")
    out = Path(analysis_path) / "income_validation"
    out.mkdir(parents=True, exist_ok=True)
    # Scenario-dependent files from an earlier run must not be mistaken for
    # results of the current income_type (job_only vs total_income).
    stale_patterns = (
        "01b_shape_grid_ephc_vs_synthetic_*.png", "01b_shape_grid_ephc_vs_synthetic_*.pdf",
        "05_hts_vs_synthetic_table_*.png", "05_hts_vs_synthetic_table_*.pdf",
        "05_hts_vs_synthetic_table_*.csv", "05_hts_vs_synthetic_table_*.md",
        "06_synthetic_department_table_*.png", "06_synthetic_department_table_*.pdf",
        "06_synthetic_department_table_*.csv", "06_synthetic_department_table_*.md",
    )
    for pattern in stale_patterns:
        for stale in out.glob(pattern):
            stale.unlink()

    hts_frames, missing, diagnostic = _observed_hts(raw, cleaned)
    synthetic, synthetic_path = _synthetic(synthetic_root, prefix, income_type, synthetic_scale_pyg)
    data = pd.concat([_ephc(ephc_path), *hts_frames, synthetic], ignore_index=True)
    stats, classes = _stats(data)

    stats.to_csv(out / "weighted_statistics.csv", index=False, float_format="%.7g")
    classes.to_csv(out / "income_class_shares.csv", index=False, float_format="%.7g")
    pd.DataFrame([missing]).to_csv(out / "hts_income_quality.csv", index=False)
    diagnostic.to_csv(out / "hts_households_raw_vs_fixed.csv", index=False)

    for mode in INCOME_TYPES:
        _shape_grid(data, mode, income_type, out)
        _summary_table(stats, mode, income_type, out)
        _income_class_plot_with_table(classes, mode, income_type, out)
    _dept_means_table(stats, income_type, out)
    _shape_grid_synthetic_vs_ephc(data, income_type, out)
    _hts_vs_synthetic_table(stats, income_type, out)
    _synthetic_department_table(stats, income_type, out)
    _missing_table_and_plot(diagnostic, out)
    _write_methods(out, income_type, missing, ephc_path, synthetic_path, synthetic_scale_pyg)
    _summary_report(out, stats, income_type)

    quality = {
        "configured_income_type": income_type,
        "synthetic_scale_pyg": synthetic_scale_pyg,
        "hts_missing": missing,
        "ephc_source": str(ephc_path),
        "synthetic_file": str(synthetic_path),
        "warnings": [
            "HTS complete cases selects only households with no unknown income fields.",
            "Repaired HTS total is a numeric best guess; any unresolved components contribute zero. Use is_total_income_complete to identify incomplete estimates.",
            "EPHC total includes imputed rent and is not fully conceptually identical to HTS total.",
            "Synthetic results are shown only for the configured income_type.",
            "No quantile-mapped HTS income is included in this report."
        ]
    }
    (out / "quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False, default=str), encoding='utf8')
    print(f"[income_validation] Created report in {out}")
    return str(out)


def execute(context):
    context.stage("synthesis.output")
    return create_income_report(
        raw=context.stage("data.hts.entd.raw"),
        cleaned=context.stage("asuncion.data.hts.entd.cleaned"),
        synthetic_root=context.config("output_path"),
        prefix=context.config("output_prefix"),
        analysis_path=context.config("analysis_path"),
        ephc_path=Path(context.config("data_path")) / context.config("asuncion.income_survey"),
        income_type=context.config("income_type"),
        synthetic_scale_pyg=float(context.config("asuncion.income_analysis_synthetic_scale_pyg")),
    )


def validate(context):
    path = Path(context.config("data_path")) / context.config("asuncion.income_survey")
    if not path.is_file():
        raise FileNotFoundError(f"EPHC data missing: {path}")
    return [path.stat().st_size]
