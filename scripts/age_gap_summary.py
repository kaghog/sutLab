#!/usr/bin/env python3
"""
Age distribution summary (print-only) using existing generated data.

It loads the latest Census (weighted), HTS (weighted), and Synthetic persons (unweighted)
and prints:
    - Age-class percentages for Census, HTS, and Synthetic
    - Percentage-point gaps vs Census (SYN - CEN, HTS - CEN)
    - Overall discrepancy metrics vs Census (sum abs pp, chi2, KL) for SYN and HTS

Mimics file discovery style of scripts/pt_subscription_summary.py. No files are written.
"""

import glob
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _age_bins():
    """Hannover-specific age bins with merged 0–14 group.

    Returns
    - bounds: upper edges for np.digitize(..., right=True)
    - labels: matching labels
    """
    bounds = [6, 14, 17, 23, 29, 44, 64, 79, math.inf]
    labels = ["0-5", "6-14", "15-17", "18-23", "24-29", "30-44", "45-64", "65-79", "80+"]
    return bounds, labels


def add_age_class(df: pd.DataFrame, bounds):
    df = df.copy()
    df["age_class"] = np.digitize(df["age"].astype(int), bounds, right=True)
    return df


def pct_by_age(df: pd.DataFrame, bounds, weight_col: str | None):
    df = add_age_class(df, bounds)
    if weight_col is None or weight_col not in df.columns:
        df = df.copy(); df["__w__"] = 1.0; weight_col = "__w__"
    g = df.groupby("age_class")[weight_col].sum().reset_index(name="weight")
    tot = g["weight"].sum()
    g["percent"] = (g["weight"] / tot * 100.0) if tot > 0 else 0.0
    return g[["age_class", "percent"]]


def chi_square(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    return float(np.sum(((p - q) ** 2) / (q + eps)))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def main():
    # Census (weighted) from cache
    cen_p = sorted(glob.glob('cache/hannover.ipf.attributed__*.p') + glob.glob('cache/data.census.filtered__*.p'))
    census = None
    if cen_p:
        obj = pd.read_pickle(cen_p[-1])
        if isinstance(obj, pd.DataFrame):
            census = obj.copy()
        elif isinstance(obj, tuple):
            # try to find a DataFrame with 'age' and 'weight'
            for item in obj:
                if isinstance(item, pd.DataFrame) and {'age','weight'}.issubset(item.columns):
                    census = item.copy(); break
    if census is None:
        print('Warning: Census cache not found (looked for hannover.ipf.attributed__*.p or data.census.filtered__*.p)')

    # HTS: load latest reweighted cache
    pkl = sorted(glob.glob('cache/data.hts.entd.reweighted__*.p'))
    if not pkl:
        raise SystemExit('HTS cache not found under cache/data.hts.entd.reweighted__*.p')
    hh, persons, trips = pickle.load(open(pkl[-1], 'rb'))

    # Synthetic persons
    syn_persons_csv = sorted(glob.glob('output/*persons.csv'))
    if syn_persons_csv:
        syn = pd.read_csv(syn_persons_csv[-1], sep=';')
    else:
        # Fallback to pre-output sampled cache
        sp = sorted(glob.glob('cache/synthesis.population.sampled__*.p'))
        if not sp:
            raise SystemExit('Synthetic persons not found in output/*persons.csv or cache/synthesis.population.sampled__*.p')
        syn = pd.read_pickle(sp[-1])

    # Normalize HTS columns
    persons = persons.copy()
    if 'person_weight' not in persons and 'weight' in persons:
        persons = persons.rename(columns={'weight':'person_weight'})

    bounds, labels = _age_bins()

    # Distributions
    dist_cen = pct_by_age(census, bounds, 'weight') if census is not None else None
    dist_hts = pct_by_age(persons, bounds, 'person_weight')
    dist_syn = pct_by_age(syn, bounds, None)

    # Union of bins
    parts = [x for x in [dist_cen, dist_hts, dist_syn] if x is not None]
    keys = pd.concat([d[['age_class']] for d in parts]).drop_duplicates().sort_values('age_class')
    df = keys.copy()
    if dist_cen is not None:
        df = df.merge(dist_cen.rename(columns={'percent':'pct_census'}), on='age_class', how='left')
    df = df.merge(dist_hts.rename(columns={'percent':'pct_hts'}), on='age_class', how='left')
    df = df.merge(dist_syn.rename(columns={'percent':'pct_synth'}), on='age_class', how='left')
    df = df.fillna(0.0)

    # Labels and gaps
    df['label'] = df['age_class'].apply(lambda i: labels[int(i)] if int(i) < len(labels) else f"{int(i)}+")
    if 'pct_census' in df.columns:
        df['gap_syn_vs_cen_pp'] = df['pct_synth'] - df['pct_census']
        df['gap_hts_vs_cen_pp'] = df['pct_hts'] - df['pct_census']
    else:
        df['gap_syn_vs_cen_pp'] = np.nan
        df['gap_hts_vs_cen_pp'] = np.nan

    # Metrics
    # Metrics vs Census (if available)
    metrics = {}
    p_s = (df['pct_synth'].to_numpy(dtype=float) / 100.0)
    p_h = (df['pct_hts'].to_numpy(dtype=float) / 100.0)
    if 'pct_census' in df.columns:
        p_c = (df['pct_census'].to_numpy(dtype=float) / 100.0)
        metrics['synth_total_abs_pp_vs_census'] = float(np.sum(np.abs(df['gap_syn_vs_cen_pp'].to_numpy(dtype=float))))
        metrics['synth_chi2_vs_census'] = chi_square(p_s, p_c)
        metrics['synth_kl_vs_census'] = kl_div(p_s, p_c)
        metrics['hts_total_abs_pp_vs_census'] = float(np.sum(np.abs(df['gap_hts_vs_cen_pp'].to_numpy(dtype=float))))
        metrics['hts_chi2_vs_census'] = chi_square(p_h, p_c)
        metrics['hts_kl_vs_census'] = kl_div(p_h, p_c)
    else:
        metrics['note'] = 'Census not found; showing HTS vs SYN only.'

    # Pretty print
    print('Age distribution comparison (percentages)')
    if 'pct_census' in df.columns:
        header = '  {:>8}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}'.format('Class', 'Census %', 'HTS %', 'SYN %', 'Gap SYN-CEN', 'Gap HTS-CEN')
    else:
        header = '  {:>8}  {:>10}  {:>10}'.format('Class', 'HTS %', 'SYN %')
    print(header)
    for _, r in df.iterrows():
        if 'pct_census' in df.columns:
            print('  {:>8}  {:10.2f}  {:10.2f}  {:10.2f}  {:12.2f}  {:12.2f}'.format(
                r['label'], r['pct_census'], r['pct_hts'], r['pct_synth'], r['gap_syn_vs_cen_pp'], r['gap_hts_vs_cen_pp']))
        else:
            print('  {:>8}  {:10.2f}  {:10.2f}'.format(r['label'], r['pct_hts'], r['pct_synth']))

    print('\nOverall discrepancy metrics vs Census:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
