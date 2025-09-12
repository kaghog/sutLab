#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

out = Path('output')
# Detect prefix from persons file name
persons_files = sorted(out.glob('*_persons.csv'))
if not persons_files:
    raise SystemExit('No persons CSV found in output/')
persons_path = persons_files[0]
prefix = persons_path.name.replace('persons.csv', '')
trips_path = out / f"{prefix}trips.csv"

# Hannover CSVs are semicolon-delimited; ensure booleans are parsed robustly
bool_conv = lambda v: str(v).strip().lower() in ("true", "1", "yes")
persons = pd.read_csv(persons_path, sep=';', converters={
    'has_pt_subscription': bool_conv,
    'has_driving_license': bool_conv,
})
trips = pd.read_csv(trips_path, sep=';') if trips_path.exists() else pd.DataFrame()

res = {}
res['files'] = {
    'persons': str(persons_path),
    'trips': str(trips_path) if trips_path.exists() else None,
}
res['n_persons'] = int(len(persons))
res['n_trips'] = int(len(trips)) if len(trips) else 0

# PT subscription metrics
persons['has_pt_subscription'] = persons['has_pt_subscription'].astype(bool)
res['pt_sub_overall_pct_age6plus'] = float(
    100.0 * (persons.loc[persons['age'] >= 6, 'has_pt_subscription'] == True).mean()
    if (persons['age'] >= 6).any() else 0.0
)
# By sex (native coding, typically 1/2)
res['pt_sub_by_sex_pct'] = {
    str(k): float(100.0 * (g['has_pt_subscription'] == True).mean())
    for k, g in persons.loc[persons['age'] >= 6].groupby('sex')
}
# By age bins
bins = [0,6,12,18,26,40,60,80,200]
labels = ['<6','6-11','12-17','18-25','26-39','40-59','60-79','80+']
persons['age_bin'] = pd.cut(persons['age'], bins=bins, labels=labels, right=False, include_lowest=True)
res['pt_sub_by_age_pct'] = {
    (str(k) if pd.notna(k) else 'NaN'): float(100.0 * (g['has_pt_subscription'] == True).mean())
    for k, g in persons.groupby('age_bin')
}
res['pt_sub_under6_count'] = int((persons.loc[persons['age'] < 6, 'has_pt_subscription'] == True).sum())

# Driving license metrics (column is has_driving_license in outputs)
persons['has_driving_license'] = persons['has_driving_license'].astype(bool)
res['license_overall_pct'] = float(100.0 * (persons['has_driving_license'] == True).mean())
res['license_by_sex_pct'] = {
    str(k): float(100.0 * (g['has_driving_license'] == True).mean())
    for k, g in persons.groupby('sex')
}
teen = persons[(persons['age'] >= 12) & (persons['age'] < 18)]
res['license_teen_by_sex_pct'] = {
    str(k): float(100.0 * (g['has_driving_license'] == True).mean())
    for k, g in teen.groupby('sex')
}

print(json.dumps(res, indent=2))
