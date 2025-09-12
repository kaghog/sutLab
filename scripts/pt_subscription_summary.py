#!/usr/bin/env python3
import glob
import pickle
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    # Load HTS reweighted cache
    pkl = sorted(glob.glob('cache/data.hts.entd.reweighted__*.p'))
    if not pkl:
        raise SystemExit('HTS cache not found under cache/data.hts.entd.reweighted__*.p')
    hh, persons, trips = pickle.load(open(pkl[-1], 'rb'))

    # Prepare persons with car availability and weights
    hh = hh.copy(); persons = persons.copy()
    if 'number_of_vehicles' in hh.columns:
        hh['number_of_vehicles'] = pd.to_numeric(hh['number_of_vehicles'], errors='coerce').fillna(0)
    else:
        hh['number_of_vehicles'] = 0
    hh['car_availability'] = hh['number_of_vehicles'] > 0
    persons = persons.merge(hh[['household_id','car_availability']], on='household_id', how='left')
    persons['car_availability'] = persons['car_availability'].fillna(False)
    persons = persons.rename(columns={'person_weight':'weight_person'})

    # Restrict to age >= 6
    persons6 = persons[persons['age'] >= 6].copy()

    # PT subscription share (weighted, all persons >=6)
    persons6['pt_sub'] = persons6['has_pt_subscription']
    hts_weighted_yes = persons6.loc[persons6['pt_sub'] == True, 'weight_person'].sum()
    hts_weighted_total = persons6['weight_person'].sum()
    hts_share_all = 100.0 * (hts_weighted_yes / hts_weighted_total) if hts_weighted_total>0 else 0.0

    # Trips-joined view (persons in trips, to mimic analysis join)
    trips_join = trips.merge(persons[['person_id','weight_person','age','has_pt_subscription']], on='person_id', how='left')
    trips_join = trips_join[trips_join['age'] >= 6]
    hts_weighted_yes_trips = trips_join.loc[trips_join['has_pt_subscription'] == True, 'weight_person'].sum()
    hts_weighted_total_trips = trips_join['weight_person'].sum()
    hts_share_trips = 100.0 * (hts_weighted_yes_trips / hts_weighted_total_trips) if hts_weighted_total_trips>0 else 0.0

    # Synthetic data
    syn_persons_csv = sorted(glob.glob('output/*persons.csv'))
    syn_trips_csv = sorted(glob.glob('output/*trips.csv'))
    if not syn_persons_csv or not syn_trips_csv:
        raise SystemExit('Synthetic outputs not found under output/*{persons,trips}.csv')
    sp = pd.read_csv(syn_persons_csv[-1], sep=';')
    st = pd.read_csv(syn_trips_csv[-1], sep=';')

    # All persons >=6 in synthetic
    sp6 = sp[sp['age'] >= 6].copy()
    syn_share_all = 100.0 * (sp6['has_pt_subscription'] == True).mean() if len(sp6)>0 else 0.0

    # Trips-joined synthetic (mimic analysis join)
    st_join = st.merge(sp[['person_id','age','has_pt_subscription']], on='person_id', how='inner')
    st_join = st_join[st_join['age'] >= 6]
    syn_share_trips = 100.0 * (st_join['has_pt_subscription'] == True).mean() if len(st_join)>0 else 0.0

    print('PT subscription shares (percent):')
    print(f' HTS persons>=6 (weighted): {hts_share_all:.2f}')
    print(f' HTS trips-joined (weighted): {hts_share_trips:.2f}')
    print(f' SYN persons>=6 (unweighted): {syn_share_all:.2f}')
    print(f' SYN trips-joined (unweighted): {syn_share_trips:.2f}')

    # Also compute by sex and age groups for extra insight
    bins = [0,6,15,18,24,30,45,65,80,150]
    labels = ["0-5","6-14","15-17","18-23","24-29","30-44","45-64","65-79","80+"]

    persons6['age_bin'] = pd.cut(persons6['age'], bins=bins, labels=labels)
    sp6['age_bin'] = pd.cut(sp6['age'], bins=bins, labels=labels)

    def pct_true_weighted(g, col_bool, wcol):
        num = g.loc[g[col_bool]==True, wcol].sum()
        den = g[wcol].sum()
        return 100.0 * num/den if den>0 else np.nan

    hts_by_sex = persons6.groupby('sex').apply(lambda g: pct_true_weighted(g, 'pt_sub', 'weight_person')).rename('HTS%')
    syn_by_sex = sp6.groupby('sex').apply(lambda g: 100.0 * (g['has_pt_subscription']==True).mean()).rename('SYN%')

    hts_by_age = persons6.groupby('age_bin').apply(lambda g: pct_true_weighted(g, 'pt_sub', 'weight_person')).rename('HTS%')
    syn_by_age = sp6.groupby('age_bin').apply(lambda g: 100.0 * (g['has_pt_subscription']==True).mean()).rename('SYN%')

    print('\nBy sex (0=male,1=female):')
    print(pd.concat([hts_by_sex, syn_by_sex], axis=1))

    print('\nBy age bin:')
    print(pd.concat([hts_by_age, syn_by_age], axis=1))


if __name__ == '__main__':
    main()
