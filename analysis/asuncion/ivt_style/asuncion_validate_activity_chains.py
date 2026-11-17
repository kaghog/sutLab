"""
Validate exported synthetic demand against reweighted HTS: consecutive-activity
straight-line trip distances and ordered daily activity chains among travelers.
Outputs PDF, PNG, CSV and a quality report in analysis_path/activity_chain_validation.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

PURPOSES = ('home', 'work', 'education', 'shop', 'leisure', 'other')
SHORT = dict(zip(PURPOSES, ('H', 'W', 'E', 'S', 'L', 'O')))
COLORS = {'Synthetic': '#1f77b4', 'HTS': '#ff7f0e'}
ALIASES = {'shopping': 'shop', 'school': 'education', 'study': 'education',
           'studies': 'education', 'recreation': 'leisure'}
MISSING = {'', 'none', 'nan', 'unknown', 'null', '<na>'}
DEFAULTS = {
    'activity_validation_top_n': 15,
    'activity_validation_max_km': 25.0,
    'activity_validation_bin_km': 0.5,
    'activity_validation_purpose_endpoint': 'following',
    'activity_validation_hts_trip_order': 'auto',
    'activity_validation_hts_distance_column': 'auto',
    'activity_validation_hts_distance_unit': 'm',
    # Optional explicit coordinate columns, ordered origin x/y, destination x/y.
    'activity_validation_hts_coordinate_columns': [],
    'activity_validation_hts_coordinate_crs': '',
    'activity_validation_metric_crs': 'EPSG:32721',
}


def configure(context):
    context.stage('synthesis.output')
    context.stage('asuncion.data.hts.entd.reweighted')
    for key in ('analysis_path', 'output_path', 'output_prefix'):
        context.config(key)
    for key, value in DEFAULTS.items():
        context.config(key, value)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    return plt


def _save(fig, path):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight', dpi=260)
    _plt().close(fig)


def _require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{label}: missing {sorted(missing)}; available: {list(frame.columns)}')


def _ids(frame, label):
    _require(frame, ['person_id'], label)
    frame = frame.copy()
    frame['person_id'] = (frame.person_id.astype('string').str.strip()
                         .str.replace(r'^(\d+)\.0$', r'\1', regex=True))
    if frame.person_id.isna().any() or frame.person_id.eq('').any():
        raise ValueError(f'{label}: missing person_id')
    return frame


def _purpose(series, label, quality):
    s = series.astype('string').str.strip().str.lower().replace(ALIASES)
    extra = s[~s.isin(set(PURPOSES) | MISSING)].dropna().value_counts()
    quality['other_purpose_mapping'][label] = {str(k): int(v) for k, v in extra.items()}
    # Nonstandard known purposes are grouped into other; unknowns remain missing.
    return s.where(s.isin(PURPOSES), 'other').mask(s.isna() | s.isin(MISSING))


def _index(series, label):
    v = pd.to_numeric(series, errors='coerce')
    if not np.isfinite(v.to_numpy(dtype=float, na_value=np.nan)).all() or (v % 1 != 0).any():
        raise ValueError(f'{label}: expected finite integer sequence indices')
    return v.astype('int64')


def _read_synthetic(root, prefix, metric_crs, quality):
    import geopandas as gpd
    from pyproj import CRS
    crs = CRS.from_user_input(metric_crs)
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info[:2]):
        raise ValueError('activity_validation_metric_crs must be a projected metre CRS')
    root = Path(root)
    trips = pd.read_csv(root / f'{prefix}trips.csv', sep=';', dtype={'person_id': 'string'}, low_memory=False)
    gpkg, parquet = (root / f'{prefix}activities.gpkg', root / f'{prefix}activities.geoparquet')
    if gpkg.is_file():
        activities = gpd.read_file(gpkg)
    elif parquet.is_file():
        activities = gpd.read_parquet(parquet)
    else:
        raise FileNotFoundError(f'Need {gpkg} or {parquet}')
    _require(activities, ['person_id', 'activity_index', 'purpose', 'geometry'], 'synthetic activities')
    activities = _ids(activities, 'synthetic activities')
    if activities.crs is None:
        raise ValueError('Synthetic activities have no CRS')
    activities['activity_index'] = _index(activities.activity_index, 'synthetic activity_index')
    if activities.duplicated(['person_id', 'activity_index']).any():
        raise ValueError('Duplicate synthetic activity keys')
    activities['purpose'] = _purpose(activities.purpose, 'synthetic activities', quality)
    activities = activities.to_crs(crs).sort_values(['person_id', 'activity_index'])
    valid = activities.geometry.notna() & ~activities.geometry.is_empty & activities.geom_type.eq('Point')
    xy = activities.geometry.where(valid)
    activities['x'], activities['y'] = xy.x, xy.y
    quality['coverage']['synthetic_activities_without_point'] = int((~valid).sum())
    trips = _ids(trips, 'synthetic trips')
    _require(trips, ['preceding_activity_index'], 'synthetic trips')
    trips['preceding_activity_index'] = _index(trips.preceding_activity_index, 'synthetic preceding_activity_index')
    if 'following_activity_index' in trips:
        trips['following_activity_index'] = _index(trips.following_activity_index, 'synthetic following_activity_index')
    else:
        trips['following_activity_index'] = trips.preceding_activity_index + 1
        quality['warnings'].append('Synthetic following_activity_index absent: using preceding_activity_index + 1.')
    if trips.duplicated(['person_id', 'preceding_activity_index']).any():
        raise ValueError('Duplicate synthetic trip origin activity keys')
    # Join by exact person/activity keys, never by row position or nearest point.
    for endpoint in ('preceding', 'following'):
        key = f'{endpoint}_activity_index'
        cols = activities[['person_id', 'activity_index', 'purpose', 'x', 'y']].rename(columns={
            'activity_index': key, 'purpose': f'{endpoint}_activity_purpose',
            'x': f'{endpoint}_x', 'y': f'{endpoint}_y'})
        trips = trips.merge(cols, on=['person_id', key], how='left', validate='many_to_one')
    trips['distance_km'] = np.hypot(trips.following_x - trips.preceding_x,
                                   trips.following_y - trips.preceding_y) / 1000.0
    trips['weight'] = 1.0
    # Strictly use the actual exported trip pairs to define which chains are complete.
    actual = trips[['person_id', 'preceding_activity_index', 'following_activity_index']].copy()
    next_index = activities.groupby('person_id', sort=False).activity_index.shift(-1)
    expected = activities.loc[next_index.notna(), ['person_id', 'activity_index']].rename(
        columns={'activity_index': 'preceding_activity_index'})
    expected['following_activity_index'] = next_index[next_index.notna()].astype('int64').to_numpy()
    keys = ['person_id', 'preceding_activity_index', 'following_activity_index']
    difference = expected.merge(actual, on=keys, how='outer', indicator=True)
    bad_ids = set(difference.loc[difference._merge.ne('both'), 'person_id'])
    mobile = set(trips.person_id)
    a = activities[activities.person_id.isin(mobile) & ~activities.person_id.isin(bad_ids)]
    missing_ids = set(a.loc[a.purpose.isna(), 'person_id'])
    a = a[~a.person_id.isin(missing_ids)]
    chains = a.groupby('person_id', sort=False).purpose.agg(tuple).rename('chain').reset_index()
    chains['weight'] = 1.0
    quality['coverage'].update(synthetic_trip_rows=len(trips), synthetic_travelers=len(mobile),
        synthetic_people_with_inconsistent_activity_trip_links=len(bad_ids & mobile),
        synthetic_travelers_with_missing_purpose=len(missing_ids),
        synthetic_people_with_activities_but_no_trips=int((~activities.person_id.isin(mobile)).groupby(activities.person_id).all().sum()),
        synthetic_valid_chains=len(chains))
    return trips, chains


def _hts_distances(trips, distance_column, unit, coordinate_columns, coordinate_crs, metric_crs, quality):
    if coordinate_columns:
        from pyproj import Transformer
        if len(coordinate_columns) != 4 or not coordinate_crs:
            raise ValueError('Provide four HTS coordinate columns and their CRS')
        _require(trips, coordinate_columns, 'HTS coordinates')
        transform = Transformer.from_crs(coordinate_crs, metric_crs, always_xy=True)
        ox, oy, dx, dy = [pd.to_numeric(trips[c], errors='coerce').to_numpy(float) for c in coordinate_columns]
        ox, oy = transform.transform(ox, oy)
        dx, dy = transform.transform(dx, dy)
        quality['hts_distance_source'] = {'coordinates': list(coordinate_columns), 'crs': coordinate_crs}
        return np.hypot(dx - ox, dy - oy) / 1000.0
    if distance_column == 'auto':
        candidates = [c for c in ('crowfly_distance', 'crow_fly_distance', 'crowfly_distance_m',
                                   'crow_fly_distance_m', 'crowfly_distance_km', 'crow_fly_distance_km',
                                   'euclidean_distance', 'straight_line_distance') if c in trips]
        if len(candidates) != 1:
            raise ValueError('Cannot uniquely identify HTS crow-fly distance. Set '
                             'activity_validation_hts_distance_column and unit, or supply coordinates. '
                             f'Candidates: {candidates}; columns: {list(trips.columns)}')
        distance_column = candidates[0]
        if distance_column.endswith('_km'):
            unit = 'km'
        elif distance_column.endswith('_m'):
            unit = 'm'
        else:
            quality['warnings'].append(f'HTS {distance_column} interpreted using configured unit {unit!r}; verify against source stage.')
    _require(trips, [distance_column], 'HTS distance')
    if unit not in ('m', 'km'):
        raise ValueError('HTS distance unit must be m or km')
    quality['hts_distance_source'] = {'column': distance_column, 'unit': unit}
    return pd.to_numeric(trips[distance_column], errors='coerce') / (1000. if unit == 'm' else 1.)


def _hts_order(trips, setting):
    if setting == 'auto':
        setting = next((c for c in ('trip_index', 'preceding_activity_index', 'trip_number', 'trip_order', 'trip_sequence', 'departure_time') if c in trips), None)
    if setting is None:
        raise ValueError(f'No HTS sequence column found; set activity_validation_hts_trip_order. Columns: {list(trips.columns)}')
    _require(trips, [setting], 'HTS trip order')
    # Accept numeric sequence/simulation seconds or hh:mm[:ss] departure times.
    raw = trips[setting]
    order = pd.to_numeric(raw, errors='coerce')
    if raw.notna().any() and order.notna().sum() < raw.notna().sum():
        clock = raw.astype('string').str.strip().str.extract(r'^(\d{1,3}):(\d{2})(?::(\d{2}(?:\.\d+)?))?$')
        hours, minutes, seconds = [pd.to_numeric(clock[i], errors='coerce') for i in range(3)]
        parsed = (hours * 3600 + minutes * 60 + seconds.fillna(0)).where((minutes < 60) & (seconds.fillna(0) < 60))
        order = order.fillna(parsed)
    order = order.where(np.isfinite(order.to_numpy(dtype=float, na_value=np.nan)))
    return setting, order


def _read_hts(reweighted, options, quality):
    _, persons, trips = reweighted
    persons, trips = _ids(persons, 'HTS persons'), _ids(trips, 'HTS trips')
    _require(persons, ['person_weight'], 'HTS persons')
    _require(trips, ['preceding_purpose', 'following_purpose', 'trip_weight'], 'HTS trips')
    if persons.person_id.duplicated().any():
        raise ValueError('HTS person_id must identify one survey person/day')
    persons['weight'] = pd.to_numeric(persons.person_weight, errors='coerce')
    trips['weight'] = pd.to_numeric(trips.trip_weight, errors='coerce')
    for end in ('preceding', 'following'):
        trips[f'{end}_activity_purpose'] = _purpose(trips[f'{end}_purpose'], f'HTS {end} purpose', quality)
    trips['distance_km'] = _hts_distances(trips, options['activity_validation_hts_distance_column'],
        options['activity_validation_hts_distance_unit'], options['activity_validation_hts_coordinate_columns'],
        options['activity_validation_hts_coordinate_crs'], options['activity_validation_metric_crs'], quality)
    setting, trips['_order'] = _hts_order(trips, options['activity_validation_hts_trip_order'])
    quality['hts_trip_order'] = setting
    if setting == 'departure_time':
        quality['warnings'].append('HTS ordered by departure_time; tied or missing times are excluded. Verify overnight ordering.')
    # Compare raw normalized purpose labels for continuity before grouping into other.
    for end in ('preceding', 'following'):
        trips[f'_{end}_raw'] = trips[f'{end}_purpose'].astype('string').str.strip().str.lower().replace(ALIASES)
    duplicate = trips.duplicated(['person_id', '_order'], keep=False)
    order_bad = set(trips.loc[trips._order.isna() | duplicate, 'person_id'])
    g = trips[~trips.person_id.isin(order_bad)].sort_values(['person_id', '_order']).copy()
    group = g.groupby('person_id', sort=False)
    previous = group['_following_raw'].shift()
    first = group.cumcount().eq(0)
    mismatch = ~first & ~g['_preceding_raw'].eq(previous).fillna(False)
    if setting in ('trip_index', 'preceding_activity_index', 'trip_number', 'trip_order', 'trip_sequence'):
        mismatch |= ~first & group['_order'].diff().ne(1)
    continuity_bad = set(g.loc[mismatch, 'person_id'])
    purpose_bad = set(g.loc[g.preceding_activity_purpose.isna() | g.following_activity_purpose.isna(), 'person_id'])
    g = g[~g.person_id.isin(continuity_bad | purpose_bad)]
    grouped = g.groupby('person_id', sort=False)
    starts = grouped.preceding_activity_purpose.first()
    ends = grouped.following_activity_purpose.agg(tuple)
    chains = pd.DataFrame({'chain': [(start,) + end for start, end in zip(starts, ends)]}, index=starts.index).reset_index()
    chains = chains.merge(persons[['person_id', 'weight']], on='person_id', how='left', validate='one_to_one')
    weights_valid = np.isfinite(chains.weight) & chains.weight.gt(0)
    mobile_ids = set(trips.person_id)
    quality['coverage'].update(hts_trip_rows=len(trips), hts_travelers=len(mobile_ids),
        hts_reweighted_persons=len(persons), hts_travelers_bad_order=len(order_bad),
        hts_travelers_inconsistent_purpose_or_index=len(continuity_bad),
        hts_travelers_missing_purpose=len(purpose_bad), hts_chains_missing_or_invalid_person_weight=int((~weights_valid).sum()),
        hts_valid_chains=int(weights_valid.sum()),
        hts_person_weight_all_travelers=float(persons.loc[persons.person_id.isin(mobile_ids) & np.isfinite(persons.weight) & persons.weight.gt(0), 'weight'].sum()),
        hts_person_weight_valid_chains=float(chains.loc[weights_valid, 'weight'].sum()))
    return trips, chains[weights_valid].copy()


def _distance_sample(trips, endpoint, name, quality):
    df = trips[['distance_km', 'weight', f'{endpoint}_activity_purpose']].rename(columns={f'{endpoint}_activity_purpose': 'purpose'})
    valid = np.isfinite(df.distance_km) & df.distance_km.ge(0) & np.isfinite(df.weight) & df.weight.gt(0) & df.purpose.isin(PURPOSES)
    quality['coverage'][name + '_valid_distance_trips'] = int(valid.sum())
    quality['coverage'][name + '_excluded_distance_trips'] = int((~valid).sum())
    quality['coverage'][name + '_distance_valid_weight'] = float(df.loc[valid, 'weight'].sum())
    return df[valid].copy()


def _distances(samples, out, max_km, bin_km, endpoint, quality):
    plt = _plt()
    edges = np.arange(0., max_km, bin_km)
    edges = np.append(edges, max_km)
    widths = np.diff(edges)
    fig1, axs1 = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True, layout='constrained')
    fig2, axs2 = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True, layout='constrained')
    bins, cdfs, stats = [], [], []
    for purpose, ax1, ax2 in zip(PURPOSES, axs1.flat, axs2.flat):
        for name, df in samples.items():
            sub = df[df.purpose.eq(purpose)]
            if sub.empty:
                quality['warnings'].append(f'{name}: no valid distance records for {purpose}.')
                stats.append(dict(dataset=name, purpose=purpose, n=0, weight=0.))
                continue
            d, w = sub.distance_km.to_numpy(float), sub.weight.to_numpy(float)
            ix = np.argsort(d); d, w = d[ix], w[ix]
            cw = np.cumsum(w) / w.sum()
            hist = np.histogram(d, bins=edges, weights=w)[0] / (w.sum() * widths)
            ax1.bar(edges[:-1], hist, width=widths, align='edge', alpha=.50, color=COLORS[name], label=name)
            # Exact weighted ECDF including a possible atom at zero.
            visible = d <= max_km
            last = cw[visible][-1] if visible.any() else 0.
            ax2.step(np.r_[0., d[visible], max_km], np.r_[0., cw[visible], last],
                     where='post', color=COLORS[name], linewidth=1.7, label=name)
            cdf_grid = np.r_[0., cw][np.searchsorted(d, edges, side='right')]
            bins.extend(dict(dataset=name, purpose=purpose, lower_km=l, upper_km=r, density=v, share=v*(r-l)) for l,r,v in zip(edges[:-1],edges[1:],hist))
            cdfs.extend(dict(dataset=name, purpose=purpose, distance_km=x, cumulative_share=y) for x,y in zip(edges,cdf_grid))
            q = lambda p: float(d[min(np.searchsorted(cw, p), len(d)-1)])
            stats.append(dict(dataset=name, purpose=purpose, n=len(d), weight=float(w.sum()),
                mean_km=float(np.average(d, weights=w)), median_km=q(.5), p90_km=q(.9),
                share_above_plot_limit=float(w[d > max_km].sum()/w.sum())))
        for ax in (ax1, ax2):
            ax.set_title(purpose); ax.set_xlim(0, max_km); ax.grid(alpha=.18); ax.set_axisbelow(True)
        ax2.set_ylim(0, 1.025)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axs1.flat[0].legend(handles=[Patch(facecolor=c, alpha=.5, label=n) for n,c in COLORS.items()])
    axs2.flat[0].legend(handles=[Line2D([0], [0], color=c, label=n) for n,c in COLORS.items()])
    for fig, ylabel in ((fig1,'Density (1/km)'), (fig2,'Cumulative share')):
        fig.supxlabel('Crow-fly distance [km]'); fig.supylabel(ylabel)
        fig.suptitle(f'Trip distance by {"destination" if endpoint == "following" else "origin"} activity purpose', fontsize=14)
    _save(fig1, out/'distance_distribution_by_purpose')
    _save(fig2, out/'distance_cdf_by_purpose')
    pd.DataFrame(bins).to_csv(out/'distance_distribution_by_purpose.csv', index=False)
    pd.DataFrame(cdfs).to_csv(out/'distance_cdf_by_purpose.csv', index=False)
    pd.DataFrame(stats).to_csv(out/'distance_summary_by_purpose.csv', index=False)


def _chain_plots(samples, out, top_n, quality):
    plt = _plt()
    series, counts = {}, {}
    for name, df in samples.items():
        if df.empty or df.weight.sum() <= 0:
            raise ValueError(f'{name}: no usable daily activity chains; check trip order and quality.')
        # Construct hashable/displayable keys without collapsing repeated activities.
        labels = df.chain.map(lambda seq: '–'.join(SHORT[p] for p in seq))
        series[name] = df.groupby(labels).weight.sum() / df.weight.sum()
        counts[name] = labels.value_counts()
    shares = pd.DataFrame(series).fillna(0.)
    shares = shares.assign(rank=shares.mean(axis=1)).sort_values('rank', ascending=False, kind='stable').drop(columns='rank')
    table = shares.rename(columns={'Synthetic':'synthetic_share', 'HTS':'hts_share'}).copy()
    table['synthetic_n'] = counts['Synthetic'].reindex(table.index, fill_value=0)
    table['hts_n'] = counts['HTS'].reindex(table.index, fill_value=0)
    table['difference_pp'] = 100*(table.synthetic_share-table.hts_share)
    table.to_csv(out/'activity_chains.csv', index_label='chain')
    quality['activity_chain_total_variation'] = float(.5 * (shares.Synthetic - shares.HTS).abs().sum())
    shown = shares.head(top_n).copy()
    if len(shares) > top_n:
        shown.loc['All other chains'] = shares.iloc[top_n:].sum()
    shown.to_csv(out/'activity_chains_top.csv', index_label='chain')
    fig, ax = plt.subplots(figsize=(11, max(5.2, .42*len(shown)+1.7)), layout='constrained')
    y = np.arange(len(shown)); h = .36
    for i, name in enumerate(('Synthetic','HTS')):
        ax.barh(y + (i-.5)*h, shown[name]*100, height=h, color=COLORS[name], label=name)
    ax.set_yticks(y, shown.index); ax.invert_yaxis(); ax.set_xlim(left=0)
    ax.set_xlabel('Share of travelers with valid daily chains (%)')
    ax.set_title('Most frequent daily activity chains'); ax.legend(); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    fig.supxlabel('H = Home · W = Work · E = Education · S = Shop · L = Leisure · O = Other', fontsize=9)
    _save(fig, out/'activity_chains_top')
    # A chain-length comparison distinguishes simple and complex daily patterns.
    lengths = {}
    for name, df in samples.items():
        n = df.chain.map(len)-1
        lengths[name] = df.groupby(n).weight.sum()/df.weight.sum()
    full = pd.DataFrame(lengths).fillna(0.).sort_index()
    full.to_csv(out/'trips_per_person.csv', index_label='number_of_trips')
    plot = full.reindex(range(1,10), fill_value=0.)
    plot.loc[10] = full.loc[full.index >=10].sum()
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained'); x=np.arange(len(plot)); w=.36
    for i, name in enumerate(('Synthetic','HTS')):
        ax.bar(x+(i-.5)*w, plot[name]*100, width=w, color=COLORS[name], label=name)
    ax.set_xticks(x, [str(i) for i in range(1,10)]+['10+']); ax.set_ylim(bottom=0)
    ax.set(xlabel='Trips per traveler', ylabel='Share of travelers (%)', title='Daily activity-chain length')
    ax.legend(); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    _save(fig, out/'trips_per_person')


def create_validation(reweighted, synthetic_root, prefix, analysis_path, **overrides):
    unknown = set(overrides)-set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown validation options: {sorted(unknown)}')
    options = {**DEFAULTS, **overrides}
    max_km = float(options['activity_validation_max_km']); bin_km = float(options['activity_validation_bin_km'])
    top_n = int(options['activity_validation_top_n']); endpoint = options['activity_validation_purpose_endpoint']
    if not (np.isfinite(max_km) and np.isfinite(bin_km) and 0 < bin_km <= max_km and top_n > 0 and endpoint in ('preceding','following')):
        raise ValueError('Invalid distance range, bin size, top_n, or purpose endpoint')
    out = Path(analysis_path)/'activity_chain_validation'; out.mkdir(parents=True, exist_ok=True)
    quality = {'warnings': [], 'coverage': {}, 'other_purpose_mapping': {}, 'options': options,
        'definitions': {'distance':'One trip between consecutive activity locations; HTS trip_weight, synthetic unit weights; full distribution denominator even above plot limit.',
                        'chains':'One ordered activity sequence per valid traveler; HTS person_weight, synthetic unit weights; no forced home anchors or collapsed repetitions. No-trip persons excluded on both sides.',
                        'chain_ranking':'Mean of synthetic and HTS shares; all unshown chains pooled.'}}
    try:
        syn_trips, syn_chains = _read_synthetic(synthetic_root, prefix, options['activity_validation_metric_crs'], quality)
        hts_trips, hts_chains = _read_hts(reweighted, options, quality)
        distances = {name: _distance_sample(df, endpoint, name.lower(), quality) for name,df in (('Synthetic',syn_trips),('HTS',hts_trips))}
        for name, df in distances.items():
            if df.empty:
                raise ValueError(f'{name}: no valid trip distances')
        _distances(distances, out, max_km, bin_km, endpoint, quality)
        _chain_plots({'Synthetic':syn_chains,'HTS':hts_chains},out,top_n,quality)
        for name in ('synthetic', 'hts'):
            c = quality['coverage']
            excluded = c[name + '_travelers'] - c[name + '_valid_chains']
            if excluded:
                quality['warnings'].append(f'{name}: {excluded} travelers excluded from chain comparison; see coverage.')
            if c[name + '_excluded_distance_trips']:
                quality['warnings'].append(f'{name}: {c[name + "_excluded_distance_trips"]} trips excluded from distance comparison; see coverage.')
        quality['status'] = 'complete'
    except Exception as exc:
        quality['status'] = 'failed'; quality['error'] = str(exc)
        raise
    finally:
        (out/'quality_report.json').write_text(json.dumps(quality,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    for message in quality['warnings']:
        warnings.warn('[activity chain validation] '+message, stacklevel=2)
    print(f'Activity-chain validation: {out}')
    return str(out)


def execute(context):
    context.stage('synthesis.output')
    return create_validation(reweighted=context.stage('asuncion.data.hts.entd.reweighted'),
        synthetic_root=context.config('output_path'), prefix=context.config('output_prefix'),
        analysis_path=context.config('analysis_path'),
        **{key: context.config(key) for key in DEFAULTS})
