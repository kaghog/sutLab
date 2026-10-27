"""Comparison of archived Asuncion gravity runs.

Reads archives from <analysis_path>/gravity_runs/<run_name>/, takes ONE current
asuncion.gravity.observed HTS OD as reference, saves plots to
<analysis_path>/gravity_runs/_comparison/. No synthesis rerun.
"""
from pathlib import Path
import hashlib
import json
import re
import textwrap
import warnings

import numpy as np
import pandas as pd


def configure(context):
    context.config('analysis_path')
    context.stage('asuncion.gravity.observed')


def _ids(df, columns):
    df = df.copy()
    for c in columns:
        if c in df:
            df[c] = (df[c].astype('string').str.strip()
                     .str.replace(r'^(\d+)\.0$', r'\1', regex=True))
    return df


def _od(df, field='weight'):
    required = {'origin_id', 'destination_id', field}
    if not required.issubset(df.columns):
        raise ValueError(f'OD missing {required - set(df.columns)}')
    a = _ids(df[list(required)], ['origin_id', 'destination_id']).dropna()
    a[field] = pd.to_numeric(a[field], errors='raise')
    if not np.isfinite(a[field].to_numpy(float)).all() or (a[field] < 0).any():
        raise ValueError(f'Nonfinite/negative {field}')
    return a.groupby(['origin_id', 'destination_id'], as_index=False)[field].sum()


def _canonical_hts_hash(hts):
    a = hts.sort_values(['origin_id', 'destination_id'])
    lines = [f'{r.origin_id}\t{r.destination_id}\t{r.weight:.17g}'
             for r in a.itertuples(index=False)]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def _read_csv(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return pd.read_csv(path, dtype='string')


def _numeric(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='raise')
    return df


def _read_run(folder):
    analysis = folder / 'analysis'
    if not (analysis / 'model_probabilities.csv').is_file():
        return None
    model = _od(_numeric(_read_csv(analysis / 'model_probabilities.csv'), ['weight']))
    row_sum = model.groupby('origin_id').weight.sum()
    if not np.allclose(row_sum, 1, atol=1e-5):
        raise ValueError(f'{folder.name}: modeled probabilities do not sum to one per origin')
    with (analysis / 'calibration_parameters.json').open(encoding='utf-8') as f:
        calibration = json.load(f)
    config = {}
    if (folder / 'run_config.json').exists():
        config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
    realized = _numeric(_read_csv(analysis / 'final_home_work_od.csv'), ['persons_sample'])
    realized = _od(realized, 'persons_sample')
    jobs = None
    for p in [folder / 'analysis/job_attraction_by_zone.csv',
              folder / 'analysis/jobs_2011_by_zone.csv']:
        if p.is_file():
            jobs = _numeric(_read_csv(p), ['employees'])
            jobs = _ids(jobs, ['macrozone_id'])
            jobs = jobs.groupby('macrozone_id', as_index=False).employees.sum()
            break
    reference = None
    p = analysis / 'reference_fingerprints.json'
    if p.is_file():
        reference = json.loads(p.read_text(encoding='utf-8'))
    return dict(name=folder.name, folder=folder, model=model, realized=realized,
                jobs=jobs, calibration=calibration, config=config, reference=reference)


def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else np.nan


def _reference_data(hts, runs):
    """Common intersection of zones/origins; no unfair model-specific OD support."""
    common_origin = set(hts.origin_id)
    common_dest = set(runs[0]['model'].destination_id)
    for r in runs:
        common_origin &= set(r['model'].origin_id)
        common_dest &= set(r['model'].destination_id)
    # Note: if a model is missing a destination, that destination is excluded
    # for ALL models, as is its HTS weight. Each model is then renormalized
    # over common destinations; capture exclusions in the diagnostics.
    origins = sorted(common_origin)
    destinations = sorted(common_dest)
    if not origins or not destinations:
        raise ValueError('No common OD support among models and HTS')
    # only origins with observed work in common destination support
    obs = hts[hts.origin_id.isin(origins) & hts.destination_id.isin(destinations)]
    origins = sorted(obs.loc[obs.weight > 0, 'origin_id'].unique())
    if not origins:
        raise ValueError('No covered HTS origin workers')
    return origins, destinations


def _make_matrices(hts, run, origins, destinations):
    oi = pd.Index(origins); dj = pd.Index(destinations)
    h = (hts.pivot(index='origin_id', columns='destination_id', values='weight')
         .reindex(index=oi, columns=dj).fillna(0).to_numpy(float))
    m = (run['model'].pivot(index='origin_id', columns='destination_id', values='weight')
         .reindex(index=oi, columns=dj).fillna(0).to_numpy(float))
    row = h.sum(axis=1)
    if (row <= 0).any():
        raise ValueError('Zero HTS row in comparable origin set')
    mr = m.sum(axis=1)
    if (mr <= 0).any():
        raise ValueError('Zero model probability row after common support restriction')
    m = m / mr[:, None]
    h_prob = h / row[:, None]
    # Main prediction uses the SAME HTS home distribution for every model.
    model_flow = m * row[:, None]
    obs_shares = h.sum(axis=0) / h.sum()
    model_shares = model_flow.sum(axis=0) / model_flow.sum()
    final = run['realized']
    # For final versus HTS destination shares, use ALL archived geocoded people,
    # not just HTS-coverage origins (the sample composition can differ).
    f = final.groupby('destination_id').persons_sample.sum().reindex(dj, fill_value=0).to_numpy(float)
    fshare = f / f.sum() if f.sum() else np.full(len(dj), np.nan)
    same = np.equal.outer(np.asarray(origins), np.asarray(destinations))
    nll = -np.sum(h * np.log(np.clip(m, 1e-300, None))) / h.sum()
    tv = np.sum(np.abs(h_prob-m) * row[:, None]) / (2*h.sum())
    pearson = _pearson(h_prob.ravel(), m.ravel())
    metrics = dict(run=run['name'], correlation=pearson, rmse=float(np.sqrt(np.mean((m-h_prob)**2))),
                   mae=float(np.mean(abs(m-h_prob))), nll=float(nll),
                   origin_weighted_tv=float(tv),
                   interzonal_correlation=_pearson(h_prob[~same], m[~same]),
                   hts_intrazonal_share=float(h[same].sum()/h.sum()),
                   predicted_intrazonal_share=float(model_flow[same].sum()/model_flow.sum()),
                   destination_share_tv=float(np.abs(model_shares-obs_shares).sum()/2),
                   synthetic_destination_share_tv=float(np.abs(fshare-obs_shares).sum()/2),
                   hts_covered_weight=float(h.sum()),
                   hts_coverage_fraction=float(h.sum()/hts.weight.sum()),
                   final_geocoded_work_people=int(final.persons_sample.sum()))
    return dict(hts=h_prob, model=m, delta=h_prob-m, hts_flow=h,
                model_flow=model_flow, hts_dest=obs_shares,
                model_dest=model_shares, final_dest=fshare, metrics=metrics)


def _get_jobs(run, destinations):
    if run['jobs'] is None:
        return None
    j = run['jobs'].groupby('macrozone_id').employees.sum().reindex(destinations, fill_value=0).to_numpy(float)
    return j / j.sum() if j.sum() else None


def _figure_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.titlesize': 12, 'axes.labelsize': 10,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return plt


def _save(fig, path, plt):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight', dpi=260)
    plt.close(fig)


def _grid(n, width=5.0, height=4.4):
    import math
    cols = min(3, n)
    rows = math.ceil(n / cols)
    plt = _figure_setup()
    fig, axes = plt.subplots(rows, cols, figsize=(cols*width, rows*height), squeeze=False,
                             constrained_layout=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return plt, fig, list(axes.flat[:n])


def _plot_scatter(runs, results, folder):
    plt, fig, axes = _grid(len(runs), 5.1, 4.7)
    limit = max(float(np.nanmax(x[k])) for x in results for k in ('hts', 'model')) * 1.04
    limit = max(limit, 0.01)
    for ax, run, data in zip(axes, runs, results):
        x, y = data['hts'].ravel(), data['model'].ravel()
        ax.scatter(x, y, s=16, alpha=0.55, color='#24769a', linewidth=0, rasterized=True)
        ax.plot([0, limit], [0, limit], lw=1.4, color='#ad4d48', ls='--')
        ax.set(xlim=(0, limit), ylim=(0, limit), aspect='equal',
               xlabel='HTS conditional OD probability', ylabel='Gravity probability')
        ax.set_title(run['name'].replace('_', ' '), loc='left', pad=13)
        ax.text(.04,.96, f"Pearson r = {data['metrics']['correlation']:.3f}\n"
                           f"RMSE = {data['metrics']['rmse']:.3f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=10,
                bbox=dict(fc='white', ec='#dddddd', alpha=.95, boxstyle='round,pad=.45'))
        ax.grid(alpha=.13)
    fig.suptitle('Gravity destination choice versus household travel survey', fontsize=15)
    _save(fig, folder/'01_od_correlation', plt)


def _heat(ax, values, title, vmin, vmax, cmap, origins, destinations, center=None):
    import matplotlib.colors as colors
    kw = {'cmap': cmap, 'interpolation': 'nearest', 'aspect': 'auto', 'origin': 'upper'}
    kw['norm'] = colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(values, **kw)
    ax.set_title(title, loc='left', pad=9)
    ax.set_xticks(range(len(destinations)), destinations, rotation=65, ha='right', fontsize=8)
    ax.set_yticks(range(len(origins)), origins, fontsize=8)
    ax.set(xlabel='Work district', ylabel='Home district')
    return im


def _plot_od(runs, results, origins, dest, folder):
    plt = _figure_setup()
    n = len(runs)
    # One common absolute scale for HTS and all models.
    vmax = max(float(a.max()) for d in results for a in (d['hts'], d['model']))
    dv = max(float(np.max(np.abs(d['delta']))) for d in results)
    dv = max(dv, 1e-9)
    fig, axes = plt.subplots(n, 3, figsize=(16, max(4.3*n, 5.3)), squeeze=False, constrained_layout=True)
    for idx,(run,data) in enumerate(zip(runs,results)):
        ims=[]
        ims.append(_heat(axes[idx,0],data['hts'],f'{run["name"]} | HTS',0,vmax,'viridis',origins,dest))
        ims.append(_heat(axes[idx,1],data['model'],'Gravity model',0,vmax,'viridis',origins,dest))
        ims.append(_heat(axes[idx,2],data['delta'],'HTS − gravity',-dv,dv,'RdBu_r',origins,dest,center=0))
    fig.colorbar(ims[0],ax=axes[:,:2].ravel().tolist(),shrink=.66,label='Conditional destination probability')
    fig.colorbar(ims[-1],ax=axes[:,2].ravel().tolist(),shrink=.66,label='Probability difference (percentage as fraction)')
    fig.suptitle('Origin-conditional OD matrices: same zones, order and colour scales',fontsize=15)
    _save(fig,folder/'02_od_matrices_and_differences',plt)



def _plot_realized_od(runs, results, origins, dest, folder):
    """HTS versus realized final population's conditional commute OD.

    Rows with no sampled worker in any run are omitted from ALL panels; we do
    not render missing origins as real zero-probability distributions.
    """
    plt = _figure_setup()
    grid=[]
    for run in runs:
        f = (run['realized'].pivot(index='origin_id', columns='destination_id',
                                   values='persons_sample')
             .reindex(index=origins, columns=dest).fillna(0).to_numpy(float))
        grid.append(f)
    keep=np.logical_and.reduce([f.sum(axis=1)>0 for f in grid])
    if not keep.any():
        warnings.warn('No shared synthetic-workforce origins; skipping realized OD chart')
        return
    reduced_origins=np.array(origins)[keep].tolist()
    normalized=[f[keep]/f[keep].sum(axis=1)[:,None] for f in grid]
    obs=results[0]['hts'][keep]
    limit=max(float(np.max(x)) for x in [obs]+normalized)
    diff_limit=max(float(np.max(np.abs(obs-x))) for x in normalized)
    diff_limit=max(diff_limit,1e-9)
    fig,axes=plt.subplots(len(runs),3,figsize=(16,max(4.3*len(runs),5.2)),
                          squeeze=False,constrained_layout=True)
    for i,(run,sim) in enumerate(zip(runs,normalized)):
        images=[_heat(axes[i,0],obs,f'{run["name"]} | HTS',0,limit,'viridis',reduced_origins,dest),
                _heat(axes[i,1],sim,'Realized synthetic',0,limit,'viridis',reduced_origins,dest),
                _heat(axes[i,2],obs-sim,'HTS − synthetic',-diff_limit,diff_limit,'RdBu_r',
                      reduced_origins,dest,center=0)]
    fig.colorbar(images[0],ax=axes[:,:2].ravel().tolist(),shrink=.66,label='Conditional OD probability')
    fig.colorbar(images[-1],ax=axes[:,2].ravel().tolist(),shrink=.66,label='HTS minus synthetic conditional probability')
    fig.suptitle('Realized synthetic OD versus HTS | shared observed origin districts',fontsize=15)
    _save(fig,folder/'06_realized_synthetic_od',plt)
    for run,sim in zip(runs,normalized):
        pd.DataFrame({'origin_id':np.repeat(reduced_origins,len(dest)),
                      'destination_id':np.tile(dest,len(reduced_origins)),
                      'hts_probability':obs.ravel(),
                      'synthetic_conditional_probability':sim.ravel(),
                      'hts_minus_synthetic_probability':(obs-sim).ravel()}).to_csv(
             folder/f"realized_od_comparison_{run['name']}.csv",index=False)


def _plot_workplaces(runs, results, dest, folder):
    plt, fig, axes = _grid(len(runs), 5.1, 5.4)
    h = results[0]['hts_dest']
    vmax = max(float(np.nanmax(v)) for d in results for v in (d['hts_dest'],d['final_dest'],d['model_dest']))*1.12
    x = np.arange(len(dest))
    for ax,run,data in zip(axes,runs,results):
        ax.bar(x-.28, 100*h, width=.27, label='HTS observed',color='#365c78')
        ax.bar(x, 100*data['model_dest'], width=.27,label='Gravity prediction¹',color='#549eab')
        ax.bar(x+.28, 100*data['final_dest'],width=.27,label='Synthetic workplaces²',color='#e2a361')
        ax.set_xticks(x,dest,rotation=60,ha='right',fontsize=8)
        ax.set_ylim(0,100*vmax)
        ax.set_ylabel('Destination share (%)')
        ax.set_title(run['name'].replace('_',' '),loc='left')
        ax.grid(axis='y',alpha=.17)
    axes[0].legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('Workplace distribution by district — HTS as reference',fontsize=15)
    fig.supxlabel('¹ Gravity probabilities aggregated using the same HTS origin weights.  ² Distinct simulated people with a geocoded work activity.',fontsize=9)
    _save(fig,folder/'03_workplace_destination_shares',plt)


def _plot_jobs(hts, runs, results, dest, folder):
    plt = _figure_setup()
    job_sets=[_get_jobs(r,dest) for r in runs]
    valid=[(r['name'], j) for r,j in zip(runs,job_sets) if j is not None]
    if not valid:
        warnings.warn('No 2011 employment attraction data archived: skipping jobs comparison')
        return None
    # Most models share the same 2011 job distribution. If not, show all distinct versions.
    unique=[]
    for name,j in valid:
        if not any(np.allclose(j,existing[1],rtol=1e-6,atol=1e-8) for existing in unique):
            unique.append((name,j))
    fig,ax=plt.subplots(figsize=(max(11,len(dest)*.72),5.9),layout='constrained')
    h=results[0]['hts_dest']; x=np.arange(len(dest))
    width=.8/(1+len(unique))
    ax.bar(x-.4+width/2,100*h,width,label='HTS workplace destinations (reference)',color='#365c78')
    palette=['#c4795f','#98927c','#8e78a9']
    for i,(name,share) in enumerate(unique):
        label='2011 workplace personnel (attraction)' if len(unique)==1 else f'2011 job attraction — {name}'
        ax.bar(x-.4+(i+1.5)*width,100*share,width,label=label,color=palette[i%len(palette)])
    ax.set(xticks=x,xticklabels=dest,ylabel='Share of workplace destinations (%)')
    ax.tick_params(axis='x',labelrotation=60)
    for lab in ax.get_xticklabels(): lab.set_ha('right')
    ax.legend(frameon=False,loc='upper right')
    ax.grid(axis='y',alpha=.17)
    ax.set_title('Why 2011 job totals are not a reliable destination constraint',loc='left',fontsize=14)
    fig.text(.02,-.02,'Shares remove uniform rescaling; the comparison shows spatial distribution, not equal survey/census coverage.\n'
                       '2011 economic census excludes sectors and predates the HTS; its shares should not be treated as ground truth.',fontsize=9)
    _save(fig,folder/'04_hts_vs_2011_job_attraction',plt)
    rows=[]
    for i,d in enumerate(dest):
        r={'destination_id':d,'hts_share':h[i]}
        for name, j in unique:
            r['jobs_share_'+re.sub('[^A-Za-z0-9]+','_',name)]=j[i]
        rows.append(r)
    pd.DataFrame(rows).to_csv(folder/'2011_jobs_vs_hts_shares.csv',index=False)
    return unique


# Stable across every metric panel; not reused for different runs in one figure.
RUN_PALETTE = ('#28689B', '#D08332', '#358A78', '#8D67A8',
               '#B75C70', '#758D38', '#795548', '#657885')


def _plot_model_comparison(runs, results, folder):
    plt = _figure_setup()
    labels = [r['name'].replace('_', '\n') for r in runs]
    colors = [RUN_PALETTE[i % len(RUN_PALETTE)] for i in range(len(runs))]
    metrics = [('correlation', 'OD Pearson r ↑'),
               ('origin_weighted_tv', 'Origin-weighted TV ↓'),
               ('nll', 'Mean HTS negative log-likelihood ↓'),
               ('synthetic_destination_share_tv', 'Synthetic workplace TV vs HTS ↓')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, (key, title) in zip(axes.flat, metrics):
        val = np.array([d['metrics'][key] for d in results], dtype=float)
        x = np.arange(len(val))
        ax.bar(x, val, color=colors, width=.67)
        ax.set_xticks(x, labels, fontsize=8)
        top = np.nanmax(val) if np.isfinite(val).any() else 1.0
        ax.set_ylim(0, max(.01, top * 1.22))
        for i, v in enumerate(val):
            if np.isfinite(v):
                ax.text(i, v + top * .025, f'{v:.3f}', ha='center', fontsize=9)
        ax.set_title(title, loc='left')
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle('Common-support gravity model evaluation', fontsize=15)
    _save(fig, folder / '05_cross_model_scorecard', plt)


def _model_distance_function(run):
    """Older PPML archives omit this key, but their gravity uses a power law."""
    value = run['calibration'].get('gravity_model_type')
    if value is None or str(value).strip().lower() in ('', 'none', 'null'):
        value = run['config'].get('gravity_model_type')
    if value is None or str(value).strip().lower() in ('', 'none', 'null'):
        value = 'power_law'
    return str(value)


def _parameter_summary(run, model_type):
    """Display only the coefficients the given model actually defines/archives."""
    c = run['calibration']
    def fmt(key, label, precision=4):
        value = c.get(key)
        if value is None:
            return None
        try:
            v = float(value)
        except (ValueError, TypeError):
            return None
        if not np.isfinite(v):
            return None
        return f'{label} = {v:.{precision}g}'

    terms = []
    for key, label in [('k', 'k'), ('alpha', 'α'), ('beta', 'β')]:
        term = fmt(key, label)
        if term:
            terms.append(term)
    if model_type == 'exponential':
        # New calibration exports both 'slope' and the generic parameter.
        distance_key = 'slope' if c.get('slope') is not None else 'distance_parameter'
        term = fmt(distance_key, 'λ (km⁻¹)')
    else:
        distance_key = 'gamma' if c.get('gamma') is not None else 'distance_parameter'
        term = fmt(distance_key, 'γ')
    if term:
        terms.append(term)
    # The old unconstrained PPML model has no extra intrazonal factor. Do
    # not print an empty q column or imply that its q was fitted.
    term = fmt('intra_multiplier', 'q')
    if term:
        if run['config'].get('gravity_use_intra_multiplier') is False:
            term += ' (fixed)'
        terms.append(term)
    return '    ·    '.join(terms) if terms else 'Coefficients not recorded in archive'


def _plot_parameters(runs, folder):
    plt = _figure_setup()
    keys = ['gravity_model_type', 'k', 'alpha', 'beta', 'gamma', 'slope',
            'distance_parameter', 'intra_multiplier', 'intra_km']
    rows = []
    for run in runs:
        c = run['calibration']
        model_type = _model_distance_function(run)
        estimator = 'PPML' if c.get('alpha') is not None or c.get('k') is not None else 'Conditional HTS choice fit'
        row = {'run': run['name'], 'calibration_method': estimator,
               'distance_function': model_type,
               'parameters': _parameter_summary(run, model_type)}
        # Keep original numeric fields in CSV for downstream analysis; the
        # publication figure intentionally shows only relevant coefficients.
        row.update({key: c.get(key) for key in keys})
        row['gravity_model_type'] = model_type
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(folder / 'calibrated_parameters.csv', index=False)

    fig, ax = plt.subplots(figsize=(16, max(3.4, 1.0 * len(rows) + 1.3)))
    ax.axis('off')
    body = [[textwrap.fill(r['run'].replace('_', ' '), width=26, break_long_words=False), r['calibration_method'],
             'Exponential' if r['distance_function'] == 'exponential' else
             'Power law' if r['distance_function'] == 'power_law' else r['distance_function'],
             r['parameters']] for r in rows]
    tb = ax.table(cellText=body,
                  colLabels=['Model run', 'Calibration', 'Distance function', 'Calibrated coefficients'],
                  loc='center', cellLoc='left', colLoc='left',
                  colWidths=[.21, .20, .13, .46], bbox=[0, .19, 1, .66])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor('#dee4e9')
        cell.set_linewidth(.6)
        if i == 0:
            cell.set_facecolor('#294d64')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f3f7f9' if i % 2 == 0 else 'white')
            if j == 0:
                cell.set_text_props(weight='bold', color=RUN_PALETTE[(i-1) % len(RUN_PALETTE)])
    ax.set_title('Calibrated gravity parameters', loc='left', pad=22, fontsize=15)
    fig.text(.02, .065,
             'Only available model parameters are shown. k and α cancel during origin normalization; '
             'q is the intrazonal multiplier (when defined).',
             fontsize=9, color='#465765')
    _save(fig, folder / '00_calibrated_parameters', plt)


def _write_tables(runs, results, dest, origins, folder):
    metrics=pd.DataFrame([d['metrics'] for d in results]);metrics.to_csv(folder/'model_metrics.csv',index=False)
    workplace=[]
    for run,d in zip(runs,results):
        for i,zone in enumerate(dest):
            workplace.append({'run':run['name'],'destination_id':zone,
                              'hts_destination_share':d['hts_dest'][i],
                              'model_hts_origin_weighted_destination_share':d['model_dest'][i],
                              'synthetic_work_activity_destination_share':d['final_dest'][i]})
        # Long format: all cells, including HTS zero flows.
        table=pd.DataFrame({'origin_id':np.repeat(origins,len(dest)),
                            'destination_id':np.tile(dest,len(origins)),
                            'hts_probability':d['hts'].ravel(),
                            'model_probability':d['model'].ravel(),
                            'hts_minus_model_probability':d['delta'].ravel(),
                            'hts_weighted_flow':d['hts_flow'].ravel(),
                            'model_hts_origin_weighted_flow':d['model_flow'].ravel()})
        table.to_csv(folder/f"od_comparison_{run['name']}.csv",index=False)
    pd.DataFrame(workplace).to_csv(folder/'workplace_destination_shares.csv',index=False)


def _select_archives(root):
    return [x for x in sorted(root.iterdir()) if x.is_dir() and
            (x/'analysis/model_probabilities.csv').exists() and not x.name.startswith('_')]


def create_report(archive_root, hts_observed, output=None):
    """Pure-Python entry point for offline tests, and the SynPP execute function."""
    root=Path(archive_root).resolve()
    if not root.exists(): raise FileNotFoundError(root)
    folders=_select_archives(root)
    if len(folders)<2: raise ValueError(f'Need at least 2 complete archived models in {root}; found {len(folders)}')
    runs=[_read_run(x) for x in folders]
    hts=_od(hts_observed)
    digest=_canonical_hts_hash(hts)
    # Warn rather than quietly mix different reference observations.
    for run in runs:
        fingerprint=run['reference'] or {}
        previous=fingerprint.get('hts_observed_od_sha256')
        if previous and previous!=digest:
            raise ValueError(f"{run['name']}: archived HTS fingerprint differs from current HTS OD. "
                             'Do not compare different survey references.')
    geometry_hashes={r['reference'].get('model_zones_geometry_sha256') for r in runs if r['reference'] and r['reference'].get('model_zones_geometry_sha256')}
    if len(geometry_hashes)>1:
        raise ValueError('Archived spatial zone boundaries differ between runs')
    origins,dest=_reference_data(hts,runs)
    results=[_make_matrices(hts,r,origins,dest) for r in runs]
    out=Path(output) if output is not None else root/'_comparison'
    out.mkdir(parents=True,exist_ok=True)
    _plot_parameters(runs,out)
    _plot_scatter(runs,results,out)
    _plot_od(runs,results,origins,dest,out)
    _plot_workplaces(runs,results,dest,out)
    _plot_realized_od(runs,results,origins,dest,out)
    _plot_jobs(hts,runs,results,dest,out)
    _plot_model_comparison(runs,results,out)
    _write_tables(runs,results,dest,origins,out)
    excluded=hts.weight.sum()-results[0]['hts_flow'].sum()
    metadata={'runs':[r['name'] for r in runs], 'hts_reference_sha256':digest,
              'common_origins':origins,'common_destinations':dest,
              'hts_weight_total':float(hts.weight.sum()),
              'hts_weight_on_common_od_support':float(results[0]['hts_flow'].sum()),
              'hts_weight_excluded':float(excluded),
              'reference_zones_geometry_sha256':next(iter(geometry_hashes),None),
              'interpretation':{
                  'gravity_od':'Conditional P(work zone | home zone), common HTS origin mix',
                  'synthetic_od':'Distinct people with a geocoded work activity on modeled day; different home mix from HTS',
                  'census_jobs':'2011 workplace personnel distribution, incomplete sector coverage; illustrative only',
                  'correlation':'Pearson across all common conditional OD pairs, including HTS zero cells',
                  'comparison':'In-sample if HTS is used to calibrate each model; no out-of-sample performance claim'}}
    (out/'comparison_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    readme='''# Gravity model comparison — figure guide\n\n'''+ '\n'.join([
        '- `00_calibrated_parameters`: fitted coefficients; PPML versus conditional estimation not identical.',
        '- `01_od_correlation`: common-support row-normalized probabilities; fixed axes.',
        '- `02_od_matrices_and_differences`: HTS, predicted and HTS-minus-model; fixed common heatmap scales.',
        '- `03_workplace_destination_shares`: common HTS reference, model prediction weighted by HTS home totals, realized simulated workers.',
        '- `04_hts_vs_2011_job_attraction`: evidence for mismatch with dated, incomplete workplace personnel data.',
        '- `05_cross_model_scorecard`: cross-run fit metrics calculated on identical OD support.',
        '- `06_realized_synthetic_od`: HTS versus actual unique-person synthetic OD, with common scales.',
        '\n**Caveats:** All model-vs-HTS fit is in-sample unless HTS respondents were held out.',
        'Work activity people are not identical to all employed residents or physical establishments.',
        'The 2011 personnel distribution is not a valid 2022 workplace ground truth.',
        'Changing HTS cleaning, OD support, spatial definitions, or synthetic sampling across runs',
        'can compromise causal attribution of differences to the gravity specification.',
        'All figures are saved in vector PDF and 260 dpi PNG formats.\n'])
    (out/'README.md').write_text(readme,encoding='utf-8')
    print(f'Compared {len(runs)} gravity runs; saved report figures under {out}')
    return str(out)


def execute(context):
    archive_root=Path(context.config('analysis_path'))/'gravity_runs'
    observed=context.stage('asuncion.gravity.observed')
    return create_report(archive_root,observed)
