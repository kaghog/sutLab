"""Publication-oriented synthetic-population validation for Asuncion.
SynPP stage: analysis.asuncion.asuncion_validate_population.

HTS 'filtered' supplies households and person attributes; HTS 'reweighted'
supplies trip-purpose observations. Census employment and income stages provide
separate references. Outputs: PDF, 260 dpi PNG, machine-readable CSVs, methods
and quality warnings under <analysis_path>/population_validation.

Important: HTS has no recorded household district, so home district is inferred
from trip endpoints (explicitly partial coverage). HTS education participation
is inferred from trip purposes, not the 'studies' flag; it is a day-level proxy
for education participation, not an enrollment register.
"""
from pathlib import Path
import json
import shutil
import re
import textwrap
import warnings
import numpy as np
import pandas as pd

from asuncion.data.census.income import (
    ASUNCION_INCOME_CLASS_BINS,
    ASUNCION_INCOME_CLASS_LABELS,
)

BLUE='#28689B'; ORANGE='#D08332'; TEAL='#358A78'; INK='#213240'; GREY='#687782'
# Five-year age bands, covering 0–4, 5–9, ..., 95–99, and 100+.
AGE_BINS=list(range(0,105,5))+[np.inf]
AGE_NAMES=[f'{start}–{start+4}' for start in range(0,100,5)]+['100+']
# Census age_class codes 0, 5, 10, ..., 60 represent separate five-year
# intervals; code 65 is a single open-ended 65+ interval in BOTH census inputs.
# Employment data do not contain ages 0-14: count plots include these classes
# as zero census-reference rows to reveal any synthetic under-15 employment.
# Rate plots distinguish unavailable census under-15 rates from measured zeros.
AGE_BIN_MAPPING={start:start for start in range(0,70,5)}
CENSUS_AGE_BINS=list(range(0,70,5))+[np.inf]
CENSUS_AGE_NAMES=[f'{start}–{start+4}' for start in range(0,65,5)]+['65+']
EMPLOYMENT_AGE_NAMES=CENSUS_AGE_NAMES.copy()
CENSUS_UNDER_15=CENSUS_AGE_NAMES[:3]
# Reuse the authoritative class boundaries and labels used by the census stage.
INCOME_BINS=list(ASUNCION_INCOME_CLASS_BINS)
SIZE_NAMES=['1','2','3','4','5','6+']
INCOME_NAMES=list(ASUNCION_INCOME_CLASS_LABELS)
OTHER_EXCLUDE={'home','work','education','unknown','none','nan',''}


def configure(context):
    for stage in ('synthesis.output','asuncion.data.hts.entd.filtered',
                  'asuncion.data.hts.entd.reweighted','asuncion.data.census.employment',
                  'asuncion.data.census.income','asuncion.data.census.households',
                  'asuncion.data.census.population','asuncion.gravity.od_zones',
                  'asuncion.data.codes'):
        context.stage(stage)
    for k in ('analysis_path','output_path','output_prefix','sampling_rate'):
        context.config(k)
    context.config('population_validation_income_scale',1.0)
    # Generate the condensed age-by-district figure by default; the 21 individual
    # five-year district map panels per attribute can be explicitly requested.
    context.config('population_validation_detailed_age_maps', False)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white',
        'savefig.facecolor':'white','axes.grid':False})
    return plt


def _save(fig,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(p.with_suffix('.png'),bbox_inches='tight',dpi=260)
    _plt().close(fig)


def _id(series):
    return series.astype('string').str.strip().str.replace(r'^(\d+)\.0$',r'\1',regex=True)


def _binary(s):
    a=s.astype('string').str.strip().str.lower()
    return a.map({'true':1.,'false':0.,'yes':1.,'no':0.,'si':1.,'sí':1.,'1':1.,'0':0.,'t':1.,'f':0.})


def _require(df,keys,tag):
    missing=set(keys)-set(df.columns)
    if missing: raise ValueError(f'{tag} missing columns {sorted(missing)}')


def _read(path):
    if not path.is_file(): raise FileNotFoundError(path)
    return pd.read_csv(path,sep=';',low_memory=False)


def _synthetic(root,prefix):
    import geopandas as gpd
    root=Path(root)
    result=tuple(_read(root/f'{prefix}{name}.csv') for name in ('persons','households','trips'))
    def geo(name,needed=True):
        gpkg=root/f'{prefix}{name}.gpkg'; parquet=root/f'{prefix}{name}.geoparquet'
        if gpkg.exists():return gpd.read_file(gpkg)
        if parquet.exists():return gpd.read_parquet(parquet)
        if needed: raise FileNotFoundError(f'Need {gpkg} or {parquet}')
        return None
    return (*result,geo('activities'),geo('homes',False))


def _zones(payload,codes):
    import geopandas as gpd
    if not isinstance(payload,(list,tuple)) or len(payload)<4:
        raise ValueError('od_zones must return (zones,pop_centroids,employment_centroids,commune2macrozone_map)')
    geom,_,_,mapping=payload[:4]
    if not isinstance(geom,gpd.GeoDataFrame): raise TypeError('od_zones[0] must be GeoDataFrame')
    _require(geom,['macrozone_id','geometry'],'zones')
    z=geom[['macrozone_id','geometry']].copy().rename(columns={'macrozone_id':'zone_id'})
    z.zone_id=_id(z.zone_id)
    if z.crs is None: raise ValueError('Zones have no CRS')
    if z.zone_id.duplicated().any():z=z.dissolve(by='zone_id',as_index=False)
    if isinstance(mapping,dict): m=mapping
    elif isinstance(mapping,pd.Series):m=mapping.to_dict()
    elif isinstance(mapping,pd.DataFrame) and len(mapping.columns)>1:
        src=next((c for c in ('commune_id','district_id') if c in mapping),mapping.columns[0])
        tgt=next((c for c in ('macrozone_id','zone_id') if c in mapping),mapping.columns[1])
        m=dict(zip(mapping[src],mapping[tgt]))
    else:m={}
    conv={str(k).strip().removesuffix('.0'):str(v).strip().removesuffix('.0') for k,v in m.items() if pd.notna(k) and pd.notna(v)}
    valid=set(z.zone_id)
    def translate(series):
        v=_id(series);dest=v.map(conv)
        return dest.fillna(v.where(v.isin(valid))).where(lambda x:x.isin(valid))
    z['label']=z.zone_id
    if isinstance(codes,pd.DataFrame):
        i=next((c for c in ('district_id','commune_id','macrozone_id') if c in codes),None)
        n=next((c for c in ('district_name','commune_name','district','commune','name','nombre') if c in codes),None)
        if i and n:
            labels=codes[[i,n]].dropna().copy();labels['zone_id']=translate(labels[i])
            by=labels.dropna(subset=['zone_id']).groupby('zone_id')[n].agg(lambda s:' / '.join(sorted(set(map(str,s))))).to_dict()
            z['label']=z.zone_id.map(by).fillna(z.zone_id)
    return z,translate,conv


def _locate(points,zones,keys):
    import geopandas as gpd
    _require(points,keys+['geometry'],'spatial points')
    if points.crs is None:raise ValueError('Point file missing CRS')
    allkeys=points[keys].drop_duplicates()
    if len(allkeys)!=len(points):raise ValueError(f'Duplicate point keys: {keys}')
    p=points[keys+['geometry']].dropna(subset=['geometry'])
    target=zones[['zone_id','geometry']].to_crs(points.crs)
    matched=gpd.sjoin(p,target,how='left',predicate='within')
    missing=matched[matched.zone_id.isna()][keys+['geometry']]
    matched=matched.dropna(subset=['zone_id'])
    if len(missing):
        boundary=gpd.sjoin(missing,target,how='left',predicate='intersects')
        matched=pd.concat([matched,boundary.dropna(subset=['zone_id'])],ignore_index=True)
    matched=matched.drop_duplicates(keys)
    return allkeys.merge(matched[keys+['zone_id']],on=keys,how='left',validate='one_to_one')


def _hts_homes(persons,trips,translate):
    """Location from actual home activity endpoints, never from departement_id."""
    _require(trips,['person_id','origin_district_id','destination_district_id',
                    'preceding_purpose','following_purpose'],'filtered HTS trips')
    p=persons[['person_id','household_id']].copy()
    p.person_id=_id(p.person_id);p.household_id=_id(p.household_id)
    t=trips.copy();t.person_id=_id(t.person_id)
    x=t[t.preceding_purpose.astype('string').str.lower().eq('home')][['person_id','origin_district_id']].rename(columns={'origin_district_id':'district'})
    y=t[t.following_purpose.astype('string').str.lower().eq('home')][['person_id','destination_district_id']].rename(columns={'destination_district_id':'district'})
    a=pd.concat([x,y]).merge(p,on='person_id',how='inner')
    a['zone_id']=translate(a.district);a=a.dropna(subset=['zone_id'])
    votes=a.groupby(['household_id','zone_id']).size().reset_index(name='votes')
    if votes.empty:return pd.DataFrame(columns=['household_id','zone_id']),0
    votes=votes.sort_values(['household_id','votes','zone_id'],ascending=[True,False,True])
    top=votes.drop_duplicates('household_id')
    ntop=votes.merge(top[['household_id','votes']],on=['household_id','votes']).groupby('household_id').size()
    tied=set(ntop[ntop>1].index)
    return top[~top.household_id.isin(tied)][['household_id','zone_id']],len(tied)


def _pct(series):
    s=series.astype(float)
    return s/s.sum() if s.sum()>0 else s*np.nan


def _weighted(df,column,weight):
    x=df[[column,weight]].dropna().copy()
    x[weight]=pd.to_numeric(x[weight],errors='coerce')
    x=x[np.isfinite(x[weight])&(x[weight]>0)]
    return x.groupby(column)[weight].sum()

# ---------------------------------------------------------------------------
# Report figures (matching the compare_models style)
# ---------------------------------------------------------------------------

def _bar_categories(title,subtitle,categories,ref,syn,path,reference='HTS',color=BLUE,
                    share_label='Share of all observations in reference area (%)'):
    plt=_plt();cats=list(categories)
    r=pd.Series(ref,dtype=float).reindex(cats,fill_value=0)
    s=pd.Series(syn,dtype=float).reindex(cats,fill_value=0)
    export=pd.DataFrame({'category':cats,'reference_share':_pct(r).values,'synthetic_share':_pct(s).values})
    export.to_csv(path.with_suffix('.csv'),index=False)
    fig,ax=plt.subplots(figsize=(max(9.2,len(cats)*.70),5.2),layout='constrained')
    x=np.arange(len(cats));w=.37
    ax.bar(x-w/2,export.reference_share*100,w,color=color,label=reference)
    ax.bar(x+w/2,export.synthetic_share*100,w,color=ORANGE,label='Synthetic')
    rotation=50 if len(cats)>10 else 0
    ax.set_xticks(x,cats,rotation=rotation,ha='right' if rotation else 'center',fontsize=8.5 if rotation else 9)
    ax.set_ylabel(share_label);ax.set_ylim(bottom=0)
    ax.yaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
    ax.set_title(title,loc='left',fontsize=15,weight='bold',pad=14)
    ax.text(0,-.27 if rotation else -.19,subtitle,transform=ax.transAxes,
            color=GREY,fontsize=8.5,va='top')
    ax.legend(frameon=False,ncol=2)
    _save(fig,path)


def _paired_geo(zones,ref,syn,title,filename,out,reference='HTS',color=BLUE,
                percent=True,note='',scale=None,measure=None,
                share_label='Share of all observations in reference area (%)'):
    """District horizontal paired bars (left) + synthetic-reference map (right)."""
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    plt=_plt()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    z=zones[['zone_id','label','geometry']].copy()
    z['reference']=z.zone_id.map(pd.Series(ref,dtype=float))
    z['synthetic']=z.zone_id.map(pd.Series(syn,dtype=float))
    z['delta']=z.synthetic-z.reference
    z.drop(columns='geometry').to_csv(out/f'{filename}.csv',index=False)
    z=z.sort_values('reference',na_position='first')
    factor=100.0 if percent else 1.0
    n=len(z);h=max(6.4,min(18.5,2.0+n*.35))
    fig,(ax,mapax)=plt.subplots(1,2,figsize=(16,h),
            gridspec_kw={'width_ratios':[1.25,1]},layout='constrained')
    y=np.arange(n);w=.38
    ax.barh(y-w/2,z.reference.to_numpy(float)*factor,w,color=color,label=reference)
    ax.barh(y+w/2,z.synthetic.to_numpy(float)*factor,w,color=ORANGE,label='Synthetic')
    labels=[textwrap.shorten(str(v),width=30,placeholder='…') for v in z.label]
    ax.set_yticks(y,labels,fontsize=8)
    ax.set_xlabel(share_label if percent else (measure or 'Value'))
    ax.xaxis.grid(True,alpha=.16);ax.set_axisbelow(True)
    ax.legend(ncol=2,frameon=False,loc='lower right')
    ax.set_title('A   District comparison',loc='left',weight='bold')
    d=z.delta.to_numpy(float)*factor
    cap=max(float(np.nanmax(abs(d))) if np.isfinite(d).any() else 0.,.01)
    if scale is not None:cap=max(float(scale),.01)
    norm=TwoSlopeNorm(vmin=-cap,vcenter=0,vmax=cap)
    z.plot(column='delta',ax=mapax,cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-cap/factor,vcenter=0,vmax=cap/factor),
        linewidth=.6,edgecolor='#f2f5f8',missing_kwds={'color':'#dde2e6','edgecolor':'white'})
    mapax.set_axis_off()
    mapax.set_title('B   Synthetic − reference',loc='left',weight='bold')
    m=ScalarMappable(norm=norm,cmap='RdBu_r');m.set_array([])
    cb=fig.colorbar(m,ax=mapax,shrink=.7,pad=.015)
    cb.set_label('Synthetic − reference (percentage points)' if percent else f'Synthetic − reference ({measure or "units"})')
    fig.suptitle(title,x=.04,y=1+.70/h,ha='left',fontsize=15,weight='bold')
    if note: fig.supxlabel(textwrap.fill(note,115),x=.04,ha='left',fontsize=8.5,color=GREY)
    _save(fig,out/filename)


def _age_summary(df,value,weight):
    x=df[['age',value,weight,'zone_id']].copy()
    x['group']=pd.cut(pd.to_numeric(x.age,errors='coerce'),AGE_BINS,
        labels=AGE_NAMES,right=False,include_lowest=True)
    x['v']=_binary(x[value])
    x['w']=pd.to_numeric(x[weight],errors='coerce')
    x=x[x.v.notna() & x.w.notna() & (x.w>0)]
    x['yes']=x.v*x.w;x['wsq']=x.w*x.w
    def aggregate(keys):
        g=x.groupby(keys,observed=True,dropna=True).agg(
            yes=('yes','sum'),total=('w','sum'),sq=('wsq','sum'),n=('v','size')).reset_index()
        g['rate']=g.yes/g.total
        g['effective_n']=g.total*g.total/g.sq
        return g
    return aggregate(['group']),aggregate(['zone_id']),aggregate(['zone_id','group'])


def _age_bars(a,b,title,filename,out,reference='HTS',color=BLUE,note='',
              share_label='People with attribute, among all residents in age group (%)',
              age_names=None):
    age_names=list(AGE_NAMES if age_names is None else age_names)
    plt=_plt();r=a.set_index('group').reindex(age_names);s=b.set_index('group').reindex(age_names)
    csv=pd.DataFrame({'age_group':age_names,'reference_rate':r.rate.to_numpy(float),
       'synthetic_rate':s.rate.to_numpy(float),
       'reference_effective_n':r.effective_n.to_numpy(float),
       'synthetic_n':s.n.to_numpy(float)})
    csv.to_csv(out/f'{filename}.csv',index=False)
    fig,ax=plt.subplots(figsize=(max(12.5,len(age_names)*.72),5.5),layout='constrained')
    x=np.arange(len(age_names));w=.36
    ax.bar(x-w/2,csv.reference_rate*100,w,color=color,label=reference)
    ax.bar(x+w/2,csv.synthetic_rate*100,w,color=ORANGE,label='Synthetic')
    ax.set(ylim=(0,100),xlabel='Age (years)',ylabel=share_label)
    ax.set_xticks(x,age_names,rotation=50,ha='right',fontsize=8)
    ax.set_title(title,loc='left',weight='bold',fontsize=14)
    ax.yaxis.grid(True,alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False)
    if note:fig.text(.07,-.01,textwrap.fill(note,150),fontsize=8.5,color=GREY)
    _save(fig,out/filename)


def _age_heatmaps(zones,a,b,title,filename,out,reference='HTS',ref_color='Blues',
                  min_ref=8,min_syn=8,
                  share_label='Within each district × age group: attribute rate (%)'):
    """One row per district, fixed [0,100] rate scales, symmetric difference."""
    from matplotlib.colors import TwoSlopeNorm
    plt=_plt();ids=zones.zone_id.tolist()
    def mat(df,sample,mincount):
        x=df.copy();x.loc[x[sample]<mincount,'rate']=np.nan
        return (x.pivot(index='zone_id',columns='group',values='rate')
                .reindex(index=ids,columns=AGE_NAMES).to_numpy(float))
    r=mat(a,'effective_n',min_ref);s=mat(b,'n',min_syn)
    delta=(s-r)*100
    pd.DataFrame([{'zone_id':z,'age_group':age,'reference_rate':r[i,j],
        'synthetic_rate':s[i,j],'difference_pp':delta[i,j]}
        for i,z in enumerate(ids) for j,age in enumerate(AGE_NAMES)]).to_csv(out/f'{filename}.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(21.,max(6.,len(ids)*.28+2)),layout='constrained')
    cap=max(float(np.nanmax(abs(delta))) if np.isfinite(delta).any() else 0.,5.)
    for ax,arr,name,cmap in zip(axs,[r*100,s*100,delta],
                                 [reference,'Synthetic','Synthetic − reference'],
                                 [ref_color,'Oranges','RdBu_r']):
        cmap=plt.get_cmap(cmap).copy();cmap.set_bad('#e5e9ed')
        opts={'vmin':0,'vmax':100} if name!='Synthetic − reference' else {
            'norm':TwoSlopeNorm(vmin=-cap,vcenter=0,vmax=cap)}
        im=ax.imshow(arr,aspect='auto',cmap=cmap,**opts)
        fig.colorbar(im,ax=ax,shrink=.7,pad=.01,
                    label='Synthetic − reference (pp)' if name=='Synthetic − reference' else share_label)
        ax.set_title(name,loc='left',weight='bold')
        ax.set_xticks(range(len(AGE_NAMES)),AGE_NAMES,rotation=55,ha='right',fontsize=8)
        ax.set_yticks(range(len(ids)),[textwrap.shorten(str(v),width=23,placeholder='…')
                         for v in zones.label],fontsize=7.5)
        ax.tick_params(length=0)
    fig.suptitle(title,x=.04,y=1+.58/max(6.,len(ids)*.28+2),ha='left',weight='bold',fontsize=15)
    fig.text(.04,-.009,f'Grey cells: reference effective sample < {min_ref} or synthetic people < {min_syn}.',
             color=GREY,fontsize=8.5)
    _save(fig,out/filename)


def _age_district_bars_maps(zones,a,b,title_prefix,folder,reference='HTS',color=BLUE,
                             share_label='Within district × age group: people with attribute (%)'):
    """Age-specific district figure set; identical residual map scale per attribute."""
    ra=a.pivot(index='zone_id',columns='group',values='rate').reindex(columns=AGE_NAMES)
    sy=b.pivot(index='zone_id',columns='group',values='rate').reindex(columns=AGE_NAMES)
    rn=a.pivot(index='zone_id',columns='group',values='effective_n').reindex(columns=AGE_NAMES)
    sn=b.pivot(index='zone_id',columns='group',values='n').reindex(columns=AGE_NAMES)
    r=ra.where(rn>=8);s=sy.where(sn>=8)
    differences=(s-r).abs()*100
    cap=max(float(np.nanmax(differences.to_numpy(float))) if np.isfinite(differences.to_numpy(float)).any() else 0.,2.)
    folder.mkdir(parents=True,exist_ok=True)
    for age in AGE_NAMES:
        slug=age.replace('–','_').replace('+','plus')
        _paired_geo(zones,r[age] if age in r else pd.Series(dtype=float),
                    s[age] if age in s else pd.Series(dtype=float),
                    f'{title_prefix} — ages {age}',f'age_{slug}',folder,
                    reference=reference,color=color,percent=True,scale=cap,
                    note='Denominator: residents in this district and five-year age group. Grey districts indicate insufficient sample; same map scale across age panels.',
                    share_label=share_label)

# ---------------------------------------------------------------------------
# Census references (respect denominator limitations)
# ---------------------------------------------------------------------------

def _age_range(label):
    """Parse an explicit textual census age interval (if one is supplied)."""
    s=str(label).strip().lower()
    m=re.search(r'(\d{1,3})\s*(?:-|–|a|to|_)\s*(\d{1,3})',s)
    if m:
        lo,hi=map(int,m.groups());return (lo,hi) if lo<=hi else None
    m=re.search(r'(\d{1,3})\s*(?:\+|y mas|y más|o mas|o más|or more)',s)
    if m:return int(m.group(1)),None
    m=re.search(r'(?:>=|≥|more than|mas de|más de|desde)\s*(\d{1,3})',s)
    if m:return int(m.group(1)),None
    return None


def _census_group(label):
    """Map the census's integer lower-bound codes to its *published* groups.

    Numeric lower-bound codes 0, 5, 10, ..., 60 designate distinct five-year
    groups; code 65 includes ALL residents aged 65 and above.
    Textual intervals are also accepted when their bounds are unambiguous.
    We never split the open-ended 65+ group into finer bands.
    """
    raw=str(label).strip()
    try:
        code=float(raw)
    except (ValueError,TypeError):
        code=None
    if code is not None and np.isfinite(code) and code.is_integer():
        start=AGE_BIN_MAPPING.get(int(code))
        if start is not None:
            if start==65:return '65+'
            return f'{start}–{start+4}'
    interval=_age_range(label)
    if interval is None:return None
    lo,hi=interval
    if lo==65 and hi is None:return '65+'
    if lo in range(0,65,5) and hi==lo+4:return f'{lo}–{hi}'
    return None


def _map_age_classes(values,tag,report):
    """Map the supplied census lower-bound age codes to comparable groups.

    '0', '5', and '10' mean 0–4, 5–9, and 10–14 respectively;
    '65' means 65+, not 65–69. Unknown age codes invalidate the comparison.
    """
    labels=[str(v).strip() for v in pd.Series(values).dropna().unique()]
    mapping={label:_census_group(label) for label in labels}
    report[f'{tag}_age_classes']=mapping
    if not mapping or any(g is None for g in mapping.values()):
        unknown=[k for k,v in mapping.items() if v is None]
        report['warnings'].append(
            f'{tag}: unsupported census age_class values {unknown!r}; age comparison omitted. '
            'Expected numeric class starts 0, 5, 10, ..., 60, 65 (65+), '
            'or matching explicit age intervals.')
        return None
    return mapping


def _census_age_for_people(ages):
    """Bin exact synthetic ages into the census-comparable age groups."""
    return pd.cut(pd.to_numeric(ages,errors='coerce'),CENSUS_AGE_BINS,
                  labels=CENSUS_AGE_NAMES,right=False,include_lowest=True)


def _absolute_age_count_table(zones, reference, synthetic_sample, sampling_rate, age_names=None):
    """Complete census-versus-expanded-synthetic district-by-age-group data.

    Both wide inputs are count matrices on the same zone/age support. Preserve
    the *unexpanded* synthetic sample for transparency. This is a count check,
    not an age-share or employment-rate check.
    """
    if not (0 < float(sampling_rate) <= 1):
        raise ValueError('sampling_rate must be in (0, 1]')
    age_names=list(AGE_NAMES if age_names is None else age_names)
    ids=zones.zone_id.astype(str).tolist()
    ref=reference.reindex(index=ids,columns=age_names,fill_value=0).astype(float)
    syn=synthetic_sample.reindex(index=ids,columns=age_names,fill_value=0).astype(float)
    if not (np.isfinite(ref.to_numpy()).all() and np.isfinite(syn.to_numpy()).all()):
        raise ValueError('Age count matrices must contain only finite numbers')
    if (ref.to_numpy()<0).any() or (syn.to_numpy()<0).any():
        raise ValueError('Age count matrices must not contain negative counts')
    labels=zones.set_index('zone_id')['label'].to_dict()
    records=[]
    for zone in ids:
        for age in age_names:
            census=float(ref.at[zone,age]); sample=float(syn.at[zone,age]); expanded=sample/float(sampling_rate)
            records.append({'zone_id':zone,'district':str(labels.get(zone,zone)),
                'age_group':age,'census_people':census,
                'synthetic_people_sample':sample,'synthetic_people_expanded':expanded,
                'synthetic_minus_census_people':expanded-census,
                'relative_difference_pct':100*(expanded-census)/census if census>0 else np.nan})
    return pd.DataFrame.from_records(records)


def _plot_age_count_totals(counts,title,filename,out,unit='Residents (people)',note='',age_names=None):
    """Absolute age-group counts: two side-by-side bars, not normalized shares."""
    from matplotlib.ticker import StrMethodFormatter
    plt=_plt()
    age_names=list(AGE_NAMES if age_names is None else age_names)
    overall=(counts.groupby('age_group',observed=True)[
        ['census_people','synthetic_people_sample','synthetic_people_expanded']]
        .sum().reindex(age_names,fill_value=0))
    overall['synthetic_minus_census_people']=(overall.synthetic_people_expanded-overall.census_people)
    overall['relative_difference_pct']=100*overall.synthetic_minus_census_people/overall.census_people.replace(0,np.nan)
    overall.to_csv(out/f'{filename}.csv',index_label='age_group')
    fig,ax=plt.subplots(figsize=(15.6,6.2),layout='constrained')
    x=np.arange(len(age_names));width=.37
    ax.bar(x-width/2,overall.census_people,width,color=TEAL,label='Census')
    ax.bar(x+width/2,overall.synthetic_people_expanded,width,color=ORANGE,label='Synthetic (expanded)')
    ax.set_xticks(x,age_names,rotation=50,ha='right',fontsize=8.5)
    ax.set_xlabel('Census-compatible age group (years)')
    ax.set_ylabel(unit+' — absolute count')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.grid(True,alpha=.16);ax.set_axisbelow(True);ax.set_ylim(bottom=0)
    ax.set_title(title,loc='left',fontsize=15,weight='bold',pad=13)
    ax.legend(frameon=False,ncol=2)
    if note:fig.supxlabel(textwrap.fill(note,160),fontsize=8.5,color=GREY)
    _save(fig,out/filename)


def _plot_district_age_count_pages(zones,counts,title,folder,unit='Residents (people)',
                                   districts_per_page=4,age_names=None,note=''):
    """Faceted *paired horizontal bars* show actual counts rather than heatmap hue.

    Four comparable district panels per page, ranked by census total. A common
    x-axis is used within each page (not silently rescaled between panels).
    CSV remains fully numeric and is not rounded or normalized.
    """
    from matplotlib.ticker import StrMethodFormatter
    plt=_plt();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    age_names=list(AGE_NAMES if age_names is None else age_names)
    ids=zones.zone_id.astype(str).tolist()
    available=(counts.groupby('zone_id').census_people.sum()
               .reindex(ids,fill_value=0).sort_values(ascending=False,kind='stable'))
    ids=available.index.tolist()
    labels=zones.set_index('zone_id')['label'].to_dict()
    page_summaries=[]
    for p,start in enumerate(range(0,len(ids),districts_per_page),1):
        page_ids=ids[start:start+districts_per_page]
        panels=[]
        for zid in page_ids:
            group=(counts[counts.zone_id==zid].set_index('age_group')
                   .reindex(age_names))
            panels.append((zid,group))
        # Sort similarly sized districts together, allowing meaningful paired
        # count bars even if the capital differs greatly in population size.
        max_value=max((float(np.nanmax(g[['census_people','synthetic_people_expanded']].to_numpy(float)))
                       for _,g in panels),default=0.)
        xlimit=max(max_value*1.09,1.)
        fig,axs=plt.subplots(2,2,figsize=(17,15.8),layout='constrained')
        for ax in axs.flat:ax.set_visible(False)
        for i,(zid,g) in enumerate(panels):
            ax=axs.flat[i];ax.set_visible(True)
            y=np.arange(len(age_names));h=.37
            ax.barh(y-h/2,g.census_people.to_numpy(float),h,color=TEAL,label='Census')
            ax.barh(y+h/2,g.synthetic_people_expanded.to_numpy(float),h,color=ORANGE,label='Synthetic (expanded)')
            ax.set_yticks(y,age_names,fontsize=8)
            ax.invert_yaxis();ax.set_xlim(0,xlimit)
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.xaxis.grid(True,alpha=.16);ax.set_axisbelow(True)
            ax.set_xlabel(unit+' by census-compatible age group')
            name=textwrap.shorten(str(labels.get(zid,zid)),width=37,placeholder='…')
            ref_total=float(g.census_people.sum());syn_total=float(g.synthetic_people_expanded.sum())
            ax.set_title(f'{name}  |  Census {ref_total:,.0f} · Synthetic {syn_total:,.0f}',
                         loc='left',weight='bold',fontsize=10,pad=9)
            if i==0:ax.legend(frameon=False,ncol=2,loc='lower right',fontsize=8)
        fig.suptitle(f'{title}  ·  page {p}',x=.035,ha='left',fontsize=15,weight='bold')
        fig.supxlabel('Absolute counts; synthetic sample expanded by 1/sampling_rate. '
            'The horizontal scale is identical for all districts ON THIS PAGE; '
            'districts are grouped by census totals.'+
            ('  '+note if note else ''),fontsize=9,color=GREY)
        stem=f'page_{p:02d}'
        _save(fig,folder/stem)
        page_summaries.append({'page':p,'districts':'; '.join(page_ids),
                               'shared_x_max':xlimit,'figure':str(folder/stem)})
    pd.DataFrame(page_summaries).to_csv(folder/'page_index.csv',index=False)
    return len(page_summaries)


def _census_age_population(census,persons,zones,translate,out,report,detailed_age_maps=False,sampling_rate=1.):
    """Census-compatible resident age composition by district, census reference.

    Uses separate five-year groups 0–4 through 60–64, plus 65+.
    """
    if not isinstance(census,pd.DataFrame):
        raise TypeError('Census population-age input must be a pandas DataFrame')
    _require(census,['commune_id','age_class','weight'],'asuncion.data.census.population')
    c=census.copy()
    c['zone_id']=translate(c.commune_id)
    c['weight']=pd.to_numeric(c.weight,errors='coerce')
    if ((c.weight.dropna()<0).any() or not np.isfinite(c.weight.dropna()).all()):
        raise ValueError('Census population age weights must be finite/nonnegative')
    c.to_csv(out/'data'/'census_population_age_mapped.csv',index=False)
    c=c.dropna(subset=['zone_id','weight','age_class'])
    if c.empty or c.weight.sum()<=0:
        report['census_population_age_status']='no_valid_weight'
        report['warnings'].append('Census population-age reference has no usable district records.')
        return None
    mapping=_map_age_classes(c.age_class,'census_population',report)
    if mapping is None:
        report['census_population_age_status']='incompatible_age_classes'
        return None
    c['age_group']=c.age_class.astype(str).map(mapping)
    ids=zones.zone_id.tolist()
    cr=(c.pivot_table(index='zone_id',columns='age_group',values='weight',
                      aggfunc='sum',fill_value=0)
        .reindex(index=ids,columns=CENSUS_AGE_NAMES,fill_value=0).astype(float))
    sp=persons.dropna(subset=['zone_id']).copy()
    sp['age_group']=_census_age_for_people(sp.age)
    sr=(sp.groupby(['zone_id','age_group'],observed=True).size().unstack(fill_value=0)
          .reindex(index=ids,columns=CENSUS_AGE_NAMES,fill_value=0).astype(float))
    cr_total=cr.sum(axis=0);sr_total=sr.sum(axis=0)
    # Unnormalized number of residents by census-compatible age group, overall
    # and in district panels. Preserve existing composition/share figures.
    count_table=_absolute_age_count_table(zones,cr,sr,sampling_rate,age_names=CENSUS_AGE_NAMES)
    count_table.to_csv(out/'00_population_age_district_counts.csv',index=False)
    _plot_age_count_totals(count_table,
        'Resident counts by census age group — census versus synthetic',
        '00_population_age_counts',out,unit='Residents (people)',
        note='Census uses five-year classes from 0–4 through 60–64 and a single 65+ class. Synthetic ages are aggregated identically; counts are expanded by inverse sampling rate.',age_names=CENSUS_AGE_NAMES)
    count_pages=_plot_district_age_count_pages(zones,count_table,
        'Resident counts by district and census age group',
        out/'00_population_age_district_count_panels',unit='Residents (people)',age_names=CENSUS_AGE_NAMES)
    report['population_age_count_pages']=count_pages
    report['population_age_count_denominator']='absolute number of residents by census-compatible age group (0–4, 5–9, ..., 60–64, 65+) in each district; synthetic expanded by inverse sampling rate'
    _bar_categories('Population age distribution — census versus synthetic',
        'Census age groups: five years from 0–64 and one 65+ class; denominator is all located AMA residents.',
        CENSUS_AGE_NAMES,cr_total,sr_total,out/'00_population_age_distribution',
        'Census',TEAL,share_label='Share of all AMA residents (%)')
    # Retain the earlier census-specific filename as an exact alias, never as
    # an HTS-based age comparison from an older validation run.
    for suffix in ('.csv','.png','.pdf'):
        shutil.copy2(out/f'00_population_age_distribution{suffix}',
                     out/f'00_census_population_age_distribution{suffix}')
    cp=cr.div(cr.sum(axis=1).replace(0,np.nan),axis=0)
    sy=sr.div(sr.sum(axis=1).replace(0,np.nan),axis=0)
    data=pd.DataFrame([{'zone_id':zone,'age_group':age,
        'census_residents':cr.at[zone,age], 'synthetic_residents_sample':sr.at[zone,age],
        'census_share_within_district':cp.at[zone,age],
        'synthetic_share_within_district':sy.at[zone,age],
        'synthetic_minus_census_pp':100*(sy.at[zone,age]-cp.at[zone,age])}
        for zone in ids for age in CENSUS_AGE_NAMES])
    data.to_csv(out/'00_census_population_age_by_district.csv',index=False)
    _plot_age_composition_heatmap(zones,cp,sy,'Age distribution within each district',
       '00_census_population_age_by_district_heatmap',out,reference='Census',
       label='Share of district residents (%)',age_names=CENSUS_AGE_NAMES)
    # District age-distribution mismatch: total variation between the two
    # full age distributions; 0 = identical, 100% = no overlapping age mass.
    tv=.5*(sy-cp).abs().sum(axis=1,min_count=1)
    tv[(cp.isna().all(axis=1)) | (sy.isna().all(axis=1))]=np.nan
    _plot_district_age_mismatch(zones,tv,out)
    if detailed_age_maps:
        cap=max(float(np.nanmax(abs((sy-cp).to_numpy(float)))) if np.isfinite((sy-cp).to_numpy(float)).any() else 0.,.02)*100
        folder=out/'00_census_population_age_districts';folder.mkdir(parents=True,exist_ok=True)
        for age in CENSUS_AGE_NAMES:
            slug=re.sub(r'[^a-zA-Z0-9]+','_',str(age)).strip('_')
            _paired_geo(zones,cp[age],sy[age],f'Population aged {age} by home district',
                f'age_{slug}',folder,reference='Census',color=TEAL,percent=True,scale=cap,
                share_label='Share of residents in each district (%)',
                note='Within-district census age-group composition; not share of the entire AMA population. All age maps use the same difference scale.')
    report['census_population_age_status']='plotted'
    report['census_population_age_groups']=CENSUS_AGE_NAMES
    report['census_population_age_denominator']='all residents within each district'
    return cr



def _plot_district_age_mismatch(zones,tv,out):
    """Summary of census/synthetic within-district age-composition differences."""
    from matplotlib.colors import Normalize
    plt=_plt()
    z=zones[['zone_id','label','geometry']].copy()
    z['age_distribution_difference_pct']=z.zone_id.map(tv)*100
    z[['zone_id','label','age_distribution_difference_pct']].to_csv(
        out/'00_census_population_age_district_discrepancy.csv',index=False)
    ordered=z.sort_values('age_distribution_difference_pct',na_position='first')
    fig,(ax,axmap)=plt.subplots(1,2,figsize=(16,max(7,2.5+len(z)*.35)),
        gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    ax.barh(range(len(ordered)),ordered.age_distribution_difference_pct,color=ORANGE,height=.68)
    ax.set_yticks(range(len(ordered)),
        [textwrap.shorten(str(v),width=30,placeholder='…') for v in ordered.label],fontsize=8)
    ax.set_xlim(left=0)
    ax.set_xlabel('Within-district age distribution difference (%)')
    ax.xaxis.grid(True,alpha=.16);ax.set_axisbelow(True)
    ax.set_title('A   Age-structure difference by district',loc='left',weight='bold')
    highest=max(float(z.age_distribution_difference_pct.max()) if z.age_distribution_difference_pct.notna().any() else 1.,1.)
    z.plot(column='age_distribution_difference_pct',ax=axmap,cmap='YlOrRd',
           vmin=0,vmax=highest,linewidth=.6,edgecolor='white',
           missing_kwds={'color':'#dde2e6','edgecolor':'white'},legend=True,
           legend_kwds={'label':'Within-district age distribution difference (%)','shrink':.65})
    axmap.set_axis_off();axmap.set_title('B   Spatial distribution',loc='left',weight='bold')
    fig.suptitle('Census vs synthetic: census age-group composition difference',x=.04,y=1.025,
                 fontsize=15,weight='bold',ha='left')
    fig.text(.04,-.012,'Difference = half the sum of absolute differences across census-compatible age shares. '
             '0% means identical district age compositions; the value is not a difference in population counts.',
             fontsize=8.5,color=GREY)
    _save(fig,out/'00_census_population_age_district_discrepancy')

def _plot_age_composition_heatmap(zones,ref,syn,title,filename,out,reference='Census',
                                  label='Share of district residents (%)',age_names=None):
    from matplotlib.colors import TwoSlopeNorm
    plt=_plt();ids=zones.zone_id.tolist()
    age_names=list(AGE_NAMES if age_names is None else age_names)
    r=ref.reindex(index=ids,columns=age_names).to_numpy(float)*100
    s=syn.reindex(index=ids,columns=age_names).to_numpy(float)*100
    delta=s-r
    cap=max(float(np.nanmax(abs(delta))) if np.isfinite(delta).any() else 0.,2.)
    fig,axes=plt.subplots(1,3,figsize=(21,max(7.5,len(ids)*.34+2)),layout='constrained')
    for ax,v,name,cmap in zip(axes,[r,s,delta],[reference,'Synthetic','Synthetic − reference'],
                              ['Greens','Oranges','RdBu_r']):
        cm=plt.get_cmap(cmap).copy();cm.set_bad('#e5e9ed')
        options={'vmin':0,'vmax':max(12.,float(np.nanmax([r,s]))*1.08)} if name!='Synthetic − reference' else {
            'norm':TwoSlopeNorm(vmin=-cap,vcenter=0,vmax=cap)}
        im=ax.imshow(v,aspect='auto',cmap=cm,**options)
        fig.colorbar(im,ax=ax,shrink=.7,pad=.012,
          label='Synthetic − reference (pp)' if name=='Synthetic − reference' else label)
        ax.set_title(name,loc='left',weight='bold')
        ax.set_xticks(range(len(age_names)),age_names,rotation=55,ha='right',fontsize=7)
        ax.set_yticks(range(len(ids)),[textwrap.shorten(str(v),width=23,placeholder='…')
                      for v in zones.label],fontsize=7.5)
        ax.tick_params(length=0)
    fig.suptitle(title,x=.04,y=1.035,ha='left',fontsize=15,weight='bold')
    _save(fig,out/filename)


def _census_employment(census,people,zones,translate,out,report,census_population_by_age=None,
                       census_population_totals=None,detailed_age_maps=False,sampling_rate=1.):
    _require(census,['commune_id','age_class','weight'],'census employment')
    c=census.copy();c['zone_id']=translate(c.commune_id)
    c['weight']=pd.to_numeric(c.weight,errors='coerce')
    c.to_csv(out/'data'/'census_employment_mapped.csv',index=False)
    c=c.dropna(subset=['zone_id','weight'])
    if c.weight.sum()<=0:raise ValueError('Census employment has no valid positive weight on study zones')
    p=people.copy();p['employed_flag']=_binary(p.employed)
    s=p[p.employed_flag.eq(1)]
    cr=c.groupby('zone_id').weight.sum();sy=s.groupby('zone_id').size().astype(float)
    _paired_geo(zones,_pct(cr),_pct(sy),'Employed residents by home district',
                '05_employment_district_shares',out,'Census',TEAL,True,
                note='Denominator: all employed AMA residents, NOT the population of each district. Resident origins, not workplaces.',
                share_label='Share of all employed AMA residents (%)')
    counts=pd.DataFrame({'zone_id':zones.zone_id,'census_employed':zones.zone_id.map(cr),
                          'synthetic_employed_sample':zones.zone_id.map(sy).fillna(0)})
    counts.to_csv(out/'data'/'employment_by_district.csv',index=False)
    if census_population_totals is not None:
        # Works even when the census uses age classes too broad to split
        # into the requested five-year age groups.
        census_rate=cr / census_population_totals.reindex(cr.index).replace(0,np.nan)
        synthetic_district_population=p.dropna(subset=['zone_id']).groupby('zone_id').size()
        synthetic_rate=sy / synthetic_district_population.reindex(sy.index).replace(0,np.nan)
        _paired_geo(zones,census_rate,synthetic_rate,
            'Employment rate among district residents',
            '05_employment_rate_by_district',out,'Census',TEAL,True,
            share_label='Employed among all district residents (%)',
            note='Denominator: all residents of each home district (census population totals vs synthetic population totals).')
    
    # Include the three under-15 groups to show synthetic vs reference zeros.
    # The employment census has no recorded observations in these groups.
    census_groups=_map_age_classes(c.age_class,'census_employment',report)
    if census_groups is None:
        report['census_employment_age_comparison']='skipped: unsupported census age_class codes'
        return
    c['five_age_group']=c.age_class.astype(str).map(census_groups)
    # Under-15 census counts are absent, not measured zero rates.
    # Complete count tables still show zeros for the missing reference cells.
    c=c[c.five_age_group.isin(EMPLOYMENT_AGE_NAMES)].copy()
    # Workers with a missing age_class are excluded from the age comparison
    # (but retained in total district employment comparisons).
    cref=c.pivot_table(index='zone_id',columns='five_age_group',values='weight',aggfunc='sum',fill_value=0)
    employed=p[p.employed_flag.eq(1)].copy()
    employed['five_age_group']=_census_age_for_people(employed.age)
    employed=employed[employed.five_age_group.isin(EMPLOYMENT_AGE_NAMES)]
    sref=employed.groupby(['zone_id','five_age_group'],observed=True).size().unstack(fill_value=0)
    ids=zones.zone_id.tolist()
    cref=cref.reindex(index=ids,columns=EMPLOYMENT_AGE_NAMES,fill_value=0).astype(float)
    sref=sref.reindex(index=ids,columns=EMPLOYMENT_AGE_NAMES,fill_value=0).astype(float)
    # Count of employed RESIDENTS by district and census-compatible age group.
    # This is not a workplace-by-district comparison and not an employment rate.
    employed_count_table=_absolute_age_count_table(zones,cref,sref,sampling_rate,age_names=EMPLOYMENT_AGE_NAMES)
    employed_count_table.to_csv(out/'06_employment_age_district_counts.csv',index=False)
    employed_pages=_plot_district_age_count_pages(zones,employed_count_table,
        'Employed resident counts by district and census age group',
        out/'06_employment_age_district_count_panels',unit='Employed residents (people)',age_names=EMPLOYMENT_AGE_NAMES,
        note='Under-15 census employment is unobserved; zero reference bars are placeholders.')
    report['employment_age_count_pages']=employed_pages
    report['employment_age_count_denominator']='absolute employed residents by age group in each HOME district (including 0–4, 5–9, 10–14 shown as census reference zeros because the source omits them); synthetic expanded by inverse sampling rate'
    cr=cref.sum(axis=0)
    sr=sref.sum(axis=0)
    if census_population_by_age is not None:
        # True population denominators by age and district: census residents
        # compared with ALL synthetic residents, not just employed people.
        ref_population=census_population_by_age.reindex(index=ids,columns=EMPLOYMENT_AGE_NAMES,fill_value=0)
        working_pop=people.dropna(subset=['zone_id']).copy()
        working_pop['five_age_group']=_census_age_for_people(working_pop.age)
        working_pop=working_pop[working_pop.five_age_group.isin(EMPLOYMENT_AGE_NAMES)]
        syn_population=(working_pop.groupby(['zone_id','five_age_group'],observed=True)
                         .size().unstack(fill_value=0)
                         .reindex(index=ids,columns=EMPLOYMENT_AGE_NAMES,fill_value=0).astype(float))
        # Age-specific census/synthetic employment rates for the whole AMA.
        cr_age_population=ref_population.sum(axis=0)
        sr_age_population=syn_population.sum(axis=0)
        ref_rate=cr/cr_age_population.replace(0,np.nan)
        # These census age classes have no employment records; do not present
        # their zero count placeholders as measured employment rates.
        ref_rate.loc[CENSUS_UNDER_15]=np.nan
        syn_rate=sr/sr_age_population.replace(0,np.nan)
        age_rate_table=pd.DataFrame({'age_group':EMPLOYMENT_AGE_NAMES,
                      'census_employed':cr.to_numpy(float),
                      'census_residents':cr_age_population.to_numpy(float),
                      'census_employment_rate':ref_rate.to_numpy(float),
                      'synthetic_employed':sr.to_numpy(float),
                      'synthetic_residents':sr_age_population.to_numpy(float),
                      'synthetic_employment_rate':syn_rate.to_numpy(float)})
        _age_bars(pd.DataFrame({'group':EMPLOYMENT_AGE_NAMES,'rate':ref_rate.to_numpy(float),
                                'effective_n':cr_age_population.to_numpy(float)}),
                  pd.DataFrame({'group':EMPLOYMENT_AGE_NAMES,'rate':syn_rate.to_numpy(float),
                                'n':sr_age_population.to_numpy(float)}),
                  'Employment rate by age', '06_employment_rate_by_age', out,
                  reference='Census',color=TEAL,
                  share_label='Employed residents within each age group (%)',
                  note='Denominator: all residents in each age group. Census employment for ages 0–14 is unavailable (blank); synthetic under-15 employment is displayed for diagnostic purposes.',
                  age_names=EMPLOYMENT_AGE_NAMES)
        # _age_bars writes its figure-source CSV with the same stem; preserve
        # the complete rate/count/denominator table for reproducible validation.
        age_rate_table.to_csv(out/'06_employment_rate_by_age.csv',index=False)
        # District × age: calculate within-cell rates, suppress synthetic small n.
        ref_rates=cref/ref_population.replace(0,np.nan)
        ref_rates.loc[:,CENSUS_UNDER_15]=np.nan
        syn_rates=sref/syn_population.replace(0,np.nan)
        syn_rates=syn_rates.where(syn_population>=8)
        _plot_age_composition_heatmap(zones,ref_rates,syn_rates,
            'Employment rate within each district and census age group',
            '06_employment_rate_district_age_heatmap',out,reference='Census',
            label='Employed among all district × age residents (%)',age_names=EMPLOYMENT_AGE_NAMES)
        pd.DataFrame([{'zone_id':zone,'age_group':age,
              'census_employment_rate':ref_rates.at[zone,age],
              'synthetic_employment_rate':syn_rates.at[zone,age],
              'census_residents':ref_population.at[zone,age],
              'synthetic_residents':syn_population.at[zone,age]}
              for zone in ids for age in EMPLOYMENT_AGE_NAMES]).to_csv(
                  out/'06_employment_rate_district_age.csv',index=False)
        report['employment_age_rate_status']='plotted_using_census_population_denominators'
    cr=cref.sum(axis=0)
    sr=sref.sum(axis=0)
    comp=pd.DataFrame({'age_group':EMPLOYMENT_AGE_NAMES,'census_employed':cr.to_numpy(float),
        'synthetic_employed':sr.to_numpy(float)})
    comp['census_share']=_pct(comp.census_employed)
    comp['synthetic_share']=_pct(comp.synthetic_employed)
    comp.to_csv(out/'06_employment_age_composition.csv',index=False)
    _bar_categories('Age composition of employed people',
        'Census age groups from 0–4 through 65+; under-15 reference counts are absent and shown as zero for diagnostics. Denominator is all employed AMA residents.',
        EMPLOYMENT_AGE_NAMES,cr,sr,out/'06_employment_age_composition','Census',TEAL,
        share_label='Share of all employed AMA residents (%)')
    crp=cref.div(cref.sum(axis=1).replace(0,np.nan),axis=0)
    srp=sref.div(sref.sum(axis=1).replace(0,np.nan),axis=0)
    pd.DataFrame([{'zone_id':zone,'age_group':a,
                   'census_share_among_employed':crp.at[zone,a],
                   'synthetic_share_among_employed':srp.at[zone,a]}
                   for zone in ids for a in EMPLOYMENT_AGE_NAMES]).to_csv(out/'06_employment_district_age_shares.csv',index=False)
    plt=_plt();from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(1,3,figsize=(18,max(6,len(ids)*.28+2)),layout='constrained')
    d=(srp-crp).to_numpy(float)*100
    cap=max(np.nanmax(abs(d)) if np.isfinite(d).any() else 0,5)
    for ax,values,name,cmap in zip(axes,[crp.to_numpy(float)*100,srp.to_numpy(float)*100,d],
                                 ['Census','Synthetic','Synthetic − census'],['Greens','Oranges','RdBu_r']):
        cm=plt.get_cmap(cmap).copy();cm.set_bad('#e5e9ed')
        opts={'vmin':0,'vmax':100} if name!='Synthetic − census' else {'norm':TwoSlopeNorm(vmin=-cap,vcenter=0,vmax=cap)}
        im=ax.imshow(values,aspect='auto',cmap=cm,**opts)
        fig.colorbar(im,ax=ax,shrink=.65,pad=.01,label='Share of employed residents in district (%)' if name!='Synthetic − census' else 'Synthetic − census (pp)')
        ax.set_title(name,loc='left',weight='bold')
        ax.set_xticks(range(len(EMPLOYMENT_AGE_NAMES)),EMPLOYMENT_AGE_NAMES,rotation=50,ha='right',fontsize=8)
        ax.set_yticks(range(len(ids)),[textwrap.shorten(str(v),width=24,placeholder='…') for v in zones.label],fontsize=7.5)
        ax.tick_params(length=0)
    fig.suptitle('Age distribution among employed residents, by district',x=.04,y=1+.58/max(6,len(ids)*.28+2),ha='left',weight='bold',fontsize=14)
    _save(fig,out/'06_employment_district_age_heatmap')
    dmax=max(float(np.nanmax(abs(d))) if np.isfinite(d).any() else 0.,2.)
    for label in EMPLOYMENT_AGE_NAMES if detailed_age_maps else []:
        slug=re.sub(r'[^a-zA-Z0-9]+','_',str(label)).strip('_')
        _paired_geo(zones,crp[label],srp[label],f'Share of employed people aged {label}',
           f'06_employed_age_{slug}',out/'employment_age_districts','Census',TEAL,True,
           note='Denominator: employed residents of this district, NOT all residents of this age group.',scale=dmax,
           share_label='Share of district employed residents (%)')
    report['census_employment_age_comparison']='census-compatible age classes 0–4, 5–9, ..., 60–64, 65+; 0–14 census counts are structurally absent (displayed as reference zero in count charts and as NA in rate charts)'
    report['census_employment_age_groups']=EMPLOYMENT_AGE_NAMES


def _income(census,households,zones,translate,out,report,scale):
    _require(census,['commune_id','income_class','weight'],'census income')
    c=census.copy();c['zone_id']=translate(c.commune_id)
    c['weight']=pd.to_numeric(c.weight,errors='coerce')
    c.to_csv(out/'data'/'census_income_mapped.csv',index=False)
    c=c.dropna(subset=['zone_id','weight'])
    if c.weight.sum()<=0:
        report['income_status']='no_census_income_weight'
        report['warnings'].append('Census income weight is empty on the study area; comparison skipped.')
        return
    # Synthetic households.csv stores raw household income; the census input
    # already has income_class, so never cut or reinterpret census records.
    values=pd.to_numeric(households['income'],errors='coerce')*scale
    if values.dropna().empty or values.dropna().eq(0).all():
        report['income_status']='unavailable_all_zero_or_null'
        report['warnings'].append('All synthetic household incomes are zero/missing; no misleading income-validation figure generated.')
        plt=_plt();fig,ax=plt.subplots(figsize=(10,3.2),layout='constrained');ax.axis('off')
        ax.text(.04,.67,'Household income cannot be validated',weight='bold',fontsize=17,transform=ax.transAxes)
        ax.text(.04,.35,'Synthetic exported income contains only zero or missing values.\nNo income distribution is claimed.',transform=ax.transAxes,color=GREY)
        _save(fig,out/'09_income_data_quality');return
    if values.dropna().lt(0).any():
        report['income_status']='invalid_negative_income'
        report['warnings'].append('Synthetic households contain negative income values; income plots skipped.')
        return
    h=households.copy()
    h['income_class']=pd.cut(
        values,
        bins=INCOME_BINS,
        labels=INCOME_NAMES,
        right=False,
    )
    h[['household_id','zone_id','income','income_class']].to_csv(
        out/'data'/'synthetic_household_income_classes.csv',index=False)
    cr=c.groupby('income_class').weight.sum().reindex(INCOME_NAMES,fill_value=0)
    sr=h.groupby('income_class',observed=True).size().reindex(INCOME_NAMES,fill_value=0)
    _bar_categories('Household income distribution','Denominator: all households in the modeled metropolitan area; census income classes versus synthetic.',
        INCOME_NAMES,cr,sr,out/'09_income_distribution','Census',TEAL,
        share_label='Share of all AMA households (%)')
    c['high']=c.income_class.isin(INCOME_NAMES[2:])
    r=c.groupby(['zone_id','high']).weight.sum().unstack(fill_value=0)
    r_high=r.get(True,pd.Series(0.,index=r.index))/r.sum(axis=1).replace(0,np.nan)
    h=h.dropna(subset=['zone_id','income_class']);h['high']=h.income_class.isin(INCOME_NAMES[2:])
    s_high=h.groupby('zone_id').high.mean()
    _paired_geo(zones,r_high,s_high,'Households in income classes ≥4m',
        '10_high_income_households_by_district',out,'Census',TEAL,True,
        note='Denominator: households within each district; same income unit assumed in synthetic after scaling.',
        share_label='District households with income ≥4m (%)')
    rc=c.pivot_table(index='zone_id',columns='income_class',values='weight',aggfunc='sum',fill_value=0)
    sc=h.groupby(['zone_id','income_class'],observed=True).size().unstack(fill_value=0)
    rc=rc.reindex(columns=INCOME_NAMES,fill_value=0);sc=sc.reindex(columns=INCOME_NAMES,fill_value=0)
    rc=rc.div(rc.sum(axis=1).replace(0,np.nan),axis=0)
    sc=sc.div(sc.sum(axis=1).replace(0,np.nan),axis=0)
    pd.DataFrame([{'zone_id':z,'income_class':q,'census_share':rc.at[z,q] if z in rc.index else np.nan,
            'synthetic_share':sc.at[z,q] if z in sc.index else np.nan}
          for z in sorted(set(rc.index)|set(sc.index)) for q in INCOME_NAMES]).to_csv(out/'10_income_class_by_district.csv',index=False)
    report['income_status']='plotted'
    report['income_class_bins']=INCOME_BINS[1:-1]
    report['income_class_right_closed']=False
    report['income_bins_source']='asuncion.data.census.income constants'
    report['synthetic_income_source']='households.csv income, classified with pd.cut(right=False)'


def _methods(out):
    content=f"""# Asuncion synthetic-population validation — methodology

## Sources
- HTS person and household attributes: `asuncion.data.hts.entd.filtered`.
- HTS activity and trip-purpose evidence: `asuncion.data.hts.entd.reweighted`.
- Census employed residents: `asuncion.data.census.employment`.
- Census household income: `asuncion.data.census.income`.
- Census household counts: `asuncion.data.census.households`.
- Census total residents by published age group and home district:
  `asuncion.data.census.population` with `commune_id`, `age_class`, `weight`
  (plus other supplied fields). Census is the reference for **every population
  age distribution**; the HTS is not used as an age-structure reference.
- Census total population by home district: the same census population stage,
  summing `weight` within each mapped home district, compared with all synthetic
  persons geolocated to a home district and expanded by `1/sampling_rate`.
  District totals do not require mapped census age classes.
- All synthetic residents and household output from `synthesis.output`.

## Age grouping
HTS-only attribute plots use five-year bands:
{', '.join(AGE_NAMES)}.
Census-versus-synthetic age figures use the *actual census resolution*:
{', '.join(CENSUS_AGE_NAMES)}.
Numeric census age classes 0, 5, and 10 represent 0–4, 5–9, and 10–14
separately. Census 65 is one open-ended 65+ class. Employment **count**
plots retain 0–4, 5–9, and 10–14 as zero census-reference rows and display
any synthetic employed children for diagnostic checks. Census employment
**rates** in those groups are missing (NA), not measured zero rates.

## Denominator conventions
- **All AMA resident share:** observations in each category divided by all
  residents in the modeled area (for household shares, all households instead).
- **Total residents by district:** absolute census resident counts versus
  expanded synthetic resident counts. This is a count, not a percentage; the
  map shows synthetic minus census residents (people).
- **District composition:** within each district, people in a given category
  divided by all district residents (e.g., census population aged 20–24).
- **District attribute rate:** people with the attribute divided by all district
  residents; district × age panels instead use residents in that specific
  district and census-compatible age group in census comparisons.
- **Employment destination/origin distribution:** employed residents living in
  each district divided by all employed residents in the modeled metropolitan
  area. This is *not* the employment rate within that district.
- **Employment rate within district:** employed residents divided by all
  census or synthetic residents of that home district.
- **Age-specific employment rate:** employed residents in a census age band
  divided by ALL residents of that age band (also within district × age cells).
- **Employed age composition:** employed residents in an age bin divided by all
  employed residents (or all employed residents of the district for district
  figures). This is not an age-specific employment rate.
- **Education / secondary activity destination share:** outgoing activity-trips
  from each district divided by all such observed / synthetic exits in the
  modeled metropolitan area, not by the district population.
- **Age-group and district × age-group COUNT panels:** census people versus
  synthetic sampled people multiplied by `1/sampling_rate`. No normalization:
  units are residents (or employed residents), not percentages. Each page of
  four district panels uses the same x-axis range; the numeric CSV allows
  exact cross-page comparisons.
- **Employment × district × age counts:** employed RESIDENTS in their HOME
  district, rather than jobs physically located in the district. This is a
  different measure from employment *rates* (employed / all residents).
- **Income > or = 4m / large household share:** qualifying households divided
  by all households within that district.
- Differential maps always show synthetic minus the reference in percentage
  points, or in original units for household sizes / counts.

## Validity and restrictions
1. HTS households have no district column. Home district is inferred from
   HOME trip endpoints; households without identifiable homes remain in
   overall charts but not HTS district charts. Coverage is reported.
2. HTS education participation is inferred from preceding OR following
   `education` purpose of valid trips, never from `studies`. It measures
   activity-day participation, not enrollment.
3. HTS education/secondary destinations are origins of outgoing trips after
   those activities, based on preceding purpose, NOT physical facilities.
4. Synthetic households are expanded by `1/sampling_rate` for count figures;
   HTS households are weighted with `household_weight` for size figures.
5. Census income classes are already provided as `income_class`. Raw synthetic
   `households.csv` income is classified using `pd.cut(..., right=False)` and
   `ASUNCION_INCOME_CLASS_BINS` / `ASUNCION_INCOME_CLASS_LABELS` imported
   from `asuncion.data.census.income`. Optional unit scaling is explicit;
   all-zero synthetic incomes remain unvalidated.
6. Districts use the model-zone geometry; commune records are translated with
   the existing commune-to-macrozone mapping. These plots cover the modeled AMA.
7. Age-by-district attribute rate panels suppress cells with HTS effective
   sample < 8 and synthetic n < 8 (not a statistical confidence test).
8. Census population is required. Numeric age_class 0/5/10 are **separate**
   five-year groups, 15–60 retain their five-year groups, and 65 maps to 65+.
   The employment census lacks 0–14 records: count figures show reference
   zeros as a diagnostic; rate figures show missing census values instead.
9. Individual age-specific district bar/map pages are optional using
   `population_validation_detailed_age_maps: true` (they are numerous).
10. The overall population age chart and district age charts use only the
    census as their reference. Education and driving-licence comparisons retain
    HTS references because they concern other attributes, not age composition.
11. Some reference margins may have been used to generate the population;
    agreement can constitute internal consistency instead of held-out validation.

All figures have vector PDF and 260-dpi PNG counterparts and CSV data sources.
See quality_report.json for coverage, source status, and limitations.
"""
    (out/'methods.md').write_text(content,encoding='utf-8')


# ---------------------------------------------------------------------------
# Main population validation
# ---------------------------------------------------------------------------

def create_validation(filtered,reweighted,employment,income,od_zones,codes,
                      synthetic_root,prefix,analysis_path,sampling_rate=.01,
                      income_scale=1.,census_households=None,census_population_age=None,
                      detailed_age_maps=False):
    if not (0<float(sampling_rate)<=1):raise ValueError('sampling_rate must be in (0,1]')
    if not (float(income_scale)>0):raise ValueError('income_scale must be positive')
    out=Path(analysis_path)/'population_validation'
    (out/'data').mkdir(parents=True,exist_ok=True)
    # Prevent an old HTS-based age figure from surviving a new run when census
    # age_class values cannot be mapped safely into census-compatible groups.
    for stem in ('00_population_age_distribution','00_census_population_age_distribution'):
        for ext in ('.csv','.pdf','.png'):
            (out/f'{stem}{ext}').unlink(missing_ok=True)
    quality={'warnings':[],'coverage':{},'reference_definition':{},
             'sampling_rate':float(sampling_rate),'income_scale':float(income_scale)}
    hf,pf,tf=filtered;hr,pr,tr=reweighted
    _require(hf,['household_id','household_weight','household_size'],'HTS filtered households')
    _require(pf,['person_id','household_id','age','person_weight','has_license'],'HTS filtered persons')
    if census_households is None:
        raise ValueError('census_households must be provided from asuncion.data.census.households')
    _require(census_households,['commune_id','weight'],'census households')
    if census_population_age is None:
        raise ValueError('census_population_age is required from asuncion.data.census.population')
    _require(tr,['person_id','preceding_purpose','following_purpose',
                 'origin_district_id','trip_weight'],'HTS reweighted trips')
    zones,translate,crosswalk=_zones(od_zones,codes)
    quality['coverage']['n_geographic_zones']=len(zones)
    quality['coverage']['commune_mapping_entries']=len(crosswalk)
    quality['coverage']['multiple_communes_per_zone']=int(pd.Series(crosswalk).value_counts().gt(1).sum()) if crosswalk else 0
    pe,hh,ts,geo,homegeo=_synthetic(synthetic_root,prefix)
    _require(pe,['person_id','household_id','age','employed','has_driving_license'],'synthetic persons')
    _require(hh,['household_id','income'],'synthetic households')
    _require(ts,['person_id','preceding_activity_index','preceding_purpose'],'synthetic trips')
    _require(geo,['person_id','activity_index','purpose','geometry'],'synthetic geocoded activities')
    for frame,key in ((pe,'person_id'),(pe,'household_id'),(hh,'household_id'),
                      (ts,'person_id'),(geo,'person_id')):
        frame[key]=_id(frame[key])
    geo['activity_index']=pd.to_numeric(geo.activity_index,errors='raise')
    ts['preceding_activity_index']=pd.to_numeric(ts.preceding_activity_index,errors='raise')
    if homegeo is not None:
        _require(homegeo,['household_id','geometry'],'synthetic home locations')
        homegeo['household_id']=_id(homegeo.household_id)
        located_home=_locate(homegeo,zones,['household_id'])
    else:
        if 'household_id' not in geo: raise ValueError('Need homes.gpkg or household_id in geocoded activities')
        geo['household_id']=_id(geo.household_id)
        located_home=_locate(geo[geo.purpose.astype('string').str.lower().eq('home')].drop_duplicates('household_id'),
                            zones,['household_id'])
        quality['warnings'].append('Missing synthetic homes.gpkg: home district inferred from traveling persons only.')
    hh=hh.merge(located_home,on='household_id',how='left',validate='one_to_one')
    pe=pe.merge(hh[['household_id','zone_id']],on='household_id',how='left',validate='many_to_one')
    located_activity=_locate(geo,zones,['person_id','activity_index'])
    located_activity=geo[['person_id','activity_index','purpose']].merge(
        located_activity,on=['person_id','activity_index'],how='left',validate='one_to_one')
    located_activity.to_csv(out/'data'/'synthetic_activity_zones.csv',index=False)
    located_home.to_csv(out/'data'/'synthetic_household_zones.csv',index=False)
    quality['coverage'].update({'synthetic_persons':len(pe),
      'synthetic_persons_geolocated':int(pe.zone_id.notna().sum()),
      'synthetic_households':len(hh),
      'synthetic_households_geolocated':int(hh.zone_id.notna().sum()),
      'synthetic_activities':len(located_activity),
      'synthetic_activities_geolocated':int(located_activity.zone_id.notna().sum())})
    # Household district for HTS persons may be missing: no use of departement_id as a proxy.
    hloc,ties=_hts_homes(pf,tf,translate)
    hloc.to_csv(out/'data'/'hts_inferred_household_zones.csv',index=False)
    hf=hf.copy();pf=pf.copy()
    hf.household_id=_id(hf.household_id);pf.household_id=_id(pf.household_id)
    pf.person_id=_id(pf.person_id)
    hf=hf.merge(hloc,on='household_id',how='left',validate='one_to_one')
    pf=pf.merge(hf[['household_id','zone_id']],on='household_id',how='left',validate='many_to_one')
    for frame,cols in [(hf,['household_weight','household_size']),(pf,['person_weight','age']),(pe,['age'])]:
        for col in cols:frame[col]=pd.to_numeric(frame[col],errors='coerce')
    quality['coverage'].update({'hts_households':len(hf),
        'hts_households_with_located_home':int(hf.zone_id.notna().sum()),
        'hts_persons':len(pf),'hts_persons_with_located_home':int(pf.zone_id.notna().sum()),
        'ambiguous_hts_home_households':ties,
        'hts_household_weight_geolocated_share':float(hf.loc[hf.zone_id.notna(),'household_weight'].sum()/hf.household_weight.sum())
           if hf.household_weight.sum()>0 else None})
    if hf.zone_id.isna().any():
        quality['warnings'].append('HTS household district is only inferred for homes with recorded home trip endpoints; district comparisons exclude unlocated homes and can be biased.')

    # 00. Baseline age: exclusively CENSUS vs synthetic, not HTS vs synthetic.
    # The census age distribution and census-compatible district comparisons are made
    # by _census_age_population, conditional on compatible census age classes.
    pe['age_band']=pd.cut(pe.age,AGE_BINS,labels=AGE_NAMES,right=False)
    _require(census_population_age,['commune_id','age_class','weight'],'census population')
    census_pop_frame=census_population_age.copy()
    census_pop_frame['zone_id']=translate(census_pop_frame.commune_id)
    census_pop_frame['weight']=pd.to_numeric(census_pop_frame.weight,errors='coerce')
    if (census_pop_frame.weight.isna().any() or
        not np.isfinite(census_pop_frame.weight).all() or
        (census_pop_frame.weight<0).any()):
        raise ValueError('Census population weights must be finite and nonnegative')
    census_pop_frame.to_csv(out/'data'/'census_population_mapped.csv',index=False)
    census_pop_district=(census_pop_frame.dropna(subset=['zone_id'])
                         .groupby('zone_id').weight.sum().reindex(zones.zone_id,fill_value=0))
    synthetic_pop_district=(pe.dropna(subset=['zone_id']).groupby('zone_id').size()
                             .astype(float).reindex(zones.zone_id,fill_value=0))
    if census_pop_district.sum()<=0:
        raise ValueError('Census population has no positive weight in modeled districts')
    _paired_geo(zones,census_pop_district,
        synthetic_pop_district/float(sampling_rate),
        'Total resident population by home district — census versus synthetic',
        '00_population_district_counts',out,reference='Census',color=TEAL,
        percent=False,measure='Residents (people)',
        note=f'Absolute number of residents, not a percentage. Census residents versus synthetic sample expanded by 1/{sampling_rate:g}. '
             'Only geocoded residents in modeled districts are included; the quality report records coverage.')
    pd.DataFrame({'zone_id':zones.zone_id,'district_name':zones.label,
        'census_residents':zones.zone_id.map(census_pop_district).fillna(0),
        'synthetic_residents_sample':zones.zone_id.map(synthetic_pop_district).fillna(0),
        'synthetic_residents_expanded':zones.zone_id.map(synthetic_pop_district).fillna(0)/float(sampling_rate),
        'synthetic_minus_census':(zones.zone_id.map(synthetic_pop_district).fillna(0)/float(sampling_rate)
                                  -zones.zone_id.map(census_pop_district).fillna(0))
    }).to_csv(out/'data'/'population_counts_by_district.csv',index=False)
    quality['coverage'].update({
        'census_population_total_all_districts':float(census_pop_frame.weight.sum()),
        'census_population_in_modeled_districts':float(census_pop_district.sum()),
        'synthetic_population_expanded_total':float(len(pe)/float(sampling_rate)),
        'synthetic_population_expanded_geolocated':float(synthetic_pop_district.sum()/float(sampling_rate)),
    })
    quality['reference_definition']['population_district_counts']=         'Census total resident weight versus synthetic geocoded residents expanded by inverse sampling rate'
    census_age_matrix=_census_age_population(census_population_age,pe,zones,translate,out,quality,
                                                detailed_age_maps=detailed_age_maps,sampling_rate=sampling_rate)
    if 'sex' in pf and 'sex' in pe:
        sex_ht=pf.assign(sex_key=pf.sex.astype('string').str.lower().str.strip()).groupby('sex_key').person_weight.sum()
        sex_sy=pe.assign(sex_key=pe.sex.astype('string').str.lower().str.strip()).groupby('sex_key').size()
        sexes=sorted(set(sex_ht.index)|set(sex_sy.index))
        _bar_categories('Population sex distribution',
            'HTS person expansion weights versus all synthetic residents.',
            sexes,sex_ht,sex_sy,out/'00_population_sex_distribution',
            share_label='Share of all AMA residents (%)')
    # 01/02: origins of outgoing trips AFTER education and secondary activities.
    starts=ts[['person_id','preceding_activity_index','preceding_purpose']].rename(
        columns={'preceding_activity_index':'activity_index'}).merge(
        located_activity,on=['person_id','activity_index'],how='left',validate='many_to_one')
    starts['purpose_key']=starts.preceding_purpose.astype('string').str.strip().str.lower()
    tr=tr.copy()
    tr['purpose_key']=tr.preceding_purpose.astype('string').str.strip().str.lower()
    tr['zone_id']=translate(tr.origin_district_id)
    tr['trip_weight']=pd.to_numeric(tr.trip_weight,errors='coerce')
    for name,num,kind in [('education','01','education'),('secondary','02','secondary')]:
        def ispurpose(v):return v.eq('education') if kind=='education' else ~v.isin(OTHER_EXCLUDE)
        sel_h=tr.loc[ispurpose(tr.purpose_key)&(tr.trip_weight>0)]
        sel_s=starts.loc[ispurpose(starts.purpose_key)]
        reference=_weighted(sel_h,'zone_id','trip_weight')
        synthetic=sel_s.dropna(subset=['zone_id']).groupby('zone_id').size().astype(float)
        _paired_geo(zones,_pct(reference),_pct(synthetic),
            'Education activity destinations' if name=='education' else 'Secondary activity destinations',
            f'{num}_{name}_destination_shares',out,percent=True,
            note='Trip-origin locations where preceding purpose is '+
              ('education.' if kind=='education' else 'neither home, work, education nor unknown.')+
              ' Denominator: all such activity exits in the modeled AMA; NOT number of facilities.',
            share_label='Share of all AMA '+name+' activity exits (%)')
        quality['coverage'][name+'_hts_weight_in_study_zones']=float(reference.sum())
        quality['coverage'][name+'_synthetic_exits_in_zones']=int(synthetic.sum())
        quality['coverage'][name+'_hts_weight_outside_zones']=float(sel_h.loc[sel_h.zone_id.isna(),'trip_weight'].sum())
    # 03: driving license rates by age/district; HTS filtered includes non-travelers.
    pf['license']=pf.has_license;pe['license']=pe.has_driving_license
    pe['_sample_weight']=1.0
    lht=_age_summary(pf,'license','person_weight')
    lsy=_age_summary(pe,'license','_sample_weight')
    _age_bars(lht[0],lsy[0],'Driving licence ownership by age','03_driving_license_age',out,
              note='HTS person-weighted vs synthetic persons; five-year age groups.',
              share_label='Licensed people within each age group (%)')
    _paired_geo(zones,lht[1].set_index('zone_id').rate,lsy[1].set_index('zone_id').rate,
       'Driving licence ownership by district','03_driving_license_district',out,percent=True,
       share_label='Licensed among district residents (%)')
    _age_heatmaps(zones,lht[2],lsy[2],'Driving licence by district × age',
                  '03_driving_license_district_age_heatmap',out,
                  share_label='Licensed among residents in district × age (%)')
    if detailed_age_maps:
        _age_district_bars_maps(zones,lht[2],lsy[2],'Driving licence rate',
                 out/'03_driving_license_age_districts',
                 share_label='Licensed among residents in district × age (%)')

    # 04: Education participation from trip evidence, NEVER HTS `studies`.
    # Reweighted HTS represents persons with valid trip data. Count each such
    # person at most once, including if they have several education trips.
    _require(pr,['person_id','household_id','person_weight','age'],'HTS reweighted persons')
    rr=pr.copy();rr.person_id=_id(rr.person_id);rr.household_id=_id(rr.household_id)
    tr['person_id']=_id(tr.person_id)
    education_trip=(tr.preceding_purpose.astype('string').str.strip().str.lower().eq('education') |
                    tr.following_purpose.astype('string').str.strip().str.lower().eq('education'))
    attendees=set(tr.loc[education_trip,'person_id'].dropna())
    rr['education_today']=rr.person_id.isin(attendees)
    # Infer homes using the same trip-valid subset: HTS home distribution can
    # differ from `filtered`; this keeps numerator and denominator coherent.
    rloc,rties=_hts_homes(rr,tr,translate)
    rr=rr.merge(rloc,on='household_id',how='left',validate='many_to_one')
    rr['person_weight']=pd.to_numeric(rr.person_weight,errors='coerce')
    rr['age']=pd.to_numeric(rr.age,errors='coerce')
    pe['day_education']=pe.person_id.isin(set(located_activity.loc[
       located_activity.purpose.astype('string').str.strip().str.lower().eq('education'),'person_id']))
    eh=_age_summary(rr,'education_today','person_weight')
    es=_age_summary(pe,'day_education','_sample_weight')
    _age_bars(eh[0],es[0],'Education activity participation by age',
       '04_education_daily_participation_age',out,
       reference='HTS education trips',
       note='People with an education-purpose trip (either endpoint) vs synthetic people with an education activity; five-year groups.',
       share_label='Education participants within each age group (%)')
    _paired_geo(zones,eh[1].set_index('zone_id').rate,es[1].set_index('zone_id').rate,
        'Education activity participation by home district',
        '04_education_participation_district',out,
        reference='HTS education trips',percent=True,
        note='Denominator: all reweighted HTS residents or synthetic residents in each district; unlocated HTS homes excluded.',
        share_label='Education participants among district residents (%)')
    _age_heatmaps(zones,eh[2],es[2],
                  'Education participation by district × age',
                  '04_education_district_age_heatmap',out,
                  reference='HTS education trips',
                  share_label='Education participants in district × age (%)')
    if detailed_age_maps:
        _age_district_bars_maps(zones,eh[2],es[2],
             'Education activity participation',out/'04_education_age_districts',
             reference='HTS education trips',
             share_label='Education participants in district × age (%)')
    quality['coverage']['hts_education_persons']=len(rr)
    quality['coverage']['hts_education_participants']=int(rr.education_today.sum())
    quality['coverage']['hts_education_persons_geolocated']=int(rr.zone_id.notna().sum())
    quality['coverage']['hts_education_ambiguous_home_households']=int(rties)
    quality['reference_definition']['education']='HTS reweighted trip-based participation; preceding OR following purpose education'
    quality['warnings'].append('Trip-based education participation is a day-level indicator, not confirmed student enrollment; HTS and synthetic participation depend on modeled/survey activity-day coverage.')
    # 05/06: employment counts, composition, and age-specific population rates.
    _census_employment(employment,pe,zones,translate,out,quality,census_age_matrix,
                       census_population_totals=census_pop_district,
                       detailed_age_maps=detailed_age_maps,sampling_rate=sampling_rate)
    # 07: household sizes, computed from synthetic persons membership.
    sizes=pe.groupby('household_id').size().rename('computed_size')
    hh=hh.join(sizes,on='household_id')
    hh['size_cat']=pd.cut(hh.computed_size,[0,1,2,3,4,5,np.inf],labels=SIZE_NAMES)
    hf['size_cat']=pd.cut(hf.household_size,[0,1,2,3,4,5,np.inf],labels=SIZE_NAMES)
    hs=hf.groupby('size_cat',observed=False).household_weight.sum().reindex(SIZE_NAMES,fill_value=0)
    ss=hh.groupby('size_cat',observed=False).size().reindex(SIZE_NAMES,fill_value=0)
    _bar_categories('Household size distribution','HTS household weights versus size from synthetic persons; 6+ combined.',
       SIZE_NAMES,hs,ss,out/'07_household_size_distribution',
       share_label='Share of all AMA households (%)')
    ha=hf.dropna(subset=['zone_id','household_size']).copy()
    sa=hh.dropna(subset=['zone_id','computed_size']).copy()
    ha['weighted_size']=ha.household_size*ha.household_weight
    mean_ht=ha.groupby('zone_id').weighted_size.sum()/ha.groupby('zone_id').household_weight.sum()
    mean_sy=sa.groupby('zone_id').computed_size.mean()
    _paired_geo(zones,mean_ht,mean_sy,'Mean household size by district',
                '07_mean_household_size_district',out,percent=False,measure='People per household')
    large_ht=ha.assign(large_weight=ha.household_size.ge(4)*ha.household_weight)
    large_ht=large_ht.groupby('zone_id').large_weight.sum()/ha.groupby('zone_id').household_weight.sum()
    large_sy=sa.assign(large=sa.computed_size.ge(4)).groupby('zone_id').large.mean()
    _paired_geo(zones,large_ht,large_sy,'Households containing four or more people',
                '07_large_households_district',out,percent=True,
                share_label='District households with ≥4 residents (%)')

    # 08: absolute household COUNTS against census (not HTS weights).
    hcensus=census_households.copy()
    hcensus['zone_id']=translate(hcensus.commune_id)
    hcensus['weight']=pd.to_numeric(hcensus.weight,errors='coerce')
    if hcensus.weight.isna().any() or not np.isfinite(hcensus.weight).all() or (hcensus.weight<0).any():
        raise ValueError('Census household weights must be finite and nonnegative')
    hcensus.to_csv(out/'data'/'census_households_mapped.csv',index=False)
    hc=hcensus.dropna(subset=['zone_id']).groupby('zone_id').weight.sum()
    if hc.sum()<=0:
        raise ValueError('Census households have no nonzero weight in modeled zones')
    sc=sa.groupby('zone_id').size().astype(float)
    _paired_geo(zones,_pct(hc),_pct(sc),'Household location shares by district',
                '08_household_district_shares',out,reference='Census',color=TEAL,percent=True,
                note='Denominator: all households in modeled AMA districts. Synthetic counts use geocoded homes.',
                share_label='Share of all AMA households (%)')
    _paired_geo(zones,hc,sc/float(sampling_rate),'Household counts by district',
                '08_household_district_counts',out,reference='Census',color=TEAL,
                percent=False,measure='Number of households',
                note=f'Census household weights versus synthetic sampled households expanded by 1/{sampling_rate:g}.')
    pd.DataFrame({'zone_id':zones.zone_id,'census_households':zones.zone_id.map(hc).fillna(0),
       'synthetic_sample':zones.zone_id.map(sc).fillna(0),
       'synthetic_expanded':zones.zone_id.map(sc).fillna(0)/float(sampling_rate)}
       ).to_csv(out/'data'/'household_counts.csv',index=False)
    census_total=float(hcensus.weight.sum())
    census_mapped=float(hc.sum())
    synthetic_total=len(hh)/float(sampling_rate)
    synthetic_mapped=float(sc.sum())/float(sampling_rate)
    whole=pd.Series({'Census (all districts)':census_total,
                     'Census (modeled districts)':census_mapped,
                     'Synthetic (all, expanded)':synthetic_total,
                     'Synthetic (located, expanded)':synthetic_mapped})
    whole.to_csv(out/'08_total_household_counts.csv',header=['households'])
    plt=_plt();fig,ax=plt.subplots(figsize=(11,4.9),layout='constrained')
    ax.barh(range(4),whole.values,color=[TEAL,TEAL,ORANGE,ORANGE])
    ax.set_yticks(range(4),[textwrap.fill(a,31) for a in whole.index],fontsize=9)
    ax.invert_yaxis();ax.set_xlabel('Households')
    ax.set_title('Total household counts and geographic coverage',loc='left',fontsize=14,weight='bold')
    ax.xaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
    _save(fig,out/'08_total_household_counts')
    # 09/10: income, with strict data-quality and unit checks.
    _income(income,hh,zones,translate,out,quality,float(income_scale))
    zones[['zone_id','label']].to_csv(out/'data'/'zone_names.csv',index=False)
    quality['coverage']['hts_household_weight_total']=float(hf.household_weight.sum())
    quality['coverage']['hts_household_weight_located']=float(ha.household_weight.sum())
    quality['coverage']['census_households_total']=float(census_total)
    quality['coverage']['census_households_in_modeled_zones']=float(census_mapped)
    quality['coverage']['synthetic_household_expanded']=len(hh)/float(sampling_rate)
    quality['reference_definition']={
      'hts_population':'filtered, including respondents without valid trips',
      'hts_trips':'reweighted, preceding activity purpose and trip origin district',
      'employment':'census employed residents by HOME district',
      'education':'HTS reweighted preceding/following education trip purposes versus synthetic education day',
      'household_counts':'Census households by residence commune and weight versus synthetic household expansion',
      'income':'census income_class directly; synthetic raw households.csv income classified using constants imported from asuncion.data.census.income and pd.cut(right=False)',
      'population_age_district':'census total residents from asuncion.data.census.population; published ages 0–4, 5–9, ..., 60–64, 65+',
      'population_district_counts':'census resident counts versus synthetic expanded residents by home district',
      'population_age_absolute_counts':'census vs synthetic expanded residents by home district in published census age groups (0–4, 5–9, ..., 60–64, 65+)',
      'employment_age_absolute_counts':'census vs synthetic expanded employed residents by home district, including under-15 groups as census-reference zero placeholders'}
    _methods(out)
    (out/'quality_report.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(f'Population validation: {out} ({len(list(out.rglob("*.pdf")))} PDF charts)')
    for m in quality['warnings']:warnings.warn('[population validation] '+m,stacklevel=2)
    return str(out)


def execute(context):
    context.stage('synthesis.output')
    return create_validation(
      filtered=context.stage('asuncion.data.hts.entd.filtered'),
      reweighted=context.stage('asuncion.data.hts.entd.reweighted'),
      employment=context.stage('asuncion.data.census.employment'),
      income=context.stage('asuncion.data.census.income'),
      census_households=context.stage('asuncion.data.census.households'),
      census_population_age=context.stage('asuncion.data.census.population'),
      od_zones=context.stage('asuncion.gravity.od_zones'),
      codes=context.stage('asuncion.data.codes'),
      synthetic_root=context.config('output_path'),
      prefix=context.config('output_prefix'),
      analysis_path=context.config('analysis_path'),
      sampling_rate=float(context.config('sampling_rate')),
      income_scale=float(context.config('population_validation_income_scale')),
      detailed_age_maps=bool(context.config('population_validation_detailed_age_maps')),
    )
