"""
SEPARATE CONCEPTS:
 - synthetic person's residential district: exported district_id
 - matched diary-person's *residential* district: synthetic.hts_id -> HTS person
 - assigned trips: entire HTS diary copied once for each matched synthetic person
 - actual synthetic physical trip origin/destination: assigned later and is NOT
   inferred from the diary donor's district or the donor trip's OD.

This stage does not claim to audit the *IPU seed household* donor selection;
there is no original IPU source household ID in the currently exported output.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

CODES = {
    '0000': 'Asunción', '1102': 'Capiatá', '1103': 'Fernando de la Mora',
    '1107': 'Lambaré', '1108': 'Limpio', '1109': 'Luque',
    '1110': 'Mariano Roque Alonso', '1112': 'Ñemby', '1113': 'San Antonio',
    '1114': 'San Lorenzo', '1115': 'Villa Elisa',
}
DEPT = {'00': 'Asunción', '11': 'Central'}
PURPOSE_MAP = {1:'home',2:'work',3:'leisure',4:'other',5:'shop',6:'other',
               7:'education',8:'other',9:'other',10:'work'}
PURPOSES = ['home','work','education','shop','leisure','other']
DEFAULT_AGE_BINS = [4,14,20,29,44,59,74,1000]


def _id_code(v):
    if pd.isna(v) or str(v).strip() in ('', 'nan', '<NA>'):
        return pd.NA
    try:
        return f'{int(float(v)):04d}'
    except (ValueError,TypeError,OverflowError):
        return str(v).strip().zfill(4)


def _bool(v):
    if pd.isna(v):
        return pd.NA
    if isinstance(v, (bool,np.bool_)):
        return bool(v)
    if isinstance(v,(int,float,np.number)):
        return bool(v)
    val=str(v).strip().lower()
    if val in ('1','true','yes','t'):
        return True
    if val in ('0','false','no','f'):
        return False
    return pd.NA


def _pick(df, possibilities, label, required=True):
    for c in possibilities:
        if c in df.columns:
            return df[c]
    if required:
        raise ValueError(f'{label} requires one of {possibilities}; available {list(df.columns)}')
    return pd.Series(pd.NA,index=df.index)


def _normalize_hts(hts_households, hts_persons, hts_trips):
    hh=hts_households.copy();p=hts_persons.copy();t=hts_trips.copy()
    hh['donor_household_id']=pd.to_numeric(_pick(hh,['household_id','Id'],'HTS household id'),errors='coerce')
    hh['donor_district']=_pick(hh,['hts_household_district_id','IdDepartamentoDistrito'],'HTS household district').map(_id_code)
    hh['donor_dept']=hh.donor_district.str[:2]
    hh['hh_weight']=pd.to_numeric(_pick(hh,['household_weight','Fex'],'HTS household weight'),errors='coerce')
    if hh.donor_household_id.isna().any() or hh.donor_household_id.duplicated().any():
        raise ValueError('HTS household identifier must be unique and nonmissing')
    p['hts_id']=pd.to_numeric(_pick(p,['person_id','Id'],'HTS person id'),errors='coerce')
    p['donor_household_id']=pd.to_numeric(_pick(p,['household_id','IdViviendaHogar'],'HTS person household id'),errors='coerce')
    p['age_donor']=pd.to_numeric(_pick(p,['age','Edad'],'HTS age'),errors='coerce')
    p['person_weight']=pd.to_numeric(_pick(p,['person_weight','Fex'],'HTS person weight'),errors='coerce')
    def hts_yes(v):
        if isinstance(v,(bool,np.bool_)):
            return bool(v)
        try:
            return float(v)==1.0
        except (TypeError,ValueError):
            return str(v).strip().lower() in ('true','yes')
    p['employed_donor']=_pick(p,['employed','ActLaboralReciente'],'HTS employment').map(hts_yes)
    license_source=_pick(p,['has_license','LicenciaConducir'],'HTS licence')
    p['licence_donor']=license_source.map(hts_yes)
    p['license_response_valid']=(pd.to_numeric(license_source,errors='coerce').isin([1,2,3]) if 'LicenciaConducir' in p.columns else pd.Series(True,index=p.index))
    sex=_pick(p,['sex','sexo','IdSexo'],'HTS sex')
    p['sex_donor']=sex.map(lambda v: 'female' if str(v).lower() in ('6','female','mujer') else ('male' if str(v).lower() in ('1','male','hombre') else str(v).lower()))
    p['diary_n_declared']=pd.to_numeric(_pick(p,['number_of_trips','Viajes'],'HTS diary declared trips',required=False),errors='coerce')
    p=p.merge(hh[['donor_household_id','donor_district','donor_dept']],on='donor_household_id',how='left',validate='many_to_one')
    if p.hts_id.duplicated().any():
        raise ValueError('HTS person IDs are not unique')
    t['hts_id']=pd.to_numeric(_pick(t,['person_id','IdPersona'],'HTS trip person id'),errors='coerce')
    if 'following_purpose' in t.columns:
        t['purpose']=t.following_purpose.map(lambda v:PURPOSE_MAP.get(int(v),str(v).lower()) if str(v).isdigit() else str(v).lower())
    else:
        t['purpose']=pd.to_numeric(_pick(t,['IdCategoriaActividad'],'HTS purpose'),errors='coerce').map(PURPOSE_MAP)
    t['trip_w']=pd.to_numeric(_pick(t,['trip_weight','Fex'],'HTS trip weight',required=False),errors='coerce')
    t=t.merge(p[['hts_id','donor_district','donor_dept','person_weight']],on='hts_id',how='left',validate='many_to_one')
    t['trip_w']=t.trip_w.fillna(t.person_weight)
    return hh,p,t


def _prep_synthetic(synth_people, synth_households, synth_trips, donors):
    p=synth_people.copy();h=synth_households.copy();t=synth_trips.copy()
    for key in ['person_id','hts_id','household_id','age','employed','sex']:
        if key not in p.columns:
            raise ValueError(f'Synthetic persons lack {key!r}')
    if 'district_id' not in p.columns:
        if 'district_id' not in h.columns:
            raise ValueError('No residential district in synthetic persons or households')
        p=p.merge(h[['household_id','district_id']],on='household_id',how='left',validate='many_to_one')
    p['person_id']=pd.to_numeric(p.person_id,errors='raise')
    p['hts_id']=pd.to_numeric(p.hts_id,errors='coerce')
    p['home_district']=p.district_id.map(_id_code)
    p['home_dept']=p.home_district.str[:2]
    p['age']=pd.to_numeric(p.age,errors='coerce')
    p['employed_synth']=p.employed.map(_bool)
    p['licence_synth']=_pick(p,['has_driving_license','has_license'],'Synthetic licence',required=False).map(_bool)
    p['sex_synth']=p.sex.astype('string').str.lower()
    if p.person_id.duplicated().any():
        raise ValueError('Duplicated synthetic person IDs')
    if p.home_district.isna().any():
        raise ValueError('Missing synthetic residential district')
    donor_cols=['hts_id','donor_household_id','donor_district','donor_dept','age_donor','employed_donor','licence_donor','sex_donor','person_weight','diary_n_declared','license_response_valid']
    p=p.merge(donors[donor_cols],on='hts_id',how='left',validate='many_to_one',indicator='donor_match')
    if (p.donor_match!='both').any():
        raise ValueError(f'{(p.donor_match!="both").sum()} synthetic persons have no matching HTS donor id')
    p['same_district']=p.home_district==p.donor_district
    p['same_dept']=p.home_dept==p.donor_dept
    p['same_sex']=p.sex_synth==p.sex_donor
    p['same_employed']=(p.employed_synth.astype('boolean')==p.employed_donor.astype('boolean')).astype('boolean')
    p['same_licence']=(p.licence_synth.astype('boolean')==p.licence_donor.astype('boolean')).astype('boolean')
    p['age_abs_error']=(p.age-p.age_donor).abs()
    if t.person_id.isna().any() or not t.person_id.isin(p.person_id).all():
        raise ValueError('Trip records refer to unknown synthetic people')
    t['purpose']=t.following_purpose.astype('string').str.lower()
    t=t.merge(p[['person_id','hts_id','home_district','home_dept','donor_district','donor_dept','age','employed_synth']],on='person_id',how='left',validate='many_to_one')
    trip_counts=t.groupby('person_id').size().rename('n_trips_synth')
    p['n_trips_synth']=p.person_id.map(trip_counts).fillna(0).astype(int)
    p['traveller']=p.n_trips_synth.gt(0)
    p['age_4plus']=p.age.ge(4)
    p['school_age']=p.age.between(6,17)
    p['employment_age']=p.age.ge(15)
    p['licence_age']=p.age.ge(18)
    for purpose in ['education','work']:
        c=t[t.purpose==purpose].groupby('person_id').size()
        p[f'{purpose}_trips']=p.person_id.map(c).fillna(0).astype(int)
        p[f'has_{purpose}_trip']=p[f'{purpose}_trips'].gt(0)
    return p,h,t


def _weighted_avg(s,col,w='person_weight'):
    if not len(s):return np.nan
    x=pd.to_numeric(s[col],errors='coerce').to_numpy(float)
    wt=pd.to_numeric(s[w],errors='coerce').to_numpy(float)
    valid=np.isfinite(x)&np.isfinite(wt)&(wt>0)
    return float(np.average(x[valid],weights=wt[valid])) if valid.any() else np.nan


def _eff_n(weights):
    w=np.asarray(weights,float);w=w[np.isfinite(w)&(w>0)]
    return float(w.sum()**2/np.square(w).sum()) if len(w) else np.nan


def _pct(s):
    return float(100*s.mean()) if len(s) else np.nan


def _save(fig, path, matching_kind='trip'):
    """Save a figure with a persistent, plainly colored provenance label.

    `trip` covers HTS diary-person matching, diary-derived trip behaviour, and
    final spatial-trip diagnostics. `ipu` is strictly for IPU seed selection.
    """
    assert matching_kind in ('trip', 'ipu'), matching_kind
    caption = '[IPU matching]' if matching_kind == 'ipu' else '[trip matching]'
    color = "#881F1B" if matching_kind == 'ipu' else '#325EA8'
    fig.text(0.014, 0.995, caption, ha='left', va='top', fontsize=10,
             fontweight='bold', color=color, transform=fig.transFigure)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white', pad_inches=0.20)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _make_table(df, heading, path, small=False, matching_kind='trip'):
    import matplotlib.pyplot as plt
    columns=len(df.columns); rows=len(df)
    fig,ax=plt.subplots(figsize=(max(9.8,1.65*columns),max(2.3,.51*(rows+2))))
    ax.axis('off')
    text=[]
    for row in df.itertuples(index=False,name=None):
        text.append(['' if pd.isna(v) else str(v) for v in row])
    tab=ax.table(cellText=text,colLabels=df.columns.tolist(),loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(8 if small else 9);tab.scale(1,1.35)
    for (i,j),c in tab.get_celld().items():
        c.set_edgecolor('#DDE2E6')
        if i==0:c.set_facecolor('#E5EAF1');c.set_text_props(weight='bold')
        if i>0 and i%2==0:c.set_facecolor('#F7F8FA')
        if j==0 and i>0:c.set_text_props(weight='bold')
    ax.set_title(heading,fontsize=13,fontweight='bold',loc='left',pad=14)
    fig.tight_layout()
    _save(fig, path, matching_kind=matching_kind)


def _matrix_counts(src,dest,order):
    return pd.crosstab(src,dest).reindex(index=order,columns=order,fill_value=0)


def _matrix_pair_plot(mat_d,mat_p,level,output,heading='HTS donor origin vs synthetic residence',titles=None,axis_labels=None,matching_kind='trip'):
    import matplotlib.pyplot as plt
    names=[CODES.get(k,k) if level=='district' else DEPT.get(k,k) for k in mat_d.index]
    figs=[]
    fig,axs=plt.subplots(1,2,figsize=(15.5 if level=='district' else 11,7.7 if level=='district' else 5),layout='constrained')
    if titles is None:
        titles=['Person donor assignment','Trip diary donor assignment']
    for ax,mat,title in zip(axs,[mat_d,mat_p],titles):
        vals=mat.to_numpy(dtype=float)
        denom=vals.sum(axis=1,keepdims=True)
        vals=100*np.divide(vals,denom,out=np.zeros_like(vals),where=denom>0)
        im=ax.imshow(vals,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
        ax.set_title(title)
        ax.set_xticks(range(len(names)),names,rotation=65 if level=='district' else 0,ha='right' if level=='district' else 'center',fontsize=8)
        ax.set_yticks(range(len(names)),names,fontsize=8)
        ax.set_xlabel((axis_labels or ('HTS donor residential '+level,'Synthetic resident '+level))[0])
        ax.set_ylabel((axis_labels or ('HTS donor residential '+level,'Synthetic resident '+level))[1])
        for i in range(len(names)):
            for j in range(len(names)):
                v=vals[i,j]
                if v>=7 or level=='department':
                    ax.text(j,i,f'{v:.0f}%',ha='center',va='center',fontsize=7.7,color='white' if v>50 else '#122136')
    fig.colorbar(im,ax=axs,label='% of recipient row',shrink=.78)
    fig.suptitle(f'{heading} — {level}',fontsize=14,fontweight='bold')
    _save(fig,output,matching_kind=matching_kind)


def _bar_table(rows,level,out):
    import matplotlib.pyplot as plt
    names=rows['area'].tolist()
    fig,ax=plt.subplots(figsize=(12,5.0))
    ind=np.arange(len(rows));width=.36
    ax.bar(ind-width/2,rows.local_people_pct,width,label='Persons: donor lives locally',color='#2E7191')
    ax.bar(ind+width/2,rows.local_trips_pct,width,label='Trips: donor lives locally',color='#D67D43')
    ax.set_xticks(ind,names,rotation=55 if level=='district' else 0,ha='right' if level=='district' else 'center',fontsize=9)
    ax.set_ylabel('Within-'+level+' donor matching (%)');ax.set_ylim(0,105)
    ax.grid(axis='y',alpha=.19);ax.legend(frameon=False);ax.set_title(f'Local HTS donor assignments by synthetic residential {level}',fontweight='bold')
    fig.tight_layout();_save(fig,out)


def _behaviour_plot(frame,level,out):
    import matplotlib.pyplot as plt
    if frame.empty:return
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    x=np.arange(len(frame));labels=frame.area.tolist()
    for ax,a,b,title in [
        (axs[0,0],'hts_trips_age4plus','syn_trips_age4plus','Trips per person aged 4+'),
        (axs[0,1],'hts_immobile_pct','syn_immobile_pct','Share with zero trips, all ages (%)'),
        (axs[1,0],'hts_education_6to17_pct','syn_education_6to17_pct','6–17: people with education trips (%)'),
        (axs[1,1],'hts_work_trip_employed_pct','syn_work_trip_employed_pct','15+: employed people with work trip (%)'),
    ]:
        ax.plot(x,frame[a],marker='o',label='HTS weighted',color='#172E4A',lw=2)
        ax.plot(x,frame[b],marker='o',label='Synthetic',color='#BB4E40',lw=2)
        ax.set_title(title,fontsize=11)
        ax.set_xticks(x,labels,rotation=55 if level=='district' else 0,ha='right' if level=='district' else 'center',fontsize=8)
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Travel behaviour by residential '+level+' — HTS vs synthesis',fontsize=14,fontweight='bold')
    _save(fig,out)


def _purpose_plot(syn_trips, hts_trips, level, out):
    """Render table-only destination-purpose mix comparisons, never stacked bars.

    The grouping location is the *respondent's HOME district/department*, NOT
    the destination district of the trip. 'Destination purpose' is the purpose
    of the following activity (e.g., work, education, shopping, returning home).
    HTS uses trip survey weights; synthetic records each count as one trip.
    """
    import matplotlib.pyplot as plt
    suffix = 'district' if level == 'district' else 'dept'
    areas = [a for a in (CODES if level == 'district' else DEPT)
             if a in set(syn_trips['home_' + suffix].dropna())]
    tables = []
    long_rows = []
    for source, trips, column, weight in [
        ('HTS weighted', hts_trips, 'donor_' + suffix, 'trip_w'),
        ('Synthetic', syn_trips, 'home_' + suffix, None),
    ]:
        rows = []
        for code in areas:
            subset = trips.loc[trips[column].eq(code)]
            weights = (pd.to_numeric(subset[weight],errors='coerce').fillna(0)
                       if weight is not None else pd.Series(1.0,index=subset.index))
            denominator = weights.sum()
            row = {
                'Area': CODES.get(code,code) if level == 'district' else DEPT.get(code,code),
                'Trips': len(subset),
            }
            for purpose in PURPOSES:
                num = weights.loc[subset.purpose.eq(purpose)].sum()
                val = 100.0 * num / denominator if denominator > 0 else np.nan
                row[purpose.capitalize()] = val
                long_rows.append({
                    'area': code, 'source': source,
                    'purpose': purpose, 'share_pct': val,
                    'trip_records': int(len(subset)),
                })
            rows.append(row)
        tables.append((source, pd.DataFrame(rows)))

    # Same categorical composition as the previous stacked-bar figure, but
    # readable tables with the denominator and destination-purpose definition.
    n = len(areas)
    height = 7.8 if level == 'department' else 12.4
    fig, axes = plt.subplots(2, 1, figsize=(15.1, height))
    # Reserve substantial fixed space for the header/definition; tables should
    # never intrude into the explanation area. Avoid constrained_layout here.
    fig.subplots_adjust(top=.755, bottom=.08, hspace=.30, left=.04, right=.98)
    fig.suptitle("Trip destination-purpose mix by residents' home " + level,
                 fontsize=15, fontweight='bold', y=.945)
    fig.text(
        .04, .887,
        'This table shows the distribution of trip purposes for residents of each district or department.\n'
        'Each percentage represents the share of trips made for a particular purpose (home, work, education, shopping, leisure, or other).\n'
        'Trips are grouped by the traveler’s home district, regardless of where the trip actually ends.\n'
        'HTS results are weighted using survey trip weights (Fex), while synthetic results represent assigned trips.',
        va='top', ha='left', fontsize=9.3, color='#4A5568'
    )
    for ax, (source, table) in zip(axes, tables):
        ax.axis('off')
        ax.set_title(f'{source} — percentages of trips by following activity purpose',
                     loc='left', fontsize=11.5, weight='bold', pad=12)
        display = table.copy()
        for cat in [p.capitalize() for p in PURPOSES]:
            display[cat] = display[cat].map(lambda v: 'n/a' if pd.isna(v) else f'{v:.1f}%')
        display['Trips'] = display['Trips'].map(lambda v: f'{v:,}')
        headings = ['Residential area', 'Trip records'] + [p.capitalize() for p in PURPOSES]
        plotcols = ['Area', 'Trips'] + [p.capitalize() for p in PURPOSES]
        tab = ax.table(
            cellText=display[plotcols].values, colLabels=headings,
            cellLoc='center', loc='center',
            colWidths=[.24, .115] + [.1075]*6,
            bbox=[0,0.02,1,.91],
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.4 if level=='district' else 10)
        for (r,c),cell in tab.get_celld().items():
            cell.set_edgecolor('#E1E5EA')
            if r==0:
                cell.set_facecolor('#E9EEF5');cell.set_text_props(weight='bold')
            elif r%2==0:
                cell.set_facecolor('#F7F9FB')
            if c==0 and r>0:
                cell.set_text_props(weight='bold',ha='left')

    fig.text(.04,.030,'A 100% row represents all trip records of that home-area group, not all trips physically ending in the area.',
             fontsize=8.8,color='#4A5568',ha='left')
    _save(fig,out,matching_kind='trip')
    return pd.DataFrame(long_rows), tables


def _source_diary_check(p,t,htrip):
    a=t.groupby(['person_id','purpose']).size().rename('synthetic_count').reset_index()
    b=htrip.groupby(['hts_id','purpose']).size().rename('donor_count').reset_index()
    # Check every synthetic person including zero-trip people, using outer merge.
    pairs=p[['person_id','hts_id']].merge(b,on='hts_id',how='left')
    pairs=pairs.merge(a,on=['person_id','purpose'],how='outer')
    pairs['donor_count']=pairs.donor_count.fillna(0)
    pairs['synthetic_count']=pairs.synthetic_count.fillna(0)
    # Note: outer join on zero-trip people produces an NA purpose row with 0 counts
    mismatches=pairs[(pairs.donor_count!=pairs.synthetic_count)].person_id.dropna().unique()
    return set(mismatches),int(len(mismatches))


def _plot_ipu_household_reuse(counts, total_households, out):
    """IPU analogue to 06 diary-person donor reuse; household donors only."""
    import matplotlib.pyplot as plt
    copies = counts.number_of_synthetic_households.to_numpy(dtype=int)
    districts = counts.number_of_recipient_districts.to_numpy(dtype=int)
    fig, axs = plt.subplots(1, 2, figsize=(12.7, 4.9), layout='constrained')
    axs[0].hist(copies,bins=np.arange(.5,copies.max()+1.5),color='#287963',edgecolor='white',linewidth=.5)
    axs[0].set_xlabel('Synthetic households using one HTS IPU seed household')
    axs[0].set_ylabel('Number of distinct HTS seed households')
    axs[0].set_title('IPU household seed donor reuse')
    axs[1].hist(districts,bins=np.arange(.5,districts.max()+1.5),color='#66A78C',edgecolor='white',linewidth=.5)
    axs[1].set_xlabel('Recipient districts reached by an IPU seed household')
    axs[1].set_ylabel('Number of distinct HTS seed households')
    axs[1].set_title('Geographical spread of IPU seed households')
    for ax in axs:ax.grid(axis='y',alpha=.17)
    fig.suptitle('Original IPU household donor reuse (one record per synthetic household)',fontweight='bold',fontsize=13.8)
    _save(fig,out/'06_IPU_household_reuse.png',matching_kind='ipu')
    frame=pd.DataFrame([
        {'Measure':'Synthetic households', 'Value':f'{total_households:,}'},
        {'Measure':'Unique HTS IPU seed household donors', 'Value':f'{len(counts):,}'},
        {'Measure':'Maximum reuse of one seed household', 'Value':f'{copies.max():,}'},
        {'Measure':'Top 10 seed donor share of synthetic households','Value':f'{100*np.sort(copies)[-10:].sum()/total_households:.2f}%'},
        {'Measure':'Number of recipients from seeds used exactly once','Value':f'{(copies==1).sum():,}'},
    ])
    frame.to_csv(out/'06_IPU_household_reuse_summary.csv',index=False)
    _make_table(frame,'IPU seed-household reuse — summary',out/'06_IPU_household_reuse_summary.png',matching_kind='ipu')


def _ipu_seed_assignment(sampled,sp,out):
    """Separate audit of geographic IPU household seed source, when stage is available."""
    # Avoid displaying stale IPU reuse files from an earlier pipeline run when
    # the current sampled population lacks original donor provenance.
    for filename in ['06_IPU_household_reuse.png', '06_IPU_household_reuse.csv',
                     '06_IPU_household_reuse_summary.png', '06_IPU_household_reuse_summary.csv',
                     '06_IPU_household_reuse_NOT_AVAILABLE.txt',
                     '09_IPU_seed_household_reuse.csv']:
        old=out/filename
        if old.exists():old.unlink()
    if sampled is None or 'hts_household_district_id' not in sampled.columns:
        reason = 'sampled IPU source household district is not available'
        (out/'06_IPU_household_reuse_NOT_AVAILABLE.txt').write_text(
            reason+'\nOriginal seed provenance is needed for the household donor reuse histogram.\n',encoding='utf-8')
        return {'available':False,'reason':reason}
    cols=['person_id','household_id','hts_household_district_id']
    if 'ipu_source_household_id' in sampled.columns:
        cols.append('ipu_source_household_id')
    ipu=sampled[cols].copy()
    if ipu.person_id.duplicated().any():
        return {'available':False,'reason':'sampled person_id is not unique'}
    ipu['ipu_source_district']=ipu.hts_household_district_id.map(_id_code)
    ipu['ipu_source_dept']=ipu.ipu_source_district.str[:2]
    extra_cols=['ipu_source_household_id'] if 'ipu_source_household_id' in ipu.columns else []
    df=sp[['person_id','household_id','home_district','home_dept','donor_district']].merge(
        ipu[['person_id','household_id','ipu_source_district','ipu_source_dept']+extra_cols],
        on=['person_id','household_id'],how='left',validate='one_to_one')
    if df.ipu_source_district.isna().any():
        return {'available':False,'reason':'sampled IPU source district missing or person IDs differ'}
    hh=df.drop_duplicates('household_id')
    vals={
        'available':True,
        'ipu_seed_same_district_households_pct':_pct(hh.home_district==hh.ipu_source_district),
        'ipu_seed_same_department_households_pct':_pct(hh.home_dept==hh.ipu_source_dept),
        'ipu_seed_same_district_people_pct':_pct(df.home_district==df.ipu_source_district),
        'ipu_seed_same_department_people_pct':_pct(df.home_dept==df.ipu_source_dept),
        'matched_diary_and_ipu_seed_same_district_pct':_pct(df.donor_district==df.ipu_source_district),
    }
    if extra_cols:
        households = hh[['household_id','ipu_source_household_id','home_district','home_dept']].copy()
        if households.ipu_source_household_id.isna().any():
            raise ValueError('Cannot measure IPU household donor reuse: ipu_source_household_id has missing values')
        # Original donor IDs are only valid when recorded BEFORE integerization.
        counts = (households.groupby('ipu_source_household_id')
                  .agg(number_of_synthetic_households=('household_id','nunique'),
                       number_of_recipient_districts=('home_district','nunique'),
                       number_of_recipient_departments=('home_dept','nunique'))
                  .reset_index())
        counts.to_csv(out/'09_IPU_seed_household_reuse.csv', index=False)
        counts.to_csv(out/'06_IPU_household_reuse.csv', index=False)
        vals['ipu_unique_seed_household_donors'] = int(counts.ipu_source_household_id.nunique())
        vals['ipu_max_seed_household_reuse'] = int(counts.number_of_synthetic_households.max())
        vals['ipu_top10_seed_household_reuse_pct'] = float(100*counts.number_of_synthetic_households.nlargest(10).sum()/len(hh))
        vals['ipu_households_from_seed_used_once_pct'] = float(100*counts.loc[counts.number_of_synthetic_households.eq(1),'number_of_synthetic_households'].sum()/len(hh))
        _plot_ipu_household_reuse(counts, len(hh), out)
    else:
        vals['ipu_household_reuse_available'] = False
        (out/'06_IPU_household_reuse_NOT_AVAILABLE.txt').write_text(
            'Exact original IPU seed household donor ID not available.\n'
            'To enable the histogram, preserve ipu_source_household_id = original HTS household_id '
            'in asuncion/ipu/population.py before integerization, and ensure it survives into '
            'synthesis.population.sampled. The existing synthetic household_id is not a source donor ID.\n',
            encoding='utf-8')
    for level in ['district','department']:
        order=list(CODES) if level=='district' else list(DEPT)
        order=[k for k in order if k in set(df.home_district if level=='district' else df.home_dept)]
        m_hh=_matrix_counts(hh['home_district' if level=='district' else 'home_dept'],
                            hh['ipu_source_district' if level=='district' else 'ipu_source_dept'],order)
        m_p=_matrix_counts(df['home_district' if level=='district' else 'home_dept'],
                           df['ipu_source_district' if level=='district' else 'ipu_source_dept'],order)
        m_hh.to_csv(out/f'09_ipu_seed_household_matrix_{level}.csv')
        m_p.to_csv(out/f'09_ipu_seed_person_matrix_{level}.csv')
        _matrix_pair_plot(m_hh,m_p,level,out/f'09_ipu_seed_donor_transfer_{level}.png',heading='IPU seed-household origin versus synthetic residence',titles=['Household-weighted IPU seed assignments','Person-weighted IPU seed assignments'],matching_kind='ipu')
    return vals


def _spatial_trip_od_audit(trips,hts_trips,locations,out):
    """Compare real trip OD; requires spatial location stage commune_id for each activity.

    This is NOT a donor-assignment matrix; calculated from destinations actually
    chosen by the spatial stages. We use no approximate centroid or home inference.
    """
    if locations is None or 'commune_id' not in locations.columns:
        return {'available':False,'reason':'Spatial locations with commune_id not supplied; no synthetic trip OD computed'}
    keys=['person_id','activity_index','commune_id']
    if not set(keys).issubset(locations.columns):
        return {'available':False,'reason':'Spatial locations missing activity_index/person_id/commune_id'}
    loc=locations[keys].copy()
    if loc.duplicated(['person_id','activity_index']).any():
        return {'available':False,'reason':'Spatial locations not unique by person_id,activity_index'}
    loc['district']=loc.commune_id.map(_id_code)
    needed=['person_id','trip_index','preceding_activity_index','following_activity_index']
    if not set(needed).issubset(trips.columns):
        return {'available':False,'reason':'Synthetic trip activity indices unavailable'}
    od=trips[needed].copy()
    od=od.merge(loc[['person_id','activity_index','district']].rename(columns={'activity_index':'preceding_activity_index','district':'origin_district'}),on=['person_id','preceding_activity_index'],how='left',validate='many_to_one')
    od=od.merge(loc[['person_id','activity_index','district']].rename(columns={'activity_index':'following_activity_index','district':'destination_district'}),on=['person_id','following_activity_index'],how='left',validate='many_to_one')
    od['origin_dept']=od.origin_district.str[:2]
    od['destination_dept']=od.destination_district.str[:2]
    good=od.origin_district.notna() & od.destination_district.notna()
    if good.sum()==0:
        return {'available':False,'reason':'No spatial origin-destination assignments resolved'}
    # Raw or cleaned HTS OD can be compared separately, by survey weight.
    source=hts_trips.copy()
    has_original_od=set(['origin_district_id','destination_district_id']).issubset(source.columns)
    if not has_original_od:
        has_original_od=set(['IdDistritoOrigen','IdDistrito']).issubset(source.columns)
        ocol,dcol='IdDistritoOrigen','IdDistrito'
    else:
        ocol,dcol='origin_district_id','destination_district_id'
    result={'available':True,'synthetic_trips_with_district_od':int(good.sum()),
            'synthetic_od_coverage_pct':float(100*good.mean())}
    if has_original_od:
        source['origin_district']=source[ocol].map(_id_code)
        source['destination_district']=source[dcol].map(_id_code)
        source['origin_dept']=source.origin_district.str[:2]
        source['destination_dept']=source.destination_district.str[:2]
    for level in ['district','department']:
        origin='origin_district' if level=='district' else 'origin_dept'
        dest='destination_district' if level=='district' else 'destination_dept'
        areas=[x for x in (CODES if level=='district' else DEPT) if x in set(od[origin].dropna())]
        # retain external destinations in the raw CSV (not on the AMA matrix)
        od.groupby([origin,dest],dropna=False).size().rename('synthetic_trips').to_csv(out/f'10_actual_synthetic_trip_OD_{level}.csv')
        if has_original_od:
            source.groupby([origin,dest],dropna=False).trip_w.sum().rename('weighted_HTS_trips').to_csv(out/f'10_original_HTS_trip_OD_{level}.csv')
        a=_matrix_counts(od.loc[good,origin],od.loc[good,dest],areas)
        if has_original_od:
            # HTS flow count weights here are expansion weights, not unweighted trips.
            orig=source[source[origin].isin(areas)&source[dest].isin(areas)]
            b=orig.pivot_table(index=origin,columns=dest,values='trip_w',aggfunc='sum',fill_value=0).reindex(index=areas,columns=areas,fill_value=0)
            _matrix_pair_plot(b,a,level,out/f'10_actual_trip_OD_HTS_vs_synthetic_{level}.png',heading='Physical trip OD comparison: HTS survey vs synthetic',titles=['HTS observed weighted trip OD','Synthetic final spatial trip OD'],axis_labels=('Trip destination '+level,'Trip origin '+level))
    result['synthetic_od_within_district_pct']=_pct(od.loc[good,'origin_district'].eq(od.loc[good,'destination_district']))
    result['synthetic_od_within_department_pct']=_pct(od.loc[good,'origin_dept'].eq(od.loc[good,'destination_dept']))
    if has_original_od:
        valid_source=source[source.origin_district.notna() & source.destination_district.notna()]
        result['hts_weighted_od_within_district_pct']=100*_weighted_avg(valid_source.assign(within=valid_source.origin_district==valid_source.destination_district),'within','trip_w')
        result['hts_weighted_od_within_department_pct']=100*_weighted_avg(valid_source.assign(within=valid_source.origin_dept==valid_source.destination_dept),'within','trip_w')
    return result


def _home_donor_map(sp,homes_gpkg,out):
    if homes_gpkg is None:
        return False
    from pathlib import Path
    if not Path(homes_gpkg).is_file():
        return False
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        homes=gpd.read_file(homes_gpkg)
        if not {'household_id','geometry'}.issubset(homes.columns):return False
        homes=homes.to_crs(4326)
        ha=sp.groupby('household_id').same_district.all().rename('all_members_local')
        geo=homes[['household_id','geometry']].merge(ha,left_on='household_id',right_index=True,how='inner')
        a=geo[geo.all_members_local];b=geo[~geo.all_members_local]
        fig,ax=plt.subplots(figsize=(9,9))
        ax.scatter(a.geometry.x,a.geometry.y,c='#2D8879',s=5,alpha=.35,label=f'All local ({len(a):,})')
        ax.scatter(b.geometry.x,b.geometry.y,c='#D27E54',s=6,alpha=.6,label=f'At least one external donor ({len(b):,})')
        ax.set_title('Synthetic household home locations vs HTS diary donor geography')
        ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.legend(frameon=False)
        ax.set_aspect('equal',adjustable='datalim');fig.tight_layout()
        _save(fig,out/'11_local_donor_households_map.png')
        return True
    except Exception as exc:
        warnings.warn(f'Could not render home donor map: {exc}')
        return False


# --------------------------------------------------------------------------
# Additional audit 12: percentage of TRIPS by following purpose and age band.
# Each age row sums to 100%, and HTS trip records use the original Fex weights.
# --------------------------------------------------------------------------
def _age_purpose_mix(hts_persons, hts_trips, synth_trips, out):
    """Compare *trip* composition across 5-year age bands; not % of people.

    Zero-trip age bands are marked n/a (never silently reported as zero).
    One HTS respondent can contribute multiple trips, each weighted by Fex.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    edges = list(range(0, 81, 5)) + [np.inf]
    labels = [f'{i}–{i+4}' for i in range(0, 80, 5)] + ['80+']
    donor_age = hts_persons[['hts_id', 'age_donor']]
    if donor_age.hts_id.duplicated().any():
        raise ValueError('Cannot compute age-purpose mix: duplicate HTS person IDs')
    h = hts_trips.merge(donor_age, on='hts_id', how='left', validate='many_to_one')
    s = synth_trips.copy()
    h['age_band'] = pd.cut(h.age_donor, bins=edges, labels=labels, right=False)
    s['age_band'] = pd.cut(s.age, bins=edges, labels=labels, right=False)
    h['__weight'] = pd.to_numeric(h.trip_w, errors='coerce')
    s['__weight'] = 1.0
    rows = []
    for dep in ['ALL', '00', '11']:
        for source, data, depcol in [('HTS weighted', h, 'donor_dept'),
                                     ('Synthetic', s, 'home_dept')]:
            df = data if dep == 'ALL' else data.loc[data[depcol].eq(dep)]
            for band in labels:
                bb = df.loc[df.age_band.eq(band)]
                w = bb['__weight'].where(bb['__weight'].gt(0), 0).fillna(0)
                denom = float(w.sum())
                for purpose in PURPOSES:
                    num = float(w.loc[bb.purpose.eq(purpose)].sum())
                    rows.append(dict(
                        department=dep, source=source, age_band=band,
                        purpose=purpose, trip_records=int(len(bb)),
                        weighted_trip_total=denom,
                        trip_share_pct=(100 * num / denom) if denom > 0 else np.nan,
                    ))
    long = pd.DataFrame(rows)
    long.to_csv(out/'12_trip_purpose_by_age_5year.csv', index=False)

    # A heatmap is more legible than 6 overlapping 5-year line series.
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad('#ECEFF1')
    for dep in ['ALL', '00', '11']:
        df = long.loc[long.department.eq(dep)]
        fig, axs = plt.subplots(1, 2, figsize=(14.5, 10.2), layout='constrained')
        for ax, source in zip(axs, ['HTS weighted', 'Synthetic']):
            a = df.loc[df.source.eq(source)]
            mat = a.pivot(index='age_band', columns='purpose', values='trip_share_pct').reindex(index=labels, columns=PURPOSES)
            matrix = mat.to_numpy(float)
            im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto',
                           cmap=cmap, norm=Normalize(vmin=0, vmax=100))
            ax.set_xticks(range(len(PURPOSES)), [x.capitalize() for x in PURPOSES], rotation=30, ha='right')
            ax.set_yticks(range(len(labels)), labels)
            ax.set_ylabel('Age of traveler (years)')
            ax.set_title(source + (' (Fex trip weights)' if source == 'HTS weighted' else ' (assigned trips)'), fontsize=12)
            for r in range(len(labels)):
                for c in range(len(PURPOSES)):
                    v = matrix[r, c]
                    txt = '–' if not np.isfinite(v) else f'{v:.0f}'
                    ax.text(c, r, txt, fontsize=7.3, ha='center', va='center',
                            color=('white' if np.isfinite(v) and v > 57 else '#1B2933'))
        fig.colorbar(im, ax=axs, label='Share of trips within this age band (%)', shrink=.70)
        desc = 'Asunción + Central' if dep == 'ALL' else DEPT[dep]
        fig.suptitle(f'Trip destination-purpose mix by traveler age (5-year groups) — {desc}\n'
                     'Each age row = 100% of trips in that age group, NOT 100% of people',
                     fontsize=14, weight='bold')
        _save(fig, out/f'12_trip_purpose_by_age_5year_{dep}.png', matching_kind='trip')
    return long


# --------------------------------------------------------------------------
# Additional audit 13: person-level plausibility flags.
# NOTE: the exported synthetic trips.csv has activity purposes but NO mode.
# Car-driving indicators therefore refer to the matched donor's *recorded*
# HTS driver trips, if the raw HTS trip IDs and EtapasAutoConductor are present.
# These are potential diary/mode incompatibilities, NOT observed simulated modes.
# --------------------------------------------------------------------------
def _donor_driver_flags(hts_raw_trips, normalized_filtered_trips, hts_persons):
    if hts_raw_trips is None:
        return None, 'Original HTS Viaje.csv not supplied; cannot identify car-driver trips.'
    raw = hts_raw_trips.copy()
    id_col = next((c for c in ['Id', 'trip_id'] if c in raw.columns), None)
    person_col = next((c for c in ['IdPersona', 'person_id', 'hts_id'] if c in raw.columns), None)
    stage_col = next((c for c in ['EtapasAutoConductor', 'auto_driver_legs'] if c in raw.columns), None)
    if id_col is None or person_col is None or stage_col is None:
        return None, 'HTS raw trips lack ID, person ID, or EtapasAutoConductor; driver status unavailable.'
    raw['trip_id_compare'] = pd.to_numeric(raw[id_col], errors='coerce')
    raw['hts_id_compare'] = pd.to_numeric(raw[person_col], errors='coerce')
    raw['has_car_driver_stage'] = pd.to_numeric(raw[stage_col], errors='coerce').gt(0)
    raw = raw[['trip_id_compare', 'hts_id_compare', 'has_car_driver_stage']]
    if raw.duplicated(['trip_id_compare', 'hts_id_compare']).any():
        raise ValueError('Raw HTS trip IDs cannot be duplicated when checking car driver stages')
    t = normalized_filtered_trips
    filter_id = next((c for c in ['trip_id', 'Id'] if c in t.columns), None)
    if filter_id is not None:
        selected = pd.DataFrame({
            'trip_id_compare': pd.to_numeric(t[filter_id], errors='coerce'),
            'hts_id_compare': pd.to_numeric(t.hts_id, errors='coerce'),
        })
        if selected.isna().any().any():
            return None, 'Filtered trip IDs are missing; cannot reliably join the car-driver stages.'
        if selected.duplicated().any():
            return None, 'Filtered trip IDs repeat; cannot reliably join the car-driver stages.'
        matched = selected.merge(raw, on=['trip_id_compare', 'hts_id_compare'], how='left', validate='one_to_one')
        coverage = matched.has_car_driver_stage.notna().mean() if len(matched) else 0
        if coverage < .995:
            return None, f'Only {coverage:.1%} of filtered HTS trips match the raw driver-stage records; driver flags omitted.'
        raw = matched
        note = f'HTS donor diary car-driver stages matched to {len(matched):,} filtered HTS trips by ID.'
    else:
        note = 'Filtered HTS trip IDs absent: car-driver flag uses ALL raw HTS diary records per donor (less precise).'
    by_person = raw.groupby('hts_id_compare').has_car_driver_stage.any()
    all_flags = hts_persons[['hts_id']].copy()
    all_flags['has_donor_driver_trip'] = all_flags.hts_id.map(by_person).eq(True).astype(bool)
    return all_flags, note


def _plausibility_audit(sp, synth_households, raw_driver_trips, hts_filtered_trips,
                        hts_persons, out):
    import matplotlib.pyplot as plt
    ppl = sp.copy()
    household_cols = ['household_id', 'number_of_cars', 'number_of_motorbikes']
    owned = ['household_id'] + [c for c in household_cols[1:] if c in synth_households.columns]
    hh = synth_households[owned].copy()
    if hh.household_id.duplicated().any():
        raise ValueError('Cannot compute matching flags: synthetic household IDs repeat')
    ppl = ppl.merge(hh, on='household_id', how='left', validate='many_to_one')
    driver, driver_note = _donor_driver_flags(raw_driver_trips, hts_filtered_trips, hts_persons)
    if driver is not None:
        ppl = ppl.merge(driver, on='hts_id', how='left', validate='many_to_one')
        if ppl.has_donor_driver_trip.isna().any():
            raise ValueError('Matched synthetic people include missing donor-driver flags')
    else:
        ppl['has_donor_driver_trip'] = pd.NA

    # A check is evaluated ONLY for the population where it is meaningful.
    # Missing metadata remain unknown, never silently marked as non-violations.
    employment = ppl.employed_synth.astype('boolean')
    license_status = ppl.licence_synth.astype('boolean')
    driver_status = ppl.has_donor_driver_trip.astype('boolean')
    cars = pd.to_numeric(ppl['number_of_cars'], errors='coerce') if 'number_of_cars' in ppl.columns else pd.Series(np.nan, index=ppl.index)
    motorcycles = pd.to_numeric(ppl['number_of_motorbikes'], errors='coerce') if 'number_of_motorbikes' in ppl.columns else pd.Series(np.nan, index=ppl.index)

    checks = [
        # (machine key, display label, denominator / eligible group, mismatch condition, interpretation)
        ('work_trip_unemployed', 'Work trip while not employed (age 15+)',
         (employment.eq(False) & ppl.age.ge(15)), ppl.has_work_trip.astype('boolean'),
         'Potential purpose/employment inconsistency, not proof of error'),
        ('work_trip_under15', 'Work trip at age under 15',
         ppl.age.lt(15).astype('boolean'), ppl.has_work_trip.astype('boolean'),
         'Review respondent age and work-purpose classification'),
        ('education_trip_under4', 'Education trip at age under 4',
         ppl.age.lt(4).astype('boolean'), ppl.has_education_trip.astype('boolean'),
         'Likely age-bin donor matching artefact'),
        ('trips_under4', 'Any trips at age under 4',
         ppl.age.lt(4).astype('boolean'), ppl.traveller.astype('boolean'),
         'Review travel-survey coverage and age-bin matching'),
        ('donor_age_gap_gt10', 'Donor age differs by more than 10 years',
         pd.Series(True,index=ppl.index,dtype='boolean'), ppl.age_abs_error.gt(10).astype('boolean'),
         'Matching-quality flag, not necessarily impossible travel'),
        ('employment_donor_mismatch', 'HTS donor employment differs',
         pd.Series(True,index=ppl.index,dtype='boolean'), ppl.same_employed.eq(False).astype('boolean'),
         'Should normally be zero if employment is a hard matching condition'),
        ('sex_donor_mismatch', 'HTS donor sex differs',
         pd.Series(True,index=ppl.index,dtype='boolean'), ppl.same_sex.eq(False).astype('boolean'),
         'Should normally be zero if sex is a hard matching condition'),
        ('licence_donor_mismatch', 'HTS donor driving licence differs',
         license_status.notna().astype('boolean'), ppl.same_licence.eq(False).astype('boolean'),
         'Survey licence fields must have consistent meanings'),
        ('diary_count_mismatch', 'Assigned trip count differs from donor diary',
         pd.Series(True,index=ppl.index,dtype='boolean'), ppl.count_equal.eq(False).astype('boolean'),
         'Diary inheritance consistency check'),
    ]
    if driver is not None:
        checks.extend([
            ('donor_driver_carless', 'Donor has driver trip; household has no car',
             (driver_status.eq(True) & cars.notna()).astype('boolean'), cars.eq(0).astype('boolean'),
             'POTENTIAL donor-mode conflict; borrowed vehicles could make it valid'),
            ('donor_driver_no_licence', 'Donor has driver trip; person has no licence',
             (driver_status.eq(True) & license_status.notna()).astype('boolean'), license_status.eq(False).astype('boolean'),
             'POTENTIAL donor-mode conflict; synthetic trip mode is not exported'),
            ('donor_driver_under18', 'Donor has driver trip; person under 18',
             driver_status.eq(True).astype('boolean'), ppl.age.lt(18).astype('boolean'),
             'Check donor diary and legal/age consistency'),
        ])
    else:
        for key, title in [
            ('donor_driver_carless', 'Donor has driver trip; household has no car'),
            ('donor_driver_no_licence', 'Donor has driver trip; person has no licence'),
            ('donor_driver_under18', 'Donor has driver trip; person under 18'),
        ]:
            checks.append((key,title,None,None,'Not available: '+driver_note))

    rows = []
    for level, group_items in [
        ('ALL', [('ALL', ppl)]),
        ('department', [(d, ppl.loc[ppl.home_dept.eq(d)]) for d in DEPT]),
        ('district', [(d, ppl.loc[ppl.home_district.eq(d)]) for d in CODES]),
    ]:
        for area, group in group_items:
            size=len(group)
            for key,title,eligible,condition,note in checks:
                if eligible is None:
                    rows.append(dict(level=level,area=area,check=key,label=title,
                        affected_people=np.nan,eligible_people=np.nan,all_people=size,
                        pct_all_people=np.nan,pct_eligible_people=np.nan,
                        interpretation=note))
                    continue
                # Only index this group's rows. The original series are indexed by person.
                a=eligible.reindex(group.index).fillna(False).astype(bool)
                b=condition.reindex(group.index).fillna(False).astype(bool)
                n=int((a & b).sum()); d=int(a.sum())
                rows.append(dict(level=level,area=area,check=key,label=title,
                    affected_people=n,eligible_people=d,all_people=size,
                    pct_all_people=100*n/size if size else np.nan,
                    pct_eligible_people=100*n/d if d else np.nan,
                    interpretation=note))
    result=pd.DataFrame(rows)
    result.to_csv(out/'13_person_matching_plausibility_by_area.csv',index=False)
    main=result.loc[result.level.eq('ALL')].copy()
    main.to_csv(out/'13_person_matching_plausibility_overall.csv',index=False)
    # Deliberately keep this as a table rather than comparing unrelated denominator bars.
    display=main[['label', 'affected_people', 'eligible_people', 'pct_all_people', 'pct_eligible_people']].copy()
    display.columns=['Potential incompatibility', 'People', 'Eligible', '% all people', '% eligible']
    for col in ['People', 'Eligible']:
        display[col]=display[col].map(lambda v:'N/A' if pd.isna(v) else f'{int(v):,}')
    for col in ['% all people', '% eligible']:
        display[col]=display[col].map(lambda v:'N/A' if pd.isna(v) else f'{v:.2f}%')
    fig,ax=plt.subplots(figsize=(15.4, 9.5))
    fig.subplots_adjust(left=.025,right=.975,top=.76,bottom=.15)
    ax.axis('off')
    tbl=ax.table(cellText=display.values,colLabels=display.columns,cellLoc='center',loc='center',
                 colWidths=[.54,.095,.095,.13,.14],bbox=[0,0,1,1])
    tbl.auto_set_font_size(False);tbl.set_fontsize(9)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor('#DDE2E6')
        if r==0:
            cell.set_facecolor('#E5EAF1');cell.set_text_props(weight='bold')
        elif r%2==0:cell.set_facecolor('#F7F8FA')
        if c==0 and r>0:cell.set_text_props(ha='left')
    fig.suptitle('Person-level trip diary matching: plausibility flags',y=.93,fontsize=15,fontweight='bold')
    fig.text(.028,.865,
             'Each row counts DISTINCT people, not individual trips. “% eligible” uses only the relevant comparison group.\n'
             'Work-purpose trips and vehicle availability are screening flags, not necessarily impossible behaviour.\n'
             'Car-driver checks describe original HTS DONOR diaries, NOT simulated transport modes.',
             ha='left',va='top',fontsize=10,color='#425466')
    fig.text(.028,.057,'Driver-trip data: '+driver_note,ha='left',va='bottom',fontsize=8.7,color='#425466')
    _save(fig,out/'13_person_matching_plausibility_table.png',matching_kind='trip')
    return result, main, driver_note


def create_matching_report(synth_persons,synth_households,synth_trips,
                           hts_households,hts_persons,hts_trips,output_dir,
                           age_bins=None,homes_gpkg=None,ipu_sampled=None,spatial_locations=None,raw_hts_trips=None):
    """Generate tables/plots from exported CSV frames or raw/filtered HTS frames.

    `age_bins` should equal the pipeline's age_matching_bins; when absent uses
    defaults and reports that it was an audit diagnostic, not a guaranteed match
    to your actual config.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    hhh,hp,ht=_normalize_hts(hts_households,hts_persons,hts_trips)
    sp,sh,st=_prep_synthetic(synth_persons,synth_households,synth_trips,hp)
    if age_bins is None:age_bins=DEFAULT_AGE_BINS
    sp['same_age_bin']=(np.digitize(sp.age.to_numpy(float),age_bins,right=True)==np.digitize(sp.age_donor.to_numpy(float),age_bins,right=True))
    hp['age_4plus']=hp.age_donor>=4
    hp['school_age']=hp.age_donor.between(6,17)
    hp['employment_age']=hp.age_donor>=15
    hp['licence_age']=hp.age_donor>=18
    hcounts=ht.groupby('hts_id').size()
    hp['n_trips_donor']=hp.hts_id.map(hcounts).fillna(0).astype(int)
    hp['traveller']=hp.n_trips_donor.gt(0)
    sp['n_trips_donor']=sp.hts_id.map(hcounts).fillna(0).astype(int)
    sp['count_equal']=sp.n_trips_synth.eq(sp.n_trips_donor)
    for purpose in ['education','work']:
        nums=ht[ht.purpose==purpose].groupby('hts_id').size()
        hp[f'{purpose}_trips']=hp.hts_id.map(nums).fillna(0).astype(int)
        hp[f'has_{purpose}_trip']=hp[f'{purpose}_trips'].gt(0)

    _,purpose_mismatches=_source_diary_check(sp,st,ht)
    # donor transfer matrices: weighted by synthetic people, then by trips
    order=[code for code in CODES if code in set(sp.home_district)]
    all_depts=sorted(sp.home_dept.dropna().unique())
    for level,p_source,t_source,items in [
        ('district','home_district','donor_district',order),
        ('department','home_dept','donor_dept',all_depts)]:
        a=_matrix_counts(sp[p_source],sp[t_source],items)
        b=_matrix_counts(st[p_source],st[t_source],items)
        a.to_csv(out/f'01_person_donor_matrix_{level}.csv')
        b.to_csv(out/f'02_trip_donor_matrix_{level}.csv')
        _matrix_pair_plot(a,b,level,out/f'01_02_donor_transfer_{level}.png')

    summary=[]; behaviour=[]
    for level in ['district','department']:
        if level=='district':areas=[x for x in CODES if x in set(sp.home_district)]
        else:areas=sorted(sp.home_dept.dropna().unique())
        suffix = 'district' if level == 'district' else 'dept'
        for code in areas:
            recipient=sp.loc[sp['home_'+suffix].eq(code)]
            trips=st.loc[st['home_'+suffix].eq(code)]
            donors=hp.loc[hp['donor_'+suffix].eq(code)]
            m=recipient['same_district' if level=='district' else 'same_dept']
            mt=trips['home_'+suffix].eq(trips['donor_'+suffix])
            h4=donors[donors.age_4plus]
            s4=recipient[recipient.age_4plus]
            ha=donors[donors.school_age];sa=recipient[recipient.school_age]
            he=donors[donors.employment_age & donors.employed_donor]
            se=recipient[recipient.employment_age & recipient.employed_synth.eq(True)]
            entry=dict(level=level,code=code,area=CODES.get(code,code) if level=='district' else DEPT.get(code,code),
                n_people=len(recipient),n_trips=len(trips),n_hts_people=len(donors),
                hts_effective_n=_eff_n(donors.person_weight),
                local_people_pct=_pct(m),local_trips_pct=_pct(mt),
                people_cross_department_pct=100*(~recipient.same_dept).mean(),
                trips_cross_department_pct=100*(~trips.home_dept.eq(trips.donor_dept)).mean() if len(trips) else np.nan,
                unique_donor_people=recipient.hts_id.nunique(),
                mean_absolute_age_error=recipient.age_abs_error.mean(),
                same_age_bin_pct=_pct(recipient.same_age_bin),
                same_sex_pct=_pct(recipient.same_sex),
                same_employed_pct=_pct(recipient.same_employed.dropna()),
                same_licence_pct=_pct(recipient.same_licence.dropna()),
                trip_count_matching_pct=_pct(recipient.count_equal),
                donor_trips_per_person=recipient.n_trips_donor.mean(),
                synthetic_trips_per_person=recipient.n_trips_synth.mean())
            summary.append(entry)
            behaviour.append(dict(level=level,code=code,area=entry['area'],
                hts_trips_age4plus=_weighted_avg(h4,'n_trips_donor'),
                syn_trips_age4plus=s4.n_trips_synth.mean(),
                hts_immobile_pct=100*(1-_weighted_avg(donors,'traveller')),
                syn_immobile_pct=_pct(~recipient.traveller),
                hts_education_6to17_pct=100*_weighted_avg(ha,'has_education_trip'),
                syn_education_6to17_pct=_pct(sa.has_education_trip),
                hts_work_trip_employed_pct=100*_weighted_avg(he,'has_work_trip'),
                syn_work_trip_employed_pct=_pct(se.has_work_trip),
                hts_licence18_pct=100*_weighted_avg(donors[donors.licence_age & donors.license_response_valid],'licence_donor'),
                syn_licence18_pct=_pct(recipient[recipient.licence_age].licence_synth.dropna())))
        sm=pd.DataFrame([r for r in summary if r['level']==level]);bm=pd.DataFrame([r for r in behaviour if r['level']==level])
        sm.to_csv(out/f'03_assignment_comparison_{level}.csv',index=False)
        bm.to_csv(out/f'04_behaviour_comparison_{level}.csv',index=False)
        _bar_table(sm,level,out/f'03_locality_{level}.png')
        _behaviour_plot(bm,level,out/f'04_behaviour_{level}.png')
        mix, purpose_tables = _purpose_plot(st,ht,level,out/f'05_purpose_mix_{level}.png')
        mix.to_csv(out/f'05_purpose_mix_{level}.csv',index=False)
        for source,tab in purpose_tables:
            suffix = 'hts' if source == 'HTS weighted' else 'synthetic'
            tab.to_csv(out/f'05_purpose_table_{level}_{suffix}.csv',index=False)
        # Tables sized for legibility rather than trying to fit everything into a figure
        select=['area','n_people','n_trips','local_people_pct','local_trips_pct','same_age_bin_pct','same_employed_pct','mean_absolute_age_error']
        tb=sm[select].copy()
        for c in ['local_people_pct','local_trips_pct','same_age_bin_pct','same_employed_pct','mean_absolute_age_error']:
            tb[c]=tb[c].map(lambda v:f'{v:.1f}' if pd.notna(v) else 'n/a')
        tb.columns=['Area','People','Trips','Persons local %','Trips local %','Age bin same %','Employment same %','Mean |age diff|']
        _make_table(tb,f'{level.capitalize()} matching: people and trips compared',out/f'03_locality_table_{level}.png',small=level=='district')
        bmt=bm[['area','hts_trips_age4plus','syn_trips_age4plus','hts_education_6to17_pct','syn_education_6to17_pct']].copy()
        for c in bmt.columns[1:]:bmt[c]=bmt[c].map(lambda v:f'{v:.2f}' if pd.notna(v) else 'n/a')
        bmt.columns=['Area','HTS trips 4+','Syn trips 4+','HTS edu 6–17 %','Syn edu 6–17 %']
        _make_table(bmt,f'{level.capitalize()} travel behaviour: HTS vs synthetic',out/f'04_behaviour_table_{level}.png',small=level=='district')

    counts=sp.hts_id.value_counts()
    rec=sp.groupby(['hts_id','donor_district'],dropna=False).agg(
        copies=('person_id','size'),recipient_districts=('home_district','nunique'),recipient_departments=('home_dept','nunique')).reset_index()
    rec.sort_values('copies',ascending=False).to_csv(out/'06_donor_reuse.csv',index=False)
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axs[0].hist(counts.to_numpy(),bins=np.arange(.5,counts.max()+1.5),color='#4B87A0')
    axs[0].set_xlabel('Number of synthetic people reusing donor')
    axs[0].set_ylabel('Number of unique HTS donors');axs[0].set_title('Trip matching: repeated HTS person diaries')
    axs[1].hist(rec.recipient_districts.to_numpy(),bins=np.arange(.5,rec.recipient_districts.max()+1.5),color='#CA7850')
    axs[1].set_xlabel('Distinct synthetic districts reached by donor');axs[1].set_ylabel('Number of HTS donors')
    axs[1].set_title('Spread of person diary donors across recipient districts')
    for ax in axs:ax.grid(axis='y',alpha=.15)
    fig.suptitle('One HTS person donor supplies one entire travel diary per assigned synthetic person',fontsize=12.2,fontweight='bold')
    _save(fig,out/'06_donor_reuse.png',matching_kind='trip')

    # Cross department vs within department share of all assigned trips, by purpose
    purpose_transfer=st.groupby(['home_dept','donor_dept','purpose'],dropna=False).size().rename('assigned_trips').reset_index()
    purpose_transfer.to_csv(out/'07_trip_transfer_purpose_departments.csv',index=False)
    donor_origin=sp.groupby('donor_district').size().rename('assigned_synthetic_people').to_frame()
    donor_origin['hts_sample_people']=hp.groupby('donor_district').size()
    donor_origin['hts_weighted_people']=hp.groupby('donor_district').person_weight.sum()
    donor_origin['share_assigned_pct']=100*donor_origin.assigned_synthetic_people/donor_origin.assigned_synthetic_people.sum()
    donor_origin['share_hts_weighted_pct']=100*donor_origin.hts_weighted_people/donor_origin.hts_weighted_people.sum()
    donor_origin['difference_pp']=donor_origin.share_assigned_pct-donor_origin.share_hts_weighted_pct
    donor_origin.to_csv(out/'08_donor_pool_reweighting.csv')

    ipu_report=_ipu_seed_assignment(ipu_sampled,sp,out)
    physical_od_report=_spatial_trip_od_audit(st,ht,spatial_locations,out)
    mapped_homes=_home_donor_map(sp,homes_gpkg,out)

    # New optional reports 12 and 13; all previous plots are unchanged.
    age_purpose = _age_purpose_mix(hp, ht, st, out)
    plausibility, plausibility_all, driver_note = _plausibility_audit(
        sp, sh, raw_hts_trips, ht, hp, out)

    # Sanity diagnostics and scope limitations.
    diagnostics={
        'synthetic_people':int(len(sp)), 'synthetic_households':int(sh.household_id.nunique()),'synthetic_trips':int(len(st)),
        'hts_people':int(len(hp)), 'hts_households':int(len(hhh)), 'hts_trip_records':int(len(ht)),
        'same_district_people_pct':_pct(sp.same_district),'same_district_trips_pct':_pct(st.home_district.eq(st.donor_district)),
        'same_department_people_pct':_pct(sp.same_dept),'same_department_trips_pct':_pct(st.home_dept.eq(st.donor_dept)),
        'same_age_bin_pct':_pct(sp.same_age_bin),'same_employed_pct':_pct(sp.same_employed.dropna()),
        'same_sex_pct':_pct(sp.same_sex),'same_license_pct':_pct(sp.same_licence.dropna()),
        'age_abs_difference_years':float(sp.age_abs_error.mean()),
        'people_whose_trip_count_differs_from_donor':int((~sp.count_equal).sum()),
        'people_whose_trip_purpose_counts_differ':int(purpose_mismatches),
        'unique_donors':int(sp.hts_id.nunique()),'max_reuse':int(counts.max()),
        'top10_reuse_share_pct':float(100*counts.head(10).sum()/len(sp)),
        'households_all_members_same_district_pct':float(100*sp.groupby('household_id').same_district.all().mean()),
        'children_0_to_3_with_trips':int(sp.loc[sp.age<4,'traveller'].sum()),
        'trips_for_children_0_to_3':int(st.loc[st.age<4].shape[0]),
        'education_trips_for_children_0_to_3':int(st.loc[(st.age<4)&st.purpose.eq('education')].shape[0]),
        'work_trips_for_nonemployed':int(st.loc[st.employed_synth.eq(False)&st.purpose.eq('work')].shape[0]),
        'ipu_seed_provenance':ipu_report, 'physical_trip_od':physical_od_report,
        'household_locality_map_created':mapped_homes,
        'person_plausibility_overall':plausibility_all[['check','affected_people','pct_all_people','pct_eligible_people']].to_dict('records'),
        'donor_car_driver_check_note':driver_note,
        'notes':[
            'Donor locality is the residential district of matched HTS person, not the district of each HTS trip origin or destination.',
            'Each synthetic person receives one entire HTS trip diary; trip assignment therefore uses the same donor ID as person assignment.',
            'Figure labels: [trip matching] identifies survey diary matching/travel behaviour; [IPU matching] identifies IPU seed-household selection.',
            'Actual synthetic physical trip OD locations are assigned later; donor origin-to-recipient matching is NOT a synthetic physical OD matrix.',
            'If sampled IPU population is supplied, IPU household *source district* locality is audited separately; exact household donor reuse requires ipu_source_household_id from before integerization.',
            'HTS travel behaviour by area uses survey person/trip weights; synthetic persons/trips are equal weighted.',
            'Differences can reflect demographic reweighting and small district HTS samples, not necessarily algorithm errors.',
        ]}
    (out/'00_summary.json').write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False),encoding='utf8')
    pd.DataFrame([{'metric':k,'value':v} for k,v in diagnostics.items() if k not in ('notes','ipu_seed_provenance','physical_trip_od')]).to_csv(out/'00_summary.csv',index=False)
    summary_tables=[]
    for level in ['department','district']:
        s=pd.DataFrame([r for r in summary if r['level']==level])
        summary_tables.append(s[['level','area','n_people','n_trips','local_people_pct','local_trips_pct','unique_donor_people','same_age_bin_pct','same_employed_pct','mean_absolute_age_error']])
    short=pd.concat(summary_tables,ignore_index=True)
    report=[f'# Asunción travel-diary donor validation', '',
       f'**Run:** {len(sp):,} synthetic persons, {sh.household_id.nunique():,} households, {len(st):,} synthetic trips.',
       '', '## Assignment interpretation',
       'Main results (01–08 and 11) measure HTS diary-person matching and travel behaviour; they are **not physical trip OD matrices**. Optional plots (06_IPU and 09) separately examine IPU seed-household selection when original source IDs are available.',
       '', '## Overall matching',
       *[f'- **{k}**: {v:.2f}' for k,v in diagnostics.items() if k in ('same_district_people_pct','same_district_trips_pct','same_department_people_pct','same_department_trips_pct','same_age_bin_pct','same_employed_pct','age_abs_difference_years','top10_reuse_share_pct')],
       '', '## Optional original IPU and physical-trip OD checks',
       '- IPU source district audit: '+str(ipu_report),
       '- Physical trip OD audit: '+str(physical_od_report),
       '', '## Destination-purpose mix by residence (plot 05)',
       'Tables report the shares of trips by **following activity purpose**, grouped by the resident home district/department. They do not measure physical trip destinations. The HTS table uses expansion weights; the synthetic table uses assigned trip counts.',
       '', '## IPU seed household reuse (plot 06_IPU)',
       'This plot is available only if `ipu_source_household_id` was preserved from the original HTS household before IPU integerization and survived into `synthesis.population.sampled`; otherwise it is omitted. The existing 06 donor-reuse plot refers to HTS **person diaries**, not IPU household selection.',
       '', '## District and department summary', short.round(2).to_markdown(index=False),
       '', '## Trip purpose by age (plot 12)',
       'Five-year age-band heatmaps compare weighted HTS and synthetic trip destination purposes. Each age row sums to 100% of **trips**, not people. The CSV also splits results by department.',
       '', '## Person-level matching plausibility (table 13)',
       'The table reports distinct people who meet potential incompatibility conditions, as a percentage of all people and of the relevant eligible group. The donor-car-driver conditions use HTS source travel diaries, not a synthesized mode-choice result. In a standard eqasim export, synthetic trips do **not** include transport modes.',
       'Car driver source: '+driver_note,
       '', '## Important limitations', *['- '+v for v in diagnostics['notes']],
       '', '## Files', '01 and 02: person and trip donor transfer matrices; 03: assignment rates; 04: travel behaviour; 05: destination-purpose tables grouped by residential area; 06: HTS diary-person donor reuse and optional IPU household donor reuse; 07: trip purpose by department transfer; 08: donor pool reweighting; 09: IPU seed origin matrices.',
    ]
    (out/'README_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    print('Completed HTS donor matching audit:',out)
    print(pd.Series({k:v for k,v in diagnostics.items() if k not in ('notes','ipu_seed_provenance','physical_trip_od')}).to_string())
    return diagnostics


# SynPP integration: place file at analysis/asuncion/asuncion_matching_validation.py
# then append `analysis.asuncion.asuncion_matching_validation` under config run.
def configure(context):
    context.stage('synthesis.output')
    context.stage('data.hts.entd.filtered')
    context.stage('synthesis.population.sampled')
    context.stage('synthesis.population.spatial.locations')
    for name in ['output_path','output_prefix','analysis_path']:
        context.config(name)
    context.config('age_matching_bins',DEFAULT_AGE_BINS)
    context.config('data_path')
    context.config('asuncion.hts_trips', 'hts/Viaje.csv')


def execute(context):
    context.stage('synthesis.output')
    # Optional driver-stage evidence is in the ORIGINAL HTS Viaje.csv.
    # Filtered HTS / synthetic trips.csv do not reliably identify car *drivers*.
    raw_path = Path(context.config('data_path')) / context.config('asuncion.hts_trips')
    raw_hts_trips = (pd.read_csv(raw_path, sep=';', encoding='latin1', low_memory=False)
                     if raw_path.is_file() else None)
    root=Path(context.config('output_path'))
    prefix=context.config('output_prefix')
    hts_h,hts_p,hts_t=context.stage('data.hts.entd.filtered')
    return create_matching_report(
        pd.read_csv(root/f'{prefix}persons.csv',sep=';'),
        pd.read_csv(root/f'{prefix}households.csv',sep=';'),
        pd.read_csv(root/f'{prefix}trips.csv',sep=';'),
        hts_h,hts_p,hts_t,
        Path(context.config('analysis_path'))/'matching_validation',
        age_bins=context.config('age_matching_bins'),
        ipu_sampled=context.stage('synthesis.population.sampled'),
        spatial_locations=context.stage('synthesis.population.spatial.locations'),
        homes_gpkg=(root/f'{prefix}homes.gpkg'),
        raw_hts_trips=raw_hts_trips,
    )


def validate(context):
    return []
