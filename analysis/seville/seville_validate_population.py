"""Seville census / HTS / final synthetic-population validation.

Uses actual 2022 Seville census section population and household-size targets
(rather than Asuncion-specific district/income inputs). Saves comparable
age/sex, household and employment tables + figures, plus municipality-level
population coverage and a machine-readable summary. Scales *synthetic counts*
by the configured sampling fraction, but never scales percentages.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def configure(context):
    context.stage("synthesis.output")
    context.stage("seville.data.census.population")
    context.stage("seville.data.census.households")
    context.stage("seville.data.census.employment")
    context.stage("seville.data.hts.entd.reweighted")
    context.config("output_path")
    context.config("output_prefix")
    context.config("analysis_path")
    context.config("sampling_rate", 1.0)


def _sex(x):
    return x.astype("string").str.strip().str.lower().replace({
        "0":"male", "1":"female", "m":"male", "f":"female",
        "males":"male", "females":"female", "hombre":"male", "mujer":"female"})


def _boolean(x):
    return x.astype("string").str.strip().str.lower().map({
        "true":1., "false":0., "1":1., "0":0., "yes":1., "no":0.})


def _plot_bar(table, x, fields, output, ylabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(max(9, 0.52*len(table)), 5.3))
    q=np.arange(len(table)); w=.82/len(fields)
    for i, (name,col) in enumerate(fields.items()):
        ax.bar(q+(i-(len(fields)-1)/2)*w, table[col].to_numpy(float), width=w, label=name)
    ax.set_xticks(q, table[x].astype(str), rotation=45, ha="right")
    ax.set(ylabel=ylabel,title=title)
    ax.grid(axis="y",alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"),dpi=220,bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def _percent_table(frames, group_cols, weight_cols, groupby_name=None):
    """Align weighted shares from any number of sources on a common support."""
    group_cols=[group_cols] if isinstance(group_cols,str) else group_cols
    pieces=[]
    for (label,frame),col in zip(frames.items(),weight_cols):
        s=frame.groupby(group_cols,dropna=False,observed=True)[col].sum()
        if s.sum() <= 0:raise ValueError(f"Nonpositive {label} totals")
        pieces.append((s/s.sum()*100).rename(label))
    return pd.concat(pieces,axis=1).fillna(0).reset_index().sort_values(group_cols)


def create_validation(persons, households, census, census_households,
                      census_employment, hts_persons, out, sampling_rate):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not 0<sampling_rate<=1:raise ValueError("sampling_rate must be in (0,1]")
    for name,df,fields in [
        ("synthetic persons",persons,["person_id","age","sex","household_id"]),
        ("census population",census,["municipality_id","age_class","sex","weight"]),
        ("HTS persons",hts_persons,["age","sex","person_weight"]),
        ("census households",census_households,["municipality_id","total_households"]),
    ]:
        if not set(fields).issubset(df.columns):
            raise ValueError(f"{name} missing {set(fields)-set(df.columns)}")
    syn=persons.copy(); cen=census.copy(); hts=hts_persons.copy()
    syn["sex"]=_sex(syn.sex); cen["sex"]=_sex(cen.sex); hts["sex"]=_sex(hts.sex)
    if syn.person_id.duplicated().any():raise ValueError("Duplicate synthetic person IDs")
    syn["age"]=pd.to_numeric(syn.age,errors="coerce")
    cen["age_class"]=pd.to_numeric(cen.age_class,errors="coerce")
    hts["age"]=pd.to_numeric(hts.age,errors="coerce")
    syn["weight"]=1/sampling_rate
    cen["weight"]=pd.to_numeric(cen.weight,errors="coerce").fillna(0)
    hts["weight"]=pd.to_numeric(hts.person_weight,errors="coerce").fillna(0)
    if (cen.weight<0).any() or (hts.weight<0).any():raise ValueError("Negative survey/census weight")
    syn["age5"]=(syn.age//5*5).clip(upper=100)
    cen["age5"]=(cen.age_class//5*5).clip(upper=100)
    hts["age5"]=(hts.age//5*5).clip(upper=100)

    age=_percent_table({"Census":cen,"Synthetic":syn,"HTS":hts},"age5",["weight"]*3)
    age.to_csv(out/'age_distribution_5y.csv',index=False)
    _plot_bar(age,'age5',{'Census':'Census','Synthetic':'Synthetic','HTS':'HTS'},
              out/'age_distribution_5y','Share of people (%)','Age distribution: census 2022, synthetic, HTS')
    sex=_percent_table({"Census":cen,"Synthetic":syn,"HTS":hts},"sex",["weight"]*3)
    sex.to_csv(out/'sex_distribution.csv',index=False)
    _plot_bar(sex,'sex',{'Census':'Census','Synthetic':'Synthetic','HTS':'HTS'},
              out/'sex_distribution','Share of people (%)','Sex distribution')
    age_sex=_percent_table({"Census":cen,"Synthetic":syn,"HTS":hts},
                            ["sex","age5"],["weight"]*3)
    age_sex.to_csv(out/'age_sex_distribution.csv',index=False)

    # Census source gives number of households by size, not individual rows.
    hh_fields=["households_1_person","households_2_persons",
               "households_3_persons","households_4_persons","households_5plus_persons"]
    hh_source=census_households.copy()
    sizes=[]
    for i,col in enumerate(hh_fields,1):
        if col not in hh_source:raise ValueError(f"Missing census household column {col}")
        sizes.append({'size':str(i) if i<5 else '5+',
                      'Census':float(pd.to_numeric(hh_source[col],errors='coerce').sum())})
    hh_dist=pd.DataFrame(sizes)
    if households is not None and 'household_size' in households:
        sample_hh=households[['household_id','household_size']].drop_duplicates('household_id')
        hh_sizes=pd.to_numeric(sample_hh.household_size,errors='coerce')
    else:
        hh_sizes=syn.groupby('household_id').size()
    valid=hh_sizes.dropna()
    cat=valid.clip(upper=5).astype(int).astype(str).replace({'5':'5+'})
    hh_dist['Synthetic'] = hh_dist['size'].map(cat.value_counts()).fillna(0).to_numpy()*1/sampling_rate
    for field in ('Census','Synthetic'):
        hh_dist[field+'_pct']=hh_dist[field]/hh_dist[field].sum()*100
    hh_dist.to_csv(out/'household_size_distribution.csv',index=False)
    _plot_bar(hh_dist,'size',{'Census':'Census_pct','Synthetic':'Synthetic_pct'},
              out/'household_sizes','Share of households (%)','Household sizes')

    summary={"census_reference_year":2022,"sampling_rate":sampling_rate,
             "synthetic_persons_sample":int(len(syn)),
             "synthetic_persons_expanded":float(syn.weight.sum()),
             "census_persons":float(cen.weight.sum()),
             "hts_respondent_weight":float(hts.weight.sum()),
             "synthetic_households_sample":int(len(valid)),
             "census_households":float(pd.to_numeric(hh_source.total_households,errors='coerce').sum())}
    summary['population_total_relative_error']=(summary['synthetic_persons_expanded'] /summary['census_persons']-1)
    summary['households_total_relative_error']=(len(valid)/sampling_rate/summary['census_households']-1)
    summary['age_distribution_tv']=float(.5*abs(age.Census-age.Synthetic).sum()/100)
    summary['sex_distribution_tv']=float(.5*abs(sex.Census-sex.Synthetic).sum()/100)
    summary['household_distribution_tv']=float(.5*abs(hh_dist.Census_pct-hh_dist.Synthetic_pct).sum()/100)

    if 'employed' in syn and 'employed' in hts:
        syn['employed_n']=_boolean(syn.employed)
        hts['employed_n']=_boolean(hts.employed)
        # Age-specific *employment among persons*, not workplace job counts.
        syn['age10']=syn.age//10*10; hts['age10']=hts.age//10*10
        def rates(df):
            valid=df.loc[df.employed_n.notna() & df.age10.notna()].copy()
            grouped=valid.groupby(['sex','age10'],as_index=False).apply(
                lambda a:pd.Series({'rate':np.average(a.employed_n,weights=a.weight)}),
                include_groups=False)
            return grouped
        a=rates(syn).rename(columns={'rate':'Synthetic_employment_rate'})
        b=rates(hts).rename(columns={'rate':'HTS_employment_rate'})
        employment=a.merge(b,on=['sex','age10'],how='outer').sort_values(['sex','age10'])
        # Employment census estimates are not directly comparable at five-year
        # age cutoffs (16-19 rather than 15-19); save total separately.
        employment.to_csv(out/'employment_rate_by_sex_age10.csv',index=False)
        summary['synthetic_employment_rate']=float(np.average(
            syn.employed_n.dropna(),weights=syn.loc[syn.employed_n.notna(),'weight']))
        summary['hts_employment_rate']=float(np.average(
            hts.employed_n.dropna(),weights=hts.loc[hts.employed_n.notna(),'weight']))
    if census_employment is not None and 'weight' in census_employment:
        summary['census_employed_residents_estimate']=float(
            pd.to_numeric(census_employment.weight,errors='coerce').sum())
        if 'employed_n' in syn:
            summary['synthetic_employed_residents_expanded']=float(
                (syn.employed_n.fillna(0)*syn.weight).sum())

    # Municipality ID comparison: Synthetic commune_id is a 10-digit section,
    # and its first five characters are the municipality's INE code.
    if 'municipality_id' in syn:
        syn['municipality_id']=syn.municipality_id.astype('string').str[:5]
    elif 'commune_id' in syn:
        syn['municipality_id']=syn.commune_id.astype('string').str[:5]
    elif 'iris_id' in syn:
        syn['municipality_id']=syn.iris_id.astype('string').str[:5]
    else:
        syn['municipality_id']=pd.NA
    cen['municipality_id']=cen.municipality_id.astype('string').str[:5]
    obs=cen.groupby('municipality_id').weight.sum().rename('census_2022')
    modeled=syn.dropna(subset=['municipality_id']).groupby('municipality_id').weight.sum().rename('synthetic_expanded')
    mun=pd.concat([obs,modeled],axis=1).fillna(0).reset_index()
    mun['difference']=mun.synthetic_expanded-mun.census_2022
    mun['relative_error_pct']=np.where(mun.census_2022>0,
                                        mun.difference/mun.census_2022*100,np.nan)
    mun.to_csv(out/'municipality_population_validation.csv',index=False)
    summary['synthetic_municipality_id_coverage']=float(
        syn.municipality_id.isin(obs.index).mean())
    from .gravity_validation import json_save
    json_save(out/'validation_summary.json',summary)
    print(f"Seville validation saved to {out}")
    return summary


def execute(context):
    output_path=Path(context.config('output_path'))
    prefix=context.config('output_prefix')
    persons=pd.read_csv(output_path/f'{prefix}persons.csv',sep=';',low_memory=False)
    hh_path=output_path/f'{prefix}households.csv'
    households=pd.read_csv(hh_path,sep=';',low_memory=False) if hh_path.is_file() else None
    _,hts_persons,_=context.stage('seville.data.hts.entd.reweighted')
    return create_validation(
        persons,households,context.stage('seville.data.census.population'),
        context.stage('seville.data.census.households'),
        context.stage('seville.data.census.employment'),hts_persons,
        Path(context.config('analysis_path'))/'seville_population_validation',
        float(context.config('sampling_rate')))
