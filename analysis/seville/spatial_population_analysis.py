"""Seville spatial Final-vs-Census validation and Final-only destination diagnostics.

Geographies: city districts (41091xx), other metropolitan municipalities,
combined OD macrozones, and all municipalities (41091 as one municipality).
Residential data are aligned by the native ten-digit census section code, not
point-sampled centroids; work/study attendance is mapped using activity points.
Like Asuncion's spatial stage, saves CSV, GeoPackage, PNG and an HTML index.
"""
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from .validation_utils import (
    COLORS, SCOPE_CITY, check_weight, metric_error, numeric, prepare_persons,
    residence_macrozone, section_ids, ids, write_index, write_json,
)

LOG = logging.getLogger(__name__)
BASE_METRICS = ("population", "households", "employed_residents", "licensed_residents")


def configure(context):
    for stage in (
        "synthesis.output", "seville.gravity.od_zones",
        "seville.data.spatial.iris",
        "seville.data.census.population", "seville.data.census.households",
        "seville.data.census.employment", "seville.data.census.licenses",
    ):
        context.stage(stage)
    context.config("output_path")
    context.config("output_prefix")
    context.config("analysis_path")
    context.config("sampling_rate",1.)
    context.config("gravity_calibration_municipality_id","41091")
    context.config("spatial_validation_min_people",10)


def _validate_zones(zones):
    if not isinstance(zones,gpd.GeoDataFrame) or zones.crs is None:
        raise ValueError("Expected OD-zone GeoDataFrame with CRS")
    if not {"macrozone_id","geometry"}.issubset(zones):
        raise ValueError("Missing macrozone_id or geometry")
    z=zones[["macrozone_id","geometry"]].copy()
    z["macrozone_id"]=ids(z.macrozone_id)
    if z.macrozone_id.isna().any() or z.macrozone_id.duplicated().any():
        raise ValueError("Duplicate/missing OD-zone codes")
    if not z.geometry.notna().all() or not z.geometry.is_valid.all():
        raise ValueError("Missing/invalid OD-zone geometry")
    return z.to_crs("EPSG:25830")


def _reference_sum(df,column,metric,city_code="41091"):
    if df is None or df.empty:
        return pd.Series(dtype=float,name=metric)
    x=df.copy()
    x["macrozone_id"]=residence_macrozone(x,city_code)
    x["_value"]=check_weight(x,column,metric)
    if not x.macrozone_id.notna().any():
        raise ValueError(f"{metric}: cannot map reference census sections to macrozones")
    return x.groupby("macrozone_id")._value.sum().rename(metric)


def assign_final_residences(final, homes, zones, city_code="41091"):
    """Assign exported persons to gravity zones via their household's home POINT.

    Exported final persons have no section/municipality ID. Keep all persons,
    including those whose household has no record in exported homes.gpkg, so
    non-spatial diagnostics can still use them. Unmapped residences receive NA
    zone IDs and are excluded ONLY from geographic validation. A home that is
    present but lies outside all OD polygons remains an error.
    IPU is not used in the Final-vs-Census validation.
    """
    p=final.copy()
    existing=residence_macrozone(p,city_code)
    if homes is None:
        if existing.isna().any():
            raise ValueError("Final persons lack census-section/municipality codes. "
                             "Pass <output_prefix>homes.gpkg and the OD-zone polygons "
                             "to assign household residences; cannot validate final population without them.")
        p["macrozone_id_validation"]=existing
        p["municipality_id_validation"]=existing.str[:5]
        return p,{"mapping":"existing_sections","persons":len(p),"households":int(p.household_id.nunique()),
                  "households_matched":int(p.household_id.nunique()), "households_missing_home":0,
                  "persons_missing_home":0, "missing_home_household_ids":[],
                  "households_missing_zone":0,"persons_missing_zone":0}
    if not isinstance(homes,gpd.GeoDataFrame) or homes.crs is None:
        raise ValueError("Exported homes must be a GeoDataFrame with a CRS")
    if not {"household_id","geometry"}.issubset(homes):
        raise ValueError("Exported homes must have household_id and geometry")
    if "household_id" not in p or p.household_id.isna().any():
        raise ValueError("Final persons need non-null household_id")
    h=homes[["household_id","geometry"]].copy()
    h["_hh_key"]=ids(h.household_id)
    p["_hh_key"]=ids(p.household_id)
    if h._hh_key.isna().any() or h._hh_key.duplicated().any():
        raise ValueError("Homes.gpkg must contain exactly one row per household_id")
    if p._hh_key.isna().any():
        raise ValueError("Final persons have invalid household_id")
    needed=set(p._hh_key.unique())
    # Both a missing GeoPackage row and a row with null/empty POINT geometry
    # mean that this household cannot be spatially located.
    unusable_geometry=h.geometry.isna() | h.geometry.is_empty
    empty_geometry_ids=set(h.loc[unusable_geometry,"_hh_key"]).intersection(needed)
    h=h.loc[~unusable_geometry].copy()
    missing=needed-set(h._hh_key)
    if missing:
        LOG.warning("%s final households have no usable exported home point; "
                    "excluding them from home-location-dependent validation only", len(missing))
    h=h.loc[h._hh_key.isin(needed)].copy().reset_index(drop=True)
    h["_source_row"]=np.arange(len(h))
    mapped,quality=_point_to_macrozone(h,zones,city_code)
    h=h.merge(mapped,on="_source_row",how="left",validate="one_to_one")
    if h.macrozone_id.isna().any():
        raise ValueError(f"{h.macrozone_id.isna().sum()} final homes could not be "
                         "assigned to metro OD polygons; inspect CRS, output extent and polygons")
    lookup=h[["_hh_key","macrozone_id"]]
    p=p.merge(lookup,on="_hh_key",how="left",validate="many_to_one",sort=False)
    # Unmapped rows must correspond precisely to households without an export
    # record; anything else is an unexpected join/spatial assignment problem.
    if set(p.loc[p.macrozone_id.isna(), "_hh_key"]) != missing:
        raise ValueError("Unexpected final persons without a home macrozone")
    missing_person_count = int(p._hh_key.isin(missing).sum())
    p["macrozone_id_validation"]=p.macrozone_id.astype("string")
    p["municipality_id_validation"]=p.macrozone_id_validation.str[:5]
    p=p.drop(columns=["_hh_key","macrozone_id"])
    return p,{"mapping":"exported_homes_geometry","persons":len(p),
              "households":int(len(needed)),"households_matched":int(len(h)),
              "households_missing_home":int(len(missing)),
              "persons_missing_home":missing_person_count,
              "missing_home_household_ids":sorted(missing),
              "households_with_empty_home_geometry":int(len(empty_geometry_ids)),
              "households_missing_zone":int(len(missing)),
              "persons_missing_zone":missing_person_count,
              **quality}


def _residential_counts(people,city_code):
    df=people.copy()
    for field in ("employed_n","license_n"):
        if df[field].notna().sum()==0:
            raise ValueError(f"Synthetic person field {field} is completely missing; cannot compare against census")
    df["macrozone_id"]=(ids(df["macrozone_id_validation"]) if
                        "macrozone_id_validation" in df else residence_macrozone(df,city_code))
    df["_employment"]=df.employed_n * df._w
    df["_license"]=df.license_n * df._w
    df["_education"]=df.studies_n * df._w
    df["_female"] = df.sex_norm.eq("female").astype(float) * df._w
    df["_under15"] = df.age.lt(15).astype(float) * df._w
    df["_age65plus"] = df.age.ge(65).astype(float) * df._w
    count=df.groupby("macrozone_id").agg(
        population=("_w","sum"), employed_residents=("_employment","sum"),
        licensed_residents=("_license","sum"), students_resident=("_education","sum"),
        female_residents=("_female","sum"), under15_residents=("_under15","sum"),
        age65plus_residents=("_age65plus","sum"))
    eligible=df.assign(_eligible=df._w*df.age.ge(16).astype(float),
                       _adult=df._w*df.age.ge(18).astype(float)).groupby("macrozone_id")[["_eligible","_adult"]].sum()
    count["employment_eligible_residents"] = eligible._eligible
    count["adult_residents"] = eligible._adult
    for target, source in (("employed_residents","employed_n"),
                           ("licensed_residents","license_n"),("students_resident","studies_n")):
        known=df.groupby("macrozone_id")[source].count()
        count.loc[known.eq(0),target]=np.nan
    if "household_id" not in df:
        raise ValueError("Final people lack household_id")
    hh=df.dropna(subset=["household_id","macrozone_id"])
    n=hh.groupby("household_id").macrozone_id.nunique()
    mismatched=int(n.gt(1).sum())
    # People in each sampled household must live in same macrozone. Do not
    # silently allocate one household to multiple residential zones.
    if mismatched:
        raise ValueError(f"{mismatched} households have members in multiple macrozones")
    home=hh.drop_duplicates("household_id").groupby("macrozone_id")._w.sum()
    # One weight per household; all sampled persons have the same sampling rate.
    count["households"]=home
    sizes=hh.groupby("household_id").agg(macrozone_id=("macrozone_id","first"),
                                          sample_size=("household_id","size"),_w=("_w","first"))
    sizes["capped_size"]=sizes.sample_size.clip(upper=5)
    by_size=sizes.groupby(["macrozone_id","capped_size"])._w.sum().unstack(fill_value=0)
    for n in range(1,6):
        count[f"households_size_{n if n<5 else '5plus'}"]=by_size[n] if n in by_size else 0.
    return count, {"persons_missing_section":int(df.macrozone_id.isna().sum()),
                   "households_missing_section":int(df.loc[df.macrozone_id.isna(),"household_id"].nunique()),
                   "persons_total":len(df),"households_total":int(df.household_id.nunique())}


def _point_to_macrozone(points,zone_polygons,city_code="41091"):
    """Spatially join activity POINTS to nonoverlapping city-district/municipality zones.

    Boundary-only points fall back to intersects. Multiple candidate polygons
    are counted and resolved deterministically by lexicographic zone ID.
    """
    if points.crs is None:
        raise ValueError("Activities / locations need declared CRS")
    x=points.copy()
    x["_source_row"]=np.arange(len(x))
    x=x.loc[x.geometry.notna() & ~x.geometry.is_empty].copy()
    if x.empty:
        return pd.DataFrame(columns=["_source_row","macrozone_id"]),{"valid_points":0,"unmatched_points":0,"boundary_fallback":0,"ambiguous_points":0}
    x=x.to_crs(zone_polygons.crs)
    polys=zone_polygons[["macrozone_id","geometry"]]
    found=gpd.sjoin(x[["_source_row","geometry"]],polys,how="left",predicate="within")
    if found.index.duplicated().any() or found._source_row.duplicated().any():
        # Zone overlaps are an input error, not a property to conceal.
        raise ValueError("A point lies within more than one macrozone: overlapping zoning polygons")
    missed=found.loc[found.macrozone_id.isna(),["_source_row","geometry"]].copy()
    fallback=pd.DataFrame(columns=["_source_row","macrozone_id"])
    ambiguity=0
    if not missed.empty:
        adjacent=gpd.sjoin(gpd.GeoDataFrame(missed,geometry="geometry",crs=x.crs),
                           polys,how="left",predicate="intersects")
        ambiguity=int(adjacent.dropna(subset=["macrozone_id"])._source_row.duplicated().sum())
        if ambiguity:
            LOG.warning("Ambiguous boundary points in %s: %s; assigning lexicographically first",len(x),ambiguity)
        fallback=(adjacent.sort_values(["_source_row","macrozone_id"])
                  .drop_duplicates("_source_row")[["_source_row","macrozone_id"]])
    result=found.loc[found.macrozone_id.notna(),["_source_row","macrozone_id"]]
    if not fallback.empty:
        result=pd.concat([result,fallback.dropna(subset=["macrozone_id"])],ignore_index=True)
    result=result.drop_duplicates("_source_row")
    if result._source_row.duplicated().any():
        raise ValueError("Multiple zone assignment after spatial resolution")
    return result,{"valid_points":int(len(x)),"unmatched_points":int(result.macrozone_id.isna().sum()+len(x)-len(result)),
                   "boundary_fallback":int(len(fallback)),"ambiguous_points":ambiguity}


def _activity_counts(activities,zones,purpose,rate):
    required={"person_id","purpose","geometry"}
    if not required.issubset(activities):
        raise ValueError(f"Activity GeoDataFrame missing {required-set(activities.columns)}")
    x=activities.loc[activities.purpose.astype("string").str.strip().str.lower().eq(purpose)].copy()
    x=x.loc[x.person_id.notna()].drop_duplicates("person_id").reset_index(drop=True)
    x["_source_row"]=np.arange(len(x))
    mapped,quality=_point_to_macrozone(x,zones)
    x=x.merge(mapped,on="_source_row",how="left",validate="one_to_one")
    counts=x.groupby("macrozone_id").person_id.nunique() / rate
    return counts,quality,x


def _facilities(education,city_code="41091"):
    if "students" not in education or "commune_id" not in education:
        raise ValueError("Seville education.merged requires students and commune_id")
    e=education.copy()
    e["macrozone_id"]=residence_macrozone(e,city_code)
    e["students_proxy"] = numeric(e.students).clip(lower=0)
    if e.students_proxy.isna().any():
        LOG.warning("Education facilities contain unknown student counts; treating missing as 0 (marked in QA)")
    e["students_proxy"]=e.students_proxy.fillna(0)
    e["facility_count"] = 1.
    e["student_university_proxy"] = e.students_proxy.where(e.education_type.astype("string").eq("university"), 0)
    e["student_school_proxy"] = e.students_proxy.where(~e.education_type.astype("string").eq("university"), 0)
    result=e.groupby("macrozone_id").agg(
        student_places_proxy=("students_proxy","sum"), education_facilities=("facility_count","sum"),
        university_students_proxy=("student_university_proxy","sum"),
        school_students_proxy=("student_school_proxy","sum"))
    return result, {"facilities_total":len(e),"facilities_without_zone":int(e.macrozone_id.isna().sum()),
                    "students_proxy_missing":int(numeric(e.students).isna().sum())}


def _build_reference(pop,hh,emp,licenses,jobs,facilities,city_code):
    frame=pd.concat([
        _reference_sum(pop,"weight","census_population",city_code),
        _reference_sum(hh,"total_households","census_households",city_code),
        _reference_sum(emp,"weight","census_employed_residents",city_code),
        _reference_sum(licenses.loc[numeric(licenses.age_class).ge(18)],"weight","census_licensed_residents",city_code),
        _reference_sum(pop.loc[pop.sex.astype("string").str.lower().eq("female")],"weight","census_female_residents",city_code),
        _reference_sum(pop.loc[numeric(pop.age_class).lt(15)],"weight","census_under15_residents",city_code),
        _reference_sum(pop.loc[numeric(pop.age_class).ge(65)],"weight","census_age65plus_residents",city_code),
        *[_reference_sum(hh, f, "census_households_size_"+label, city_code) for f,label in (
            ("households_1_person","1"),("households_2_persons","2"),
            ("households_3_persons","3"),("households_4_persons","4"),
            ("households_5plus_persons","5plus"))],
    ],axis=1)
    # 16+ and 18+ census denominators: fractional split of the 15–19 bin.
    age=numeric(pop.age_class)
    w=check_weight(pop,"weight","census population")
    frac_adult=age.ge(20).astype(float)+.4*age.eq(15).astype(float)
    frac_eligible=age.ge(20).astype(float)+.8*age.eq(15).astype(float)
    with_adults=pop.assign(_adult=w*frac_adult,_eligible=w*frac_eligible)
    frame=frame.join(_reference_sum(with_adults,"_adult","census_adult_residents",city_code),how="outer")
    frame=frame.join(_reference_sum(with_adults,"_eligible","census_employment_eligible_residents",city_code),how="outer")
    if jobs is not None:
        frame=frame.join(_reference_sum(jobs,"weight","job_grid_workplaces_2024",city_code),how="outer")
    if facilities is not None:
        fac,qa=_facilities(facilities,city_code)
        frame=frame.join(fac,how="outer")
    else:qa={}
    return frame,qa


def _select_geography(macro,scope,city_code):
    if scope=="city_districts":
        return macro.loc[macro.macrozone_id.str.startswith(city_code)].copy()
    if scope=="metro_macrozones":
        return macro.copy()
    if scope=="municipalities":
        d=macro.copy()
        d["municipality_id"]=d.macrozone_id.str[:5]
        d=d.drop(columns=["macrozone_id"]).dissolve(by="municipality_id",aggfunc="sum").reset_index().rename(columns={"municipality_id":"macrozone_id"})
        if "_scale" in d:
            d["_scale"]=float(macro["_scale"].iloc[0])
        return d
    raise ValueError("Unknown spatial scope")


def _derive(g):
    x=g.copy()
    for source in ("final",):
        for m in BASE_METRICS:
            a=f"{source}_{m}";b=f"census_{m}"
            if a in x and b in x:
                x[f"{source}_{m}_difference"]=x[a]-x[b]
                x[f"{source}_{m}_error_pct"]=metric_error(x[b],x[a])
        # These are population-wide prevalence shares, not adult-only rates.
        x[f"{source}_employment_rate_pct"] = (100*x[f"{source}_employed_residents"] /
                                               x[f"{source}_population"].replace(0,np.nan))
        x[f"{source}_licence_rate_pct"] = (100*x[f"{source}_licensed_residents"] /
                                            x[f"{source}_population"].replace(0,np.nan))
    x["census_employment_rate_pct"] = 100*x.census_employed_residents/x.census_population.replace(0,np.nan)
    x["census_licence_rate_pct"] = 100*x.census_licensed_residents/x.census_population.replace(0,np.nan)
    for source in ("final",):
        for m in ("employment_rate_pct","licence_rate_pct"):
            src=f"{source}_{m}";cen=f"census_{m}"
            if src in x:x[f"{source}_{m}_gap_pp"]=x[src]-x[cen]
        for name,col in (("female","female_residents"),("under15","under15_residents"),
                         ("age65plus","age65plus_residents"),("one_person_households","households_size_1")):
            denominator = "households" if name=="one_person_households" else "population"
            a=f"{source}_{col}";b=f"census_{col}"
            if a in x and b in x:
                xx=100*x[a]/x[f"{source}_{denominator}"].replace(0,np.nan)
                yy=100*x[b]/x[f"census_{denominator}"].replace(0,np.nan)
                x[f"{source}_{name}_share_gap_pp"]=xx-yy
    # No region-wide shares: compare absolute activity counts, and rates using
    # each zone's own residential denominator. These are NOT stock-equivalent
    # measures (model-day attendance vs workplace jobs/student places).
    for metric in ("work", "education"):
        col=f"final_{metric}_attendees"
        x[f"{col}_per_1000_residents"] = 1000*x[col]/x.final_population.replace(0,np.nan)
    for metric,ref in (("work","job_grid_workplaces_2024"),
                       ("education","student_places_proxy")):
        col=f"final_{metric}_attendees"
        if ref in x:
            x[f"{metric}_attendees_to_reference_pct"]=100*x[col]/x[ref].replace(0,np.nan)
    x["area_km2"]=x.to_crs("EPSG:25830").geometry.area/1e6
    return x


def _save(fig,out,manifest,slug,title,note=""):
    # Shared colorbars on paired maps are handled without tight_layout by matplotlib.
    if len(fig.axes) < 3:
        fig.tight_layout()
    fig.savefig(out/(slug+".png"),dpi=170,bbox_inches="tight")
    plt.close(fig)
    manifest.append({"file":slug+".png","title":title,"note":note})


def _map_panel(g, col, ax, diverging=False, label=""):
    """Right-hand polygon map with one zone value; never normalize to metro."""
    values=pd.to_numeric(g[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
    valid=values.dropna()
    if valid.empty:
        ax.set_axis_off()
        ax.set_title("No mapped values")
        return
    if diverging:
        cap=max(float(np.abs(valid).max()),1e-6)
        cmap="RdBu_r";vmin=-cap;vmax=cap
    else:
        cmap="YlGnBu";vmin=0;vmax=max(float(valid.max()),1e-6)
        if (valid < 0).any():vmin=float(valid.min())
    g.plot(column=col,ax=ax,cmap=cmap,vmin=vmin,vmax=vmax,
           edgecolor="#555555",linewidth=.45,
           missing_kwds={"color":"#dedede","edgecolor":"#888888"})
    ax.set_axis_off();ax.set_title("Spatial distribution" + (" · " + label if label else ""),fontsize=11)
    im=plt.cm.ScalarMappable(norm=plt.Normalize(vmin,vmax),cmap=cmap)
    ax.figure.colorbar(im,ax=ax,shrink=.68,pad=.015,label=label)


def _bar_axes(fig,g):
    ax=fig.add_subplot(1,2,1)
    codes=g.macrozone_id.astype(str).to_numpy()
    tick_size=7 if len(codes)>20 else 9
    ax.set_xticks(np.arange(len(codes)),codes,rotation=80 if len(codes)>10 else 45,ha="right",fontsize=tick_size)
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    return ax


def _map(g,col,out,manifest,slug,title,note="",diverging=False):
    """Every geographic choropleth has its matching numeric zone bars on left."""
    if col not in g or not g[col].notna().any():return
    data=g.sort_values("macrozone_id").reset_index(drop=True)
    n=len(data)
    fig=plt.figure(figsize=(max(13,min(19,12+n*.13)),8.0))
    ax=_bar_axes(fig,data)
    values=pd.to_numeric(data[col],errors="coerce")
    ax.bar(np.arange(n),values,color=COLORS["Final"],width=.75)
    ax.axhline(0,color="#888888",linewidth=.65)
    ax.set_title("Values by zone",fontsize=11);ax.set_ylabel(col.replace("_"," "))
    map_ax=fig.add_subplot(1,2,2)
    _map_panel(data,col,map_ax,diverging=diverging,label="Percentage points" if col.endswith("_pp") else "Percent" if col.endswith("_pct") else "Value")
    fig.suptitle(title,fontsize=14)
    fig.subplots_adjust(left=.06,right=.96,bottom=.23,top=.88,wspace=.18)
    _save(fig,out,manifest,slug,title,note)


def _paired(g,one,two,out,manifest,slug,title,note="",map_col=None,map_label="",diverging=False):
    """Left: comparable zone counts/rates (reference and model); right: map.

    By default the map shows modeled values; provide a precomputed gap column
    for same-definition census comparisons. Never plot two maps alone.
    """
    if one not in g or two not in g:return
    data=g.sort_values("macrozone_id").reset_index(drop=True)
    if not (data[one].notna() & data[two].notna()).any():return
    if map_col is None:map_col=two
    if map_col not in data:return
    n=len(data)
    fig=plt.figure(figsize=(max(13,min(19,12+n*.13)),8.0))
    ax=_bar_axes(fig,data)
    locations=np.arange(n)
    reference_label=("Census" if one.startswith("census_") else
                     "Workplace job grid" if one=="job_grid_workplaces_2024" else
                     "Education facility proxy" if one=="student_places_proxy" else "Reference")
    model_label="Final"
    ax.bar(locations-.2,data[one],width=.4,label=reference_label,color=COLORS["Census"])
    ax.bar(locations+.2,data[two],width=.4,label=model_label,color=COLORS["Final"])
    ax.set_title("Values by zone",fontsize=11);ax.legend(loc="best",fontsize=9)
    ax.set_ylabel("Rate (%)" if one.endswith("_rate_pct") else "Count")
    map_ax=fig.add_subplot(1,2,2)
    _map_panel(data,map_col,map_ax,diverging=diverging,label=map_label)
    fig.suptitle(title,fontsize=14)
    fig.subplots_adjust(left=.06,right=.96,bottom=.23,top=.88,wspace=.18)
    _save(fig,out,manifest,slug,title,note)


def _scatter(g,ref,syn,out,manifest,slug,title):
    if ref not in g or syn not in g:return
    s=g[[ref,syn,"macrozone_id"]].replace([np.inf,-np.inf],np.nan).dropna()
    if s.empty:return
    fig,ax=plt.subplots(figsize=(7,7))
    ax.scatter(s[ref],s[syn],s=35,alpha=.7,color=COLORS["Final"])
    upper=max(s[ref].max(),s[syn].max(),1)
    ax.plot([0,upper],[0,upper],linestyle="--",color="#555555")
    ax.set(xlabel="Census",ylabel="Synthetic expanded",title=title)
    ax.grid(alpha=.2)
    for row in s.loc[(s[syn]-s[ref]).abs().nlargest(min(5,len(s))).index].itertuples():
        ax.annotate(str(row.macrozone_id),(getattr(row,ref),getattr(row,syn)),fontsize=8)
    _save(fig,out,manifest,slug,title)


def _render_scope(g,out,scope,min_people):
    out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    g.drop(columns="geometry").to_csv(out/"zone_comparison.csv",index=False)
    for source in ("final",):
        for metric in BASE_METRICS:
            cen=f"census_{metric}";syn=f"{source}_{metric}"
            if cen not in g or syn not in g:continue
            prefix=f"{source}_{metric}"
            _paired(g,cen,syn,out,manifest,prefix+"_counts",
                    "Census vs "+source.upper()+" · "+metric.replace("_"," "),
                    map_col=prefix+"_error_pct",map_label="Model − census (% of census)",diverging=True)
        for metric,label,denominator in (
            ("employment_rate_pct","Employed residents / all residents (%)","population"),
            ("licence_rate_pct","Licensed residents / all residents (%)","population"),
        ):
            cen=f"census_{metric}";syn=f"{source}_{metric}"
            gap=f"{source}_{metric}_gap_pp"
            if not {cen,syn,gap}.issubset(g):continue
            geo=g.copy()
            geo.loc[geo[f"{source}_{denominator}"] < min_people*geo["_scale"], [syn,gap]]=np.nan
            _paired(geo,cen,syn,out,manifest,f"{source}_{metric}_comparison",label+" · "+source.upper()+" vs census",
                    note="Rates are within each zone, not a percentage of metropolitan population. Low-count sampled zones masked.",
                    map_col=gap,map_label="Model − census (percentage points)",diverging=True)
        for name,label in (("female","Female residents"),
                           ("under15","Residents aged under 15"),
                           ("age65plus","Residents aged 65+"),
                           ("one_person_households","One-person households")):
            col=f"{source}_{name}_share_gap_pp"
            if col in g:
                _map(g,col,out,manifest,col,label+" · "+source.upper()+" − census (pp)",
                     "Within-zone percentages: female/age groups divided by residents; one-person homes divided by households.",True)
    # Census has no same-definition daily workplace/study-attendance measure.
    # These maps are Final-only descriptive summaries, never labeled validation.
    for metric in ("work", "education"):
        for col,label in (
            (f"final_{metric}_attendees", f"Final {metric} activity attendees, expanded"),
            (f"final_{metric}_attendees_density", f"Final {metric} attendees per km²"),
        ):
            if col in g:
                _map(g,col,out,manifest,col,label,
                     "Final model-day attendance only; Census has no directly comparable destination measure.")
    pd.DataFrame(manifest).to_csv(out/"plot_manifest.csv",index=False)
    write_index(out,"Seville spatial validation — "+scope,manifest,[
        "Every zone map has a numeric bar chart on its LEFT. Census-backed count maps show (model−census)/census ×100; rate maps show percentage-point gaps.",
        "All rates are calculated within each district or municipality, never divided by a metropolitan total.",
        "Only Final is compared with Census; no intermediate IPU or HTS benchmark is used.",
        "Employment share: employed residents / ALL residents; licence share: licensed residents / ALL residents (including children).",
        "Work and education attendance maps describe Final output only; no Census destination benchmark is claimed.",
        "Census employment/licence/household measures are partly estimated or spatially allocated to sections.",
        "Synthetic counts are expanded by 1/sampling_rate; shares with too few sampled residents are masked.",
        "Final households lacking an exported home point are excluded from home-location-dependent maps. See home_location_exclusions.png and home_location_affected_plots.csv in the report root.",
    ])
    return manifest



def _validate_census_sections(sections, census_population):
    """Use native 10-digit census sections, not split/estimated OD districts."""
    if not isinstance(sections, gpd.GeoDataFrame) or sections.crs is None:
        raise ValueError("seville.data.spatial.iris must provide georeferenced census polygons")
    if not {"commune_id", "geometry"}.issubset(sections):
        raise ValueError("seville.data.spatial.iris needs commune_id and geometry")
    result=sections[["commune_id","geometry"]].copy()
    result["macrozone_id"]=ids(result.commune_id)
    if result.macrozone_id.isna().any() or not result.macrozone_id.str.fullmatch(r"\d{10}").all():
        raise ValueError("Census-section polygons need valid 10-digit commune_id")
    if result.macrozone_id.duplicated().any():
        raise ValueError("Duplicate census-section polygon IDs")
    if result.geometry.isna().any() or result.geometry.is_empty.any() or not result.geometry.is_valid.all():
        raise ValueError("Invalid census-section polygon geometries")
    expected=set(section_ids(census_population).dropna())
    result=result.loc[result.macrozone_id.isin(expected),["macrozone_id","geometry"]].copy()
    if result.empty:
        raise ValueError("No census-section polygons match census population")
    missing=sorted(expected-set(result.macrozone_id))
    if missing:
        LOG.warning("Census geometries missing for %s official census sections; see section QA",len(missing))
    return result.to_crs("EPSG:25830"),missing


def _section_reference_sum(df,column,metric):
    if df is None or df.empty:
        return pd.Series(dtype=float,name=metric)
    sid=section_ids(df)
    weight=check_weight(df,column,metric)
    if sid.notna().sum()==0:
        raise ValueError(f"{metric}: census has no valid census_section_id")
    return pd.DataFrame({"section_id":sid,"value":weight}).groupby("section_id").value.sum().rename(metric)


def _section_residence_counts(people, final_homes, sections, city_code):
    """Return final person counts by true section and a separate mapping QA.

    homes.gpkg is authoritative for exported Final; its absent household IDs
    remain unassigned. A point that misses all section polygons is counted in
    the QA and omitted from section-only analysis, not fabricated as zone 0.
    """
    p=people.copy()
    if final_homes is None:
        p["macrozone_id_validation"]=section_ids(p)
        mapping={"method":"existing_person_section","persons_unmapped_to_section":int(p.macrozone_id_validation.isna().sum())}
    else:
        if "household_id" not in p or not {"household_id","geometry"}.issubset(final_homes):
            raise ValueError("Final section validation needs household_id in people and homes")
        homes=final_homes[["household_id","geometry"]].copy()
        if not isinstance(final_homes,gpd.GeoDataFrame) or homes.crs is None:
            raise ValueError("Final homes must have a known CRS")
        homes["household_key"]=ids(homes.household_id)
        if homes.household_key.isna().any() or homes.household_key.duplicated().any():
            raise ValueError("Invalid or repeated home household identifiers")
        homes=homes.loc[homes.geometry.notna() & ~homes.geometry.is_empty].copy().reset_index(drop=True)
        matched,quality=_point_to_macrozone(homes,sections,city_code)
        homes["_source_row"]=np.arange(len(homes))
        homes=homes.merge(matched,on="_source_row",how="left",validate="one_to_one")
        lookup=homes.set_index("household_key").macrozone_id
        p["macrozone_id_validation"]=ids(p.household_id).map(lookup).astype("string")
        mapping={**quality,"method":"exported_homes_to_census_sections",
                 "persons_unmapped_to_section":int(p.macrozone_id_validation.isna().sum()),
                 "households_unmapped_to_section":int(p.loc[p.macrozone_id_validation.isna(),"household_id"].nunique())}
    frame,quality=_residential_counts(p,city_code)
    return frame,{**quality,**mapping}


def _section_comparison_plot(g, census_col, model_col, error_col, out, manifest,
                             slug, title, top_n=12):
    """Readable worst-discrepancy bars next to an ALL-section choropleth.

    With hundreds of sections, a single per-section barplot becomes unusable.
    Show only the largest absolute Census-vs-model count gaps on the left, but
    map every section and keep complete numbers in zone_comparison.csv/GPKG.
    """
    if not {census_col, model_col, error_col}.issubset(g):
        return
    valid = g.loc[g[census_col].notna() & g[model_col].notna()].copy()
    if valid.empty:
        return
    valid["absolute_difference"] = (valid[model_col] - valid[census_col]).abs()
    count = min(top_n, len(valid))
    bars = valid.nlargest(count, "absolute_difference").sort_values("absolute_difference")
    fig = plt.figure(figsize=(19, max(7.5, 0.54 * count + 2.2)))
    ax = fig.add_subplot(1, 2, 1)
    y = np.arange(len(bars))
    census_vals = pd.to_numeric(bars[census_col], errors="coerce")
    model_vals = pd.to_numeric(bars[model_col], errors="coerce")
    ax.barh(y - .19, census_vals, height=.36, label="Census", color=COLORS["Census"])
    ax.barh(y + .19, model_vals, height=.36,
            label="Final", color=COLORS["Final"])
    ax.set_yticks(y, bars.macrozone_id.astype(str).to_list(), fontsize=9)
    # Direct numbers stay visible even when the difference is too small to see.
    max_count = max(float(census_vals.max()), float(model_vals.max()), 1)
    padding = max_count * 0.014
    for yi, census_value, model_value in zip(y, census_vals, model_vals):
        ax.text(census_value + padding, yi - .19, f"{census_value:,.0f}",
                ha="left", va="center", fontsize=8, color="#263238")
        ax.text(model_value + padding, yi + .19, f"{model_value:,.0f}",
                ha="left", va="center", fontsize=8, color="#263238")
    ax.set_xlim(0, max_count * 1.23)
    ax.set(xlabel="People / households (expanded)",
           title=f"{count} largest count gaps · {len(valid)} sections with data")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    map_ax = fig.add_subplot(1, 2, 2)
    _map_panel(g, error_col, map_ax, diverging=True,
               label="Model − Census (% of Census)")
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=.10, right=.96, bottom=.10, top=.88, wspace=.24)
    _save(fig, out, manifest, slug, title,
          f"Left: {count} sections with the largest absolute count differences "
          f"out of {len(valid)} comparable sections; values labeled at bar ends. "
          f"Right: all {len(g)} census sections (missing references are gray). "
          "Full section counts and errors: zone_comparison.csv or GeoPackage.")

def _section_overview(g, census_col, model_col, out, manifest, metric, sampling_rate):
    """All-section agreement diagnostics: scatter, residuals, relative error and metrics.

    Does not claim that a 1% sample can reproduce 10-digit census sections.
    No sampled section is dropped merely because the modeled count is zero.
    """
    frame=g[["macrozone_id",census_col,model_col]].copy()
    frame[census_col]=pd.to_numeric(frame[census_col],errors="coerce")
    frame[model_col]=pd.to_numeric(frame[model_col],errors="coerce")
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=[census_col,model_col])
    if frame.empty: return None
    ref=frame[census_col].to_numpy(dtype=float)
    syn=frame[model_col].to_numpy(dtype=float)
    residual=syn-ref
    valid=ref>0
    pct=np.full(len(ref),np.nan)
    pct[valid]=100*residual[valid]/ref[valid]
    pearson=float(np.corrcoef(ref,syn)[0,1]) if len(ref)>1 and np.std(ref)>0 and np.std(syn)>0 else np.nan
    sst=((ref-ref.mean())**2).sum()
    r2=float(1-(residual**2).sum()/sst) if sst>0 else np.nan
    factor=1/sampling_rate
    fig,axes=plt.subplots(2,2,figsize=(13.8,10.6))
    ax=axes[0,0]
    ax.scatter(ref,syn,color=COLORS["Final"],s=18,alpha=.4,edgecolors="none",rasterized=True)
    extent=max(float(ref.max()),float(syn.max()),1)
    ax.plot([0,extent],[0,extent],"--",color=COLORS["Census"],lw=1.6,label="Perfect agreement")
    ax.set(xlabel="Census count",ylabel="Final (expanded)",title=f"All {len(ref)} comparable census sections")
    ax.legend(loc="upper left",frameon=False);ax.grid(alpha=.18)
    ax=axes[0,1]
    ax.hist(residual,bins=min(40,max(8,int(np.sqrt(len(ref))*1.5))),color=COLORS["Final"],alpha=.9)
    ax.axvline(0,color=COLORS["Census"],ls="--",lw=1.7)
    ax.set(xlabel="Final − Census (count)",ylabel="Number of sections",title="Count error across all sections")
    ax.grid(axis="y",alpha=.2)
    ax=axes[1,0]
    finite=np.abs(pct[np.isfinite(pct)])
    if len(finite):
        ax.hist(finite,bins=min(40,max(8,int(np.sqrt(len(finite))*1.5))),color=COLORS["Final"],alpha=.9)
        ax.axvline(np.median(finite),color=COLORS["Census"],ls="--",lw=1.8,label=f"Median = {np.median(finite):.1f}%")
        ax.legend(frameon=False)
    ax.set(xlabel="Absolute error / Census (%)",ylabel="Number of sections",title="Absolute relative error · positive Census only")
    ax.grid(axis="y",alpha=.2)
    ax=axes[1,1];ax.axis("off")
    wape=100*np.abs(residual).sum()/ref.sum() if ref.sum()>0 else np.nan
    stats={"scope":"census_sections", "source":"final","metric":metric,
        "n_zones_with_reference":int(len(ref)),"census_total":float(ref.sum()),
        "synthetic_total":float(syn.sum()),
        "relative_total_error_pct":float(100*residual.sum()/ref.sum()) if ref.sum()>0 else np.nan,
        "zone_mae":float(np.abs(residual).mean()),"zone_rmse":float(np.sqrt(np.mean(residual**2))),
        "zone_wape_pct":float(wape),"pearson_r":pearson,"r_squared_identity":r2,
        "median_absolute_percentage_error":float(np.median(finite)) if len(finite) else np.nan,
        "p90_absolute_percentage_error":float(np.percentile(finite,90)) if len(finite) else np.nan,
        "zero_final_sections":int((syn==0).sum()),
        "sections_with_positive_census":int(valid.sum()),
        "sections_without_sample":int((syn==0).sum()),
        "sampling_expansion_factor":factor}
    summary=(f"All sections: {len(ref):,}\n"
        f"Pearson r: {pearson:.3f}\n"
        f"R² against identity: {r2:.3f}\n"
        f"MAE: {stats['zone_mae']:,.1f}\n"
        f"RMSE: {stats['zone_rmse']:,.1f}\n"
        f"WAPE: {wape:.1f}%\n"
        f"Median abs. % error: {stats['median_absolute_percentage_error']:.1f}%\n"
        f"90th pct. abs. error: {stats['p90_absolute_percentage_error']:.1f}%\n"
        f"Zero Final counts: {stats['zero_final_sections']:,}\n"
        f"Sampling expansion: ×{factor:g}")
    ax.text(.07,.94,summary,ha="left",va="top",family="monospace",fontsize=12,
            transform=ax.transAxes,linespacing=1.8)
    fig.suptitle(f"Census sections · Final vs Census · {metric.replace('_',' ')} — ALL sections",fontsize=15)
    fig.subplots_adjust(hspace=.3,wspace=.22)
    _save(fig,out,manifest,f"final_{metric}_overview",
          f"All-section agreement · {metric.replace('_',' ')}",
          "Every section with Census and Final values contributes; zero synthetic counts remain included.")
    frame["residual_count"]=residual
    frame["abs_error_pct"]=np.abs(pct)
    frame.to_csv(out/f"final_{metric}_all_section_errors.csv",index=False)
    return stats


def _section_sampling_qa(g, output_dir, sampling_rate, manifest):
    """Document quantization of 1%-sampled household/person counts."""
    factor=1/sampling_rate
    cols=["macrozone_id","census_households","final_households","census_population","final_population"]
    frame=g[cols].copy()
    frame["sampled_households"]=frame.final_households*sampling_rate
    frame["sampled_people"]=frame.final_population*sampling_rate
    frame["household_count_difference"]=frame.final_households-frame.census_households
    frame.to_csv(output_dir/"section_sampling_granularity.csv",index=False)
    sizes=frame.sampled_households.dropna()
    fig,ax=plt.subplots(figsize=(9.5,5.6))
    labels=sizes.round().astype(int).value_counts().sort_index()
    ax.bar(labels.index.astype(str),labels.values,color=COLORS["Final"])
    ax.set(xlabel="Sampled Final households per census section (before expansion)",
           ylabel="Number of census sections",title=f"Why expanded counts repeat · each sampled household = {factor:g} expanded")
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    _save(fig,output_dir,manifest,"sampling_granularity_households",
          "Final sample granularity by census section",
          f"For sampling_rate={sampling_rate:g}, 4 sampled households = {4*factor:g} expanded households. "
          "This histogram uses ALL sections, not only the 12 largest discrepancies.")
    return {"sampling_expansion_factor":factor,
            "sampled_households_per_section_frequency":{str(int(k)):int(v) for k,v in labels.items()},
            "sections_with_zero_sampled_households":int((sizes==0).sum())}


def _section_validation(ipu, final, homes, sections, pop, hh, employment, licences,
                        out, sampling_rate, city_code):
    """Four census-only section metrics, preserving per-section reference values."""
    out.mkdir(parents=True,exist_ok=True)
    geoms,missing_geom=_validate_census_sections(sections,pop)
    refs=pd.concat([
        _section_reference_sum(pop,"weight","census_population"),
        _section_reference_sum(hh,"total_households","census_households"),
        _section_reference_sum(employment,"weight","census_employed_residents"),
        _section_reference_sum(licences.loc[numeric(licences.age_class).ge(18)],
                               "weight","census_licensed_residents"),
    ],axis=1)
    final_counts,final_qa=_section_residence_counts(final,homes,geoms,city_code)
    result=geoms.set_index("macrozone_id",drop=False).join(refs)
    for label,frame in (("final",final_counts),):
        renamed=frame[list(BASE_METRICS)].add_prefix(label+"_")
        result=result.join(renamed)
        for metric in BASE_METRICS:
            col=f"{label}_{metric}"
            # A modeled zero-resident section means zero synthetic households /
            # people, not a missing census reference. Unmapped home points are
            # reported independently in the QA; they are NOT assigned a zone.
            if metric in ("employed_residents","licensed_residents"):
                # A present residential sample with an entirely unknown flag
                # must remain NA, not become a misleading zero prevalence.
                result.loc[result[f"{label}_population"].fillna(0).eq(0),col]=0.
            else:
                result[col]=result[col].fillna(0)
            ref=f"census_{metric}"
            result[f"{label}_{metric}_difference"]=result[col]-result[ref]
            result[f"{label}_{metric}_error_pct"]=metric_error(result[ref],result[col])
    result=result.reset_index(drop=True)
    result.drop(columns="geometry").to_csv(out/"zone_comparison.csv",index=False)
    manifest=[];summaries=[]
    sampling_qa=_section_sampling_qa(result,out,sampling_rate,manifest)
    for label in ("final",):
        for metric in BASE_METRICS:
            cen=f"census_{metric}";syn=f"{label}_{metric}";err=f"{label}_{metric}_error_pct"
            slug=f"{label}_{metric}_counts"
            _section_comparison_plot(result,cen,syn,err,out,manifest,slug,
                                      "Census sections · "+label.upper()+" vs Census · "+metric.replace("_"," "))
            stats=_section_overview(result,cen,syn,out,manifest,metric,sampling_rate)
            if stats is not None:summaries.append(stats)
    pd.DataFrame(manifest).to_csv(out/"plot_manifest.csv",index=False)
    result.to_file(out.parent/"seville_census_section_validation.gpkg",layer="census_section_metrics",driver="GPKG")
    write_index(out,"Seville census-section validation",manifest,[
        "10-digit census sections; final residence determined from exported homes.gpkg points, not OD macrozone polygons.",
        "Overview charts (final_*_overview.png) use ALL census sections: Census-vs-Final scatter, count-error and absolute-percentage-error distributions, and accuracy statistics.",
        "Top-12 bars are supplemental; right-hand maps and all-section CSV/GPKG retain every section.",
        "sampling_granularity_households.png and section_sampling_granularity.csv explain discrete 1% expansion steps.",
        "Only four Census-backed counts: population, households, employed residents and driving-licence holders.",
        "Employment and licence section reference numbers are allocated/estimated, not independent direct section observations.",
        "Households without exported homes are omitted from section metrics only; persons remain in metropolitan non-spatial totals.",
        "Section boundaries may disagree with OD polygons; section-only point misses are reported separately.",
    ])
    quality={"sections_with_geometry":len(result),"census_sections_without_geometry":missing_geom,
             "sampling":sampling_qa,"final":final_qa,
             "census_sections_without_population_reference":int(result.census_population.isna().sum())}
    write_json(out/"section_quality.json",quality)
    pd.DataFrame(summaries).to_csv(out/"census_section_accuracy_summary.csv",index=False)
    return pd.DataFrame(summaries),manifest,quality


def create_spatial_validation(ipu,final,activities,zones,census_population,census_households,
                              census_employment,census_licenses,grid_jobs,education,
                              output_dir,sampling_rate=1.,city_code="41091",min_people=10,
                              final_homes=None,final_households=None,census_sections=None):
    if not 0<sampling_rate<=1:raise ValueError("Invalid sampling_rate")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    zones=_validate_zones(zones)
    final,home_qa=assign_final_residences(final,final_homes,zones,city_code)
    fp=prepare_persons(final,"Final",sampling_rate)
    # Validation comparisons must be Final vs Census only. The legacy
    # grid_jobs/education positional parameters are accepted for API stability
    # but intentionally ignored (different-year/different-definition references).
    ref,facility_qa=_build_reference(census_population,census_households,census_employment,
                                      census_licenses,None,None,city_code)
    counts_fp,qa_final=_residential_counts(fp,city_code)
    base=zones.set_index("macrozone_id",drop=False)
    qa={"zone_count":len(base),"final":qa_final,
        "final_home_mapping":home_qa,"facilities":facility_qa}
    if final_households is not None:
        if "household_id" not in final_households or final_households.household_id.isna().any():
            raise ValueError("Exported households.csv lacks valid household_id")
        hh_ids=ids(final_households.household_id)
        if hh_ids.duplicated().any():
            raise ValueError("Exported households.csv has duplicate household_id")
        if set(ids(final.household_id)) != set(hh_ids):
            raise ValueError("Final persons/households.csv IDs do not agree")
        qa["exported_households_count"]=len(final_households)
    for source,frame in (("final",counts_fp),):
        unknown=frame.index.dropna().difference(base.index)
        qa[f"{source}_unknown_macrozones"] = list(unknown)
        if len(unknown):LOG.warning("%s: %s residential macrozone IDs unknown to gravity",source,len(unknown))
        frame.columns=[f"{source}_{col}" for col in frame]
        base=base.join(frame)
    qa["census_unknown_macrozones"]=list(ref.index.dropna().difference(base.index))
    base=base.join(ref)
    if activities is None:
        raise ValueError("Spatial validation requires exported geocoded activities.gpkg")
    for purpose in ("work","education"):
        cnt,q,person_z=_activity_counts(activities,zones,purpose,sampling_rate)
        qa[purpose+"_activity_mapping"]=q
        base[f"final_{purpose}_attendees"]=cnt
        base[f"final_{purpose}_attendees"]=base[f"final_{purpose}_attendees"].fillna(0)
        person_z[["person_id","macrozone_id"]].to_csv(out/f"{purpose}_activity_person_zones.csv",index=False)
    # Missing census values are not implicitly zero: distinguish absent reference
    # from a source that explicitly reported zero.
    for source in ("final",):
        for metric in ("population","households","employed_residents","licensed_residents","students_resident","adult_residents","employment_eligible_residents","female_residents","under15_residents","age65plus_residents",
                       "households_size_1","households_size_2","households_size_3","households_size_4","households_size_5plus"):
            col=f"{source}_{metric}"
            if col in base:
                # Unpopulated sample zones have true zero observations. Where
                # the sample exists but an attribute is unknown, preserve NA.
                if metric in ("employed_residents","licensed_residents","students_resident"):
                    base.loc[base[f"{source}_population"].fillna(0).eq(0),col]=0
                else:
                    base[col]=base[col].fillna(0)
    for source in ("final",):
        if float(base[f"{source}_population"].sum())<=0:
            raise ValueError(f"{source} has no residents mapped to the metropolitan zones; do not plot zeros")
    base["_scale"]=1/sampling_rate
    for col in ("final_work_attendees","final_education_attendees"):
        base[col+"_density"]=base[col]/(base.geometry.area/1e6)
    complete=base.reset_index(drop=True)
    summaries=[];manifest=[]
    for scope in ("city_districts","metro_macrozones","municipalities"):
        sub=_derive(_select_geography(complete,scope,city_code))
        directory=out/scope
        manifest.extend([{**entry,"scope":scope} for entry in _render_scope(sub,directory,scope,min_people)])
        # Propagate an explicit short table of global errors, with warning
        # flags for mismatched definitions and census-disaggregation assumptions.
        for metric in BASE_METRICS:
            c=f"census_{metric}"
            if c not in sub:continue
            for source in ("final",):
                s=f"{source}_{metric}"
                if s not in sub:continue
                match=sub[[c,s]].dropna()
                if match.empty:continue
                summaries.append({"scope":scope,"metric":metric,"source":source,
                                  "n_zones_with_reference":len(match),
                                  "census_total":match[c].sum(),"synthetic_total":match[s].sum(),
                                  "relative_total_error_pct": float(metric_error(np.array([match[c].sum()]),
                                                                                  np.array([match[s].sum()]))[0]),
                                  "zone_mae":np.mean(abs(match[s]-match[c])),
                                  "zone_wape_pct":100*abs(match[s]-match[c]).sum()/match[c].sum() if match[c].sum()>0 else np.nan})
        # GPKG only macrozones, for GIS reuse. CSV above always produced.
        if scope=="metro_macrozones":
            gpkg=sub.drop(columns="_scale",errors="ignore")
            gpkg.to_file(out/"seville_spatial_validation.gpkg",layer="macrozone_metrics",driver="GPKG")
    if census_sections is not None:
        section_stats,section_plots,section_qa=_section_validation(
            None,fp,final_homes,census_sections,census_population,census_households,
            census_employment,census_licenses,out/"census_sections",sampling_rate,city_code)
        qa["census_sections"]=section_qa
        manifest.extend([{**entry,"scope":"census_sections"} for entry in section_plots])
        summaries.extend(section_stats.to_dict("records"))
    else:
        LOG.warning("Section-level validation skipped: seville.data.spatial.iris polygons not supplied")
    from .validation_utils import write_home_exclusion_report
    # Activity-destination counts/geometry do not depend on the person's home.
    # Residential plots, resident-normalized attendance plots, and home-to-work
    # OD do depend on it; exclude only the households without a point.
    affected = [m for m in manifest if m["file"].startswith("final_") and
                (m["file"].endswith("_counts.png") or "_comparison.png" in m["file"] or
                 "_share_gap_pp.png" in m["file"] or "_per_1000_residents.png" in m["file"])]
    if not home_qa.get("households_missing_home", 0):
        affected=[]
    affected_outputs=[
            {"file":"home_to_work_macrozone_activity_od.csv", "scope":"metropolitan",
             "note":"Origins require a mapped household home; missing-home persons are omitted."},
            {"file":"seville_spatial_validation.gpkg", "scope":"metropolitan",
             "note":"Final residential metrics exclude missing-home households."},
            {"file":"census_accuracy_by_geography.csv", "scope":"all geographies",
             "note":"Final residential aggregates are based only on mapped homes."},
        ]
    write_home_exclusion_report(out,home_qa,affected,
        report_name="Seville spatial validation",
        additional_outputs=(affected_outputs + [
            {"file":f"{scope}/zone_comparison.csv", "scope":scope,
             "note":"Final residential counts and shares omit households without a home location."}
            for scope in ("city_districts", "metro_macrozones", "municipalities", "census_sections")
        ]) if affected else [])
    summaries=pd.DataFrame(summaries)
    summaries.to_csv(out/"census_accuracy_by_geography.csv",index=False)
    pd.DataFrame(manifest).to_csv(out/"plot_manifest.csv",index=False)
    # Regional OD: one assigned workplace point per person, scoped to those
    # with a work activity; residential zone is from the final person code.
    work_file=out/"work_activity_person_zones.csv"
    if work_file.exists():
        work=pd.read_csv(work_file,dtype={"macrozone_id":"string"})
        p=fp[["person_id","macrozone_id_validation"]]
        work=work.merge(p,on="person_id",how="left",validate="one_to_one")
        flow=(work.dropna(subset=["macrozone_id","macrozone_id_validation"])
              .groupby(["macrozone_id_validation","macrozone_id"]).size().mul(1/sampling_rate)
              .rename("one_day_work_persons_expanded").reset_index())
        flow.rename(columns={"macrozone_id_validation":"origin_id","macrozone_id":"destination_id"}).to_csv(
            out/"home_to_work_macrozone_activity_od.csv",index=False)
    write_json(out/"spatial_quality.json",{
        **qa,"sampling_rate":sampling_rate,"source_population_year":2022,
        "workplace_grid_reference_year":2024,
        "licence_reference":"2024 municipality counts allocated to 2022 census sections and adjusted to 2022",
        "education_reference":"No Census attendance reference; only Final destination attendance is plotted",
        "work_attendance":"one unique model-day work activity person, not stock of jobs",
        "education_attendance":"one unique model-day education activity person, not total enrolment",
        "census_households":"2021 municipal totals adjusted to 2022 and apportioned to census sections",
        "spatial_model_application":"Seville city districts and surrounding municipalities",
        "home_location_exclusions":"home_location_exclusions.csv and home_location_affected_plots.csv",
        "home_required":"Only residential spatial metrics and residential-origin OD; activity destinations use their own geometry",
        "census_section_comparison":"Native 10-digit census sections (only population/households/employment/licences); final homes mapped from point geometries",
    })
    write_index(out,"Seville geographic validation",[
        {"file":"home_location_exclusions.png","title":"Households excluded from home-location-dependent validation",
         "note":"Spatial plots affected by exclusions are listed in home_location_affected_plots.csv."}
    ],[
        "Only Final is compared against Census; IPU and HTS are excluded from this report.",
        "All-section census validation is summarized with scatter plots and numeric accuracy metrics.",
        "Home-location exclusion chart: home_location_exclusions.png. List of location-dependent plots: home_location_affected_plots.csv.",
        "Workplace-grid and education-facility share comparisons are exploratory because the data measure different stocks/attendances.",
    ])
    report=out/"index.html"
    report.write_text(report.read_text().replace("</body>",
        '<p><a href="home_location_exclusions.png">Home-location exclusion chart</a> — '
        '<a href="home_location_affected_plots.csv">plots affected by missing exported homes</a></p>'
        '<p><a href="city_districts/index.html">Seville city districts</a></p>'
        '<p><a href="metro_macrozones/index.html">Full metropolitan macrozones</a></p>'
        '<p><a href="municipalities/index.html">Municipalities</a></p>'
        + ('<p><a href="census_sections/index.html">Census sections</a></p>' if census_sections is not None else '')
        + '</body>'),encoding="utf-8")
    return summaries


def execute(context):
    root=Path(context.config("output_path"));prefix=context.config("output_prefix")
    people=pd.read_csv(root/f"{prefix}persons.csv",sep=";",low_memory=False)
    homes_path=root/f"{prefix}homes.gpkg"
    if not homes_path.exists():
        raise FileNotFoundError(f"Final population spatial validation requires {homes_path}")
    homes=gpd.read_file(homes_path)
    households_path=root/f"{prefix}households.csv"
    households=pd.read_csv(households_path,sep=";",low_memory=False) if households_path.exists() else None
    file=root/f"{prefix}activities.gpkg"
    if not file.exists():
        alt=root/f"{prefix}activities.geoparquet"
        if not alt.exists():raise FileNotFoundError(f"Need {file} or {alt}")
        activities=gpd.read_parquet(alt)
    else:activities=gpd.read_file(file)
    zones=context.stage("seville.gravity.od_zones")[0]
    return create_spatial_validation(
        None,people,activities,zones,
        context.stage("seville.data.census.population"),
        context.stage("seville.data.census.households"),
        context.stage("seville.data.census.employment"),
        context.stage("seville.data.census.licenses"),
        None, None,
        Path(context.config("analysis_path"))/"seville_spatial_population_validation",
        float(context.config("sampling_rate")),
        str(context.config("gravity_calibration_municipality_id")),
        int(context.config("spatial_validation_min_people")),
        final_homes=homes,final_households=households,
        census_sections=context.stage("seville.data.spatial.iris"),
    )
