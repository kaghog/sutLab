import pandas as pd
import numpy as np
import geopandas as gpd
import analysis.hannover.ivt_style.myutils as myutils
import analysis.hannover.ivt_style.myplottools as myplottools
import matplotlib.pyplot as plt
#import data.constants as c
import pyproj
#import data.utils
import data.spatial.utils
from tqdm import tqdm
import data.hts.entd.cleaned
import warnings

# Suppress NetworkX FutureWarning about node_link_data/edges kwarg emitted upstream by dependencies.
# We don't call node_link_data here; this avoids noisy logs until upstream sets edges explicitly.
warnings.filterwarnings("ignore", message=r".*edges kwarg.*node_link_data.*", category=FutureWarning)


def configure(context):
    context.config("output_path")
    context.config("data_path")
    context.config("analysis_path")
    context.config("output_prefix")

    # context.stage("data.hts.trips")
    # context.stage("data.hts.persons")
    # context.stage("synthesis.output")
    context.stage("hannover.data.census.population")
    # Also prepare census license totals for comparison plots
    context.stage("hannover.ipf.prepare")
    context.stage("data.hts.entd.reweighted")

    context.config("weekend_scenario", False)
    context.config("specific_weekend_scenario", "all") # options are "all", "saturday", "sunday"
    context.config("specific_day_scenario", "avgworkday") #options can be any of the days of the week or "avgworkday"
    
    
def import_data_synthetic(context, population_selector = None):
    filepath = "%s/%strips.csv" % (context.config("output_path"), context.config("output_prefix"))
    df_trips = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    filepath = "%s/%spersons.csv" %  (context.config("output_path"), context.config("output_prefix"))
    df_persons = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    filepath = "%s/%shouseholds.csv" %  (context.config("output_path"), context.config("output_prefix"))
    df_hhl = pd.read_csv(filepath, encoding = "latin1", sep = ";")


    # Add activity flag: whether a person has at least one trip
    df_persons["is_active"] = df_persons["person_id"].isin(df_trips["person_id"]).astype(bool)

    # NOTE (kept for reference): legacy merges that dropped non-travelers.
    # df_syn = df_persons.merge(df_hhl, left_on="person_id", right_on="household_id")
    # df_syn = df_persons.merge(df_trips, left_on="person_id", right_on="person_id")

    # Debug: how many persons are covered by trips vs. total persons
    total_persons = len(df_persons)
    persons_in_trips = df_trips["person_id"].nunique()
    print(f"[DEBUG] Synthetic persons total={total_persons}, in trips={persons_in_trips} ({persons_in_trips/total_persons*100:.2f}%)")
    
    t_id = df_trips["person_id"].values.tolist()
    df_persons_no_trip = df_persons[np.logical_not(df_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])
    print(df_persons_no_trip.shape, "persons without trip in synpop")
    # print(f"[DEBUG] Synthetic df_syn rows={len(df_syn)}; unique persons in df_syn={df_syn['person_id'].nunique()}")

    if population_selector:
        if "age_selector" in population_selector.keys():
            age_min = population_selector["age_selector"][0]
            age_max = population_selector["age_selector"][1]
            # Apply filters to both persons and trips (keep person-level intact)
            df_persons = df_persons[(df_persons["age"] <= age_max) & (df_persons["age"] >= age_min)]
            df_trips = df_trips[(df_trips["age"] <= age_max) & (df_trips["age"] >= age_min)]
            df_persons_no_trip = df_persons_no_trip[(df_persons_no_trip["age"] <= age_max) & (df_persons_no_trip["age"] >= age_min)]
            print("INFO excluding agents NOT between the age of ", age_min, " and ", age_max)
        if "gender_selector" in population_selector.keys():
            gender = population_selector["gender_selector"]
            if gender == "male":
                g = 0
            else:
                g = 1
            df_persons = df_persons[df_persons["sex"] == g]
            df_trips = df_trips[df_trips["sex"] == g]
            df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["sex"] == g]
            print("INFO only considering ", gender, " agents.")
        if "canton_selector" in population_selector.keys():
            cantons = population_selector["canton_selector"]
            df_persons = df_persons[df_persons["canton_id"].isin(cantons)]
            df_trips = df_trips[df_trips["canton_id"].isin(cantons)]
            df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["canton_id"].isin(cantons)]
            print("INFO only considering agents living in cantons n° ", cantons)

    # Return person-level dataframe with is_active and the trips dataframe.
    # df_syn (merged persons-trips) is deprecated in favor of is_active flag.
    return df_persons, df_trips, df_persons_no_trip   


def import_data_actual(context, population_selector = None):
    df_act_households , df_act_persons, df_act_trips = context.stage("data.hts.entd.reweighted")
    
    # First ensure number_of_vehicles is numeric and convert to boolean
    df_act_households["number_of_vehicles"] = pd.to_numeric(df_act_households["number_of_vehicles"], errors="coerce").fillna(0)
    df_act_households["car_availability"] = df_act_households["number_of_vehicles"] > 0
    # Merge to persons dataframe on household_id
    df_act_persons = df_act_persons.merge(
        df_act_households[["household_id", "car_availability"]],
        on="household_id",
        how="left"
    )
    # Fill missing households with False (if person household_id not in households df)
    df_act_persons["car_availability"] = df_act_persons["car_availability"].fillna(False)

    df_act_persons.rename(columns = {"person_weight": "weight_person"}, inplace = True)
    df_px = df_act_persons[["person_id", "weight_person", "employed", "studies",
                                                "age", "sex", "car_availability", "has_license", "has_pt_subscription", "socioprofessional_class"]]
    df_act = df_act_trips.merge(df_px, on=["person_id"], how='left')

    df_act["preceding_purpose"] = df_act["preceding_purpose"].astype(str)
    df_act["following_purpose"] = df_act["following_purpose"].astype(str)
    df_act["od"] = df_act["preceding_purpose"] + "_" + df_act["following_purpose"]

    # Only keep the persons that could have been used in activity chain matching, due to 
    # using specifc days of the week
    df_act = df_act[~df_act["weight_person"].isna()]
    df_act = df_act.set_index(["person_id"])
    df_act.sort_index(inplace=True)

    t_id = df_act_trips["person_id"].values.tolist()
    # Tag active persons (with at least one trip)
    df_act_persons["is_active"] = df_act_persons["person_id"].isin(t_id).astype(bool)
    df_persons_no_trip = df_act_persons[np.logical_not(df_act_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])
    print(df_persons_no_trip.shape, "persons without trip in hts")
    # Debug: size and coverage
    total_hts_persons = len(df_act_persons)
    persons_in_trips = df_act_trips["person_id"].nunique()
    print(f"[DEBUG] HTS persons total={total_hts_persons}, in trips={persons_in_trips} ({persons_in_trips/total_hts_persons*100:.2f}%)")

    if population_selector:
        if "age_selector" in population_selector.keys():
            age_min = population_selector["age_selector"][0]
            age_max = population_selector["age_selector"][1]
            # Filter trip-level by person age via merged attributes
            df_act = df_act[(df_act["age"] <= age_max) & (df_act["age"] >= age_min)]
            df_persons_no_trip = df_persons_no_trip[(df_persons_no_trip["age"] <= age_max) & (df_persons_no_trip["age"] >= age_min)]
            print("INFO excluding agents NOT between the age of ", age_min, " and ", age_max)
        if "gender_selector" in population_selector.keys():
            gender = population_selector["gender_selector"]
            if gender == "male":
                g = 0
            else:
                g = 1
            df_act = df_act[df_act["sex"] == g]
            df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["sex"] == g]
            print("INFO only considering ", gender, " agents.")

    # df_act contains only those that have trips
    # Legacy return was (df_act_trip_merged, df_persons_no_trip). We now return the
    # person-level dataframe (with is_active) alongside the trips and the no-trip view.
    return df_act_persons, df_act_trips, df_persons_no_trip


    
def import_data_census(context, population_selector = None):
    df_population = context.stage("hannover.data.census.population")
    return df_population
    

def aux_data_frame(df_act_trips, df_syn_trips, df_act_persons, df_syn_persons, population_selector = None):
    # NOTE: Any population selection should be applied upstream on person-level frames
    # and then propagated to trips via person_id. This block is kept as a comment
    # to document the previous approach that attempted to filter here.
    # if population_selector:
    #     if "age_selector" in population_selector.keys():
    #         age_min = population_selector["age_selector"][0]
    #         age_max = population_selector["age_selector"][1]
    #         print("INFO excluding agents NOT between the age of ", age_min, " and ", age_max)
    #     if "gender_selector" in population_selector.keys():
    #         gender = population_selector["gender_selector"]
    #         print("INFO only considering ", gender, " agents.")

    # Work on a local reset copy to avoid mutating caller's df_act
    # Use trip-level frames for chain summaries; merge person weights onto trips
    df_act_reset = df_act_trips.reset_index()  # brings index name 'person_id' as a column
    if 'weight_person' not in df_act_reset.columns and 'weight_person' in df_act_persons.columns:
        df_act_reset = df_act_reset.merge(
            df_act_persons[['person_id','weight_person']], on='person_id', how='left'
        )
    pers_ids = df_act_reset["person_id"].unique()

    df_aux_act = pd.DataFrame({
        "person_id": pers_ids,
        # If weight_person missing (e.g., no merge), default to 1.0 via fillna below
        "weight_person": df_act_reset.groupby("person_id")["weight_person"].mean(),
        "chain": "home-" + df_act_reset.groupby("person_id")["following_purpose"].apply(lambda x: "-".join(x))
    }).fillna({"weight_person": 1.0})

    pers_ids = df_syn_trips["person_id"].unique()

    df_aux_syn = pd.DataFrame({
        "person_id": pers_ids,
        "weights": 1,
    "chain": "home-" + df_syn_trips.groupby("person_id")["following_purpose"].apply(lambda x: "-".join(x))
    })

    return df_aux_act, df_aux_syn


def activity_chains_comparison(context, all_CC, suffix = None):
    # Get percentages, prepare for plotting
    synthetic_sum = all_CC["synthetic Count"].sum()
    if synthetic_sum != 0:
        all_CC["synthetic Count"] = all_CC["synthetic Count"] / synthetic_sum * 100
    else:
        all_CC["synthetic Count"] = 0
    all_CC["actual Count"] = all_CC["actual Count"] / all_CC["actual Count"].sum() *100
    all_CC = all_CC.sort_values(by=['actual Count'], ascending=False)
    all_CC.to_csv("%s/actchains_DF.csv" % context.config("analysis_path"), index = False)

    # First step done: plot activity chain counts
    title_plot = "Synthetic and HTS activity chain comparison"
    title_figure = "activitychains"
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot, ylabel = "Percentage", xlabel = "Activity chain", lab = all_CC["Chain"], actual = all_CC["actual Count"], synthetic = all_CC["synthetic Count"], t = 15, figsize = [12,7], dpi = 300, w = 0.35, xticksrot=True)


def activity_counts_comparison(context, all_CC, suffix = None):
    all_CC_dic = all_CC.to_dict('records')
    counts_dic = {}
    for actchain in all_CC_dic:
        chain = actchain["Chain"]
        s = actchain["synthetic Count"]
        a = actchain["actual Count"]
        if np.isnan(s):
            s = 0
        if np.isnan(a):
            a = 0
        if chain == "-" or chain == "h":
            x = 0
        else:
            act = chain.split("-")
            x = len(act) - 2
        x = min(x, 7)
        if x not in counts_dic.keys():
            counts_dic[x] = [s, a]
        else:
            counts_dic[x][0] += s
            counts_dic[x][1] += a
    
    counts = pd.DataFrame(columns = ["number", "synthetic Count", "actual Count"])
    for k in range(min(8, np.max(list(counts_dic.keys())))):
        v = counts_dic[k]
        if k == 7:
            l = "7+"
        else:
            l = str(int(k))
        counts.loc[k] = pd.Series({"number": l, 
                                      "synthetic Count": v[0],
                                      "actual Count": v[1]
                                          })
    
    # Get percentages, prepare for plotting
    counts["synthetic Count"] = counts["synthetic Count"] / counts["synthetic Count"].sum() *100
    counts["actual Count"] = counts["actual Count"] / counts["actual Count"].sum() *100
    #counts = counts.sort_values(by=['actual Count'], ascending=False)

    # First step done: plot activity chain counts
    title_plot = "Synthetic and HTS activity counts comparison"
    title_figure = "activitycounts"
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot, 
                                    ylabel = "Percentage", xlabel = "Number of activities in the activity chain",
                                    lab = counts["number"], actual = counts["actual Count"], 
                                    synthetic = counts["synthetic Count"], xticksrot=True)
    
    
def activity_counts_per_purpose(context, all_CC, suffix = None):
    all_CC_dic = all_CC.to_dict('records')
    purposes = ['home', 'work', 'education', 'shop', 'leisure', 'other', "start_out_of_home"]
    counts_dic = {}
    for actchain in all_CC_dic:
        chain = actchain["Chain"]
        s = actchain["synthetic Count"]
        a = actchain["actual Count"]
        if np.isnan(s):
            s = 0
        if np.isnan(a):
            a = 0
        if chain == "-" or chain == "h":
            pass
        else:
            acts = chain.split("-")
            for act in acts:
                if act not in purposes:
                    purposes.append(act)
            for p in purposes:
                cpt_purpose = acts.count(p)
                if cpt_purpose > 0:
                    identifier = p + " - " + str(cpt_purpose) 
                    if cpt_purpose > 1:
                        identifier += " times"
                    else:
                        identifier += " time"
                    if identifier not in counts_dic.keys():
                        counts_dic[identifier] = [s, a]
                    else:
                        counts_dic[identifier][0] += s
                        counts_dic[identifier][1] += a
    
    counts = pd.DataFrame(columns = ["number", "synthetic Count", "actual Count"])

    for k, v in counts_dic.items():
        counts.loc[k] = pd.Series({"number": k, 
                                      "synthetic Count": v[0],
                                      "actual Count": v[1]
                                          })
            

    # Get percentages, prepare for plotting
    synthetic_sum = counts["synthetic Count"].sum()
    if synthetic_sum != 0:
       counts["synthetic Count"] = counts["synthetic Count"] / synthetic_sum * 100
    else:
        counts["synthetic Count"] = 0
    actual_sum = counts["actual Count"].sum()
    if actual_sum != 0:
        counts["actual Count"] = counts["actual Count"] / actual_sum * 100
    else:
        counts["actual Count"] = 0
    counts = counts.sort_values(by=['actual Count'], ascending=False)
    
    idx = counts.index.tolist() 
    counts = counts.reindex(idx)

    # First step done: plot activity chain counts
    title_plot = "Activity counts per purpose comparison"
    title_figure = "activitycountspurpose"
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot, 
                                    ylabel = "Percentage", xlabel = "Activities with the same purpose in the activity chain",
                                    lab = counts["number"], actual = counts["actual Count"], 
                                    synthetic = counts["synthetic Count"], t = 20, xticksrot=True)
    

def demographics_comparison(context, df_act_persons, df_syn_persons, df_census, suffix=None, use_active_only=False):

    # Age bins (Hannover): keep 0-5 and 6-14 separate
    bins = [0, 6, 15, 18, 24, 30, 45, 65, 80, 150]
    labels = ["0-5", "6-14", "15-17", "18-23", "24-29", "30-44", "45-64", "65-79", "80+"]


    # Use person-level frames directly; 'is_active' marks who has trips
    cols_act = [c for c in ["person_id", "age", "weight_person", "is_active", "has_license", "has_pt_subscription", "employed"] if c in df_act_persons.columns]
    cols_syn = [c for c in ["person_id", "age", "is_active", "has_driving_license", "has_pt_subscription", "employed"] if c in df_syn_persons.columns]
    df_act_persons = df_act_persons[cols_act].drop_duplicates(subset=["person_id"]).copy()
    df_syn_persons = df_syn_persons[cols_syn].drop_duplicates(subset=["person_id"]).copy()

    # Optional: restrict to active persons for plotting
    if use_active_only:
        if "is_active" in df_act_persons.columns:
            df_act_persons = df_act_persons[df_act_persons["is_active"]]
        if "is_active" in df_syn_persons.columns:
            df_syn_persons = df_syn_persons[df_syn_persons["is_active"]]

    # Cut ages into labeled bins (person-level)
    df_act_persons['age_bin'] = pd.cut(df_act_persons["age"], bins=bins, labels=labels)
    df_syn_persons['age_bin'] = pd.cut(df_syn_persons["age"], bins=bins, labels=labels)

    # Debug: bins and basic distributions before weighting/percentages
    print(f"[DEBUG] Age bins used: {bins}")
    print("[DEBUG] HTS age_bin value_counts (raw, person-level, incl. no-trip):\n", df_act_persons['age_bin'].value_counts(dropna=False))
    print("[DEBUG] SYN age_bin value_counts (raw, person-level, incl. no-trip):\n", df_syn_persons['age_bin'].value_counts(dropna=False))

    # Weighted HTS counts, unweighted synthetic counts; both as percentages
    act_counts = myplottools.compute_counts(df_act_persons['age_bin'], weights=df_act_persons['weight_person'], categories=labels)
    syn_counts = myplottools.compute_counts(df_syn_persons['age_bin'], weights=None, categories=labels)
    # Align to labels only (drop possible NaN bucket from unweighted path)
    import pandas as _pd
    act_counts = _pd.Series(act_counts).reindex(labels).fillna(0)
    syn_counts = _pd.Series(syn_counts).reindex(labels).fillna(0)

    # Debug: sums should be ~100, show small deviations
    print(f"[DEBUG] HTS percent sum={act_counts.sum():.6f}; SYN percent sum={syn_counts.sum():.6f}")
    if df_census is None:
        print("[DEBUG] Census is None in demographics_comparison; plotting will be HTS vs SYN only.")

    # Prepare variable to satisfy linters; will be overwritten if census provided
    census_counts = pd.Series(index=labels, dtype=float)

    # Census data processing and alignment
    if df_census is not None:
        age_class_to_label = dict(zip(bins[:-1], labels))
        df_census["age_label"] = df_census["age_class"].map(age_class_to_label)
        census_counts_raw = df_census.groupby("age_label")["weight"].sum()
        census_counts = (census_counts_raw / census_counts_raw.sum()) * 100
        census_counts = census_counts.reindex(labels).fillna(0)

        # Debug: census distribution sanity
        print("[DEBUG] Census counts sum=", census_counts.sum())
        print("[DEBUG] Census age_label distribution (percent):\n", census_counts)

        # Calculate differences (Synthetic - Reference)
        diff_hts = syn_counts - act_counts
        diff_census = syn_counts - census_counts

        # Debug: show biggest gaps
        ordered_labels = list(labels)
        df_debug = pd.DataFrame({
            "label": ordered_labels,
            "pct_hts": act_counts.reindex(ordered_labels).values,
            "pct_syn": syn_counts.reindex(ordered_labels).values,
            "pct_census": census_counts.reindex(ordered_labels).values,
            "gap_syn_minus_census": (syn_counts - census_counts).reindex(ordered_labels).values,
            "gap_hts_minus_census": (act_counts - census_counts).reindex(ordered_labels).values,
        })
        print("[DEBUG] Age distribution table (percentages and gaps vs census):\n", df_debug)

        # Plot differences
        title_figure_diff = "agedistribution_differences"
        title_plot_diff = "Age Distribution Differences from Synthetic"
        if suffix:
            title_plot_diff += " - " + suffix
            title_figure_diff += "_" + suffix
        title_figure_diff += ".png"

        myplottools.plot_distribution_differences(
            context,
            imtitle=title_figure_diff,
            plottitle=title_plot_diff,
            ylabel="Difference (Percentage Points)",
            xlabel="Age groups",
            lab=labels,
            diff_actual=diff_hts.values,
            diff_census=diff_census.values,
            xticksrot=True
        )

    # Main age plots (two-way or three-way)
    title_figure = "agedistribution"
    title_plot = "Age distribution comparison "
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
    title_figure += ".png"
    if df_census is None:
        myplottools.plot_comparison_bar(
            context,
            imtitle=title_figure,
            plottitle=title_plot,
            ylabel="Percentage",
            xlabel="Age groups",
            lab=labels,
            actual=act_counts.values,
            synthetic=syn_counts.values,
            xticksrot=True
        )
    else:
        myplottools.plot_comparison_bar(
            context,
            imtitle=title_figure,
            plottitle=title_plot,
            ylabel="Percentage",
            xlabel="Age groups",
            lab=labels,
            actual=act_counts.values,
            synthetic=syn_counts.values,
            census=census_counts.values,
            xticksrot=True
        )

        # Graph 1: Synthetic and HTS
        title_figure_syn_hts = "agedistribution_syn_hts"
        title_plot_syn_hts = "Age distribution comparison (Synthetic vs HTS) "
        if suffix:
            title_plot_syn_hts += " - " + suffix
            title_figure_syn_hts += "_" + suffix
        title_figure_syn_hts += ".png"
        myplottools.plot_comparison_bar(
            context,
            imtitle=title_figure_syn_hts,
            plottitle=title_plot_syn_hts,
            ylabel="Percentage",
            xlabel="Age groups",
            lab=labels,
            actual=act_counts.values,
            synthetic=syn_counts.values,
            xticksrot=True
        )

        # Graph 2: Synthetic and Census
        title_figure_syn_census = "agedistribution_syn_census"
        title_plot_syn_census = "Age distribution comparison (Synthetic vs Census) "
        if suffix:
            title_plot_syn_census += " - " + suffix
            title_figure_syn_census += "_" + suffix
        title_figure_syn_census += ".png"
        myplottools.plot_comparison_bar(
            context,
            imtitle=title_figure_syn_census,
            plottitle=title_plot_syn_census,
            ylabel="Percentage",
            xlabel="Age groups",
            lab=labels,
            synthetic=syn_counts.values,
            census=census_counts.values,
            xticksrot=True
        )

    # -------- Employment status comparison (person-level) --------
    def employment_status(df):
        return df["employed"].replace({ False: "unemployed", True: "employed"})

    df_act_persons_local = df_act_persons.copy()
    df_act_persons_local["employment_status"] = employment_status(df_act_persons_local)
    act_counts = myplottools.compute_counts(df_act_persons_local["employment_status"], weights=df_act_persons_local["weight_person"], categories=["unemployed", "employed"])  # fixed order

    syn_persons_local = df_syn_persons.copy()
    syn_employment = employment_status(syn_persons_local)
    syn_counts = myplottools.compute_counts(syn_employment, weights=None, categories=act_counts.index.tolist())
    syn_counts = myplottools.align_series(act_counts, syn_counts)

    title_figure = "employmentstatus"
    title_plot = "Employment status comparison "
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
    title_figure += ".png"
    myplottools.plot_comparison_bar(
        context,
        imtitle=title_figure,
        plottitle=title_plot,
        ylabel="Percentage",
        xlabel="Employment status",
        lab=act_counts.index,
        actual=act_counts.values,
        synthetic=syn_counts.values,
        xticksrot=True
    )

    # -------- Driving license (HTS as reference, add Census third series) --------
    act_license_labels = myplottools.map_bool_to_labels(df_act_persons.get("has_license"), yes_label="Yes", no_label="No")
    act_counts = myplottools.compute_counts(act_license_labels, weights=df_act_persons.get("weight_person"), categories=["No", "Yes"])  # fixed order

    syn_license_labels = myplottools.map_bool_to_labels(df_syn_persons.get("has_driving_license"), yes_label="Yes", no_label="No")
    syn_counts = myplottools.compute_counts(syn_license_labels, weights=None, categories=act_counts.index.tolist())
    syn_counts = myplottools.align_series(act_counts, syn_counts)

    title_figure = "drivinglicense"
    title_plot = "Driving license comparison "
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
    title_figure += ".png"

    # print("ACT: \n", act_counts)
    # print("SYN: \n", syn_counts)

    myplottools.plot_comparison_bar(
        context,
        imtitle=title_figure,
        plottitle=title_plot,
        ylabel="Percentage",
        xlabel="Has driving license",
        lab=act_counts.index,
        actual=act_counts.values,
        synthetic=syn_counts.values,
        xticksrot=True
    )

    # Additional plot including Census (IPF) as third series
    try:
        df_population, df_employment, df_licenses_country, df_licenses_kreis = context.stage("hannover.ipf.prepare")
        total_pop = df_population["weight"].sum()
        total_license = df_licenses_country["weight"].sum()
        census_yes = 100.0 * (total_license / total_pop) if total_pop > 0 else 0.0
        census_no = 100.0 - census_yes
        import pandas as _pd
        census_counts = _pd.Series({"No": census_no, "Yes": census_yes})
        census_counts = census_counts.reindex(act_counts.index).fillna(0)

        title_figure_all = "drivinglicense_all"
        title_plot_all = "Driving license comparison (HTS vs Synthetic vs Census) "
        if suffix:
            title_plot_all += " - " + suffix
            title_figure_all += "_" + suffix
        title_figure_all += ".png"

        myplottools.plot_comparison_bar(
            context,
            imtitle=title_figure_all,
            plottitle=title_plot_all,
            ylabel="Percentage",
            xlabel="Has driving license",
            lab=act_counts.index,
            actual=act_counts.values,
            synthetic=syn_counts.values,
            census=census_counts.values,
            xticksrot=True
        )
    except Exception as _e:
        print("Warning: could not add census driving license plot:", _e)

    # -------- Public transport subscription --------
    act_pt_labels = myplottools.map_bool_to_labels(df_act_persons.get("has_pt_subscription"), yes_label="Yes", no_label="No")
    act_counts = myplottools.compute_counts(act_pt_labels, weights=df_act_persons.get("weight_person"), categories=["No", "Yes"])  # fixed order

    syn_pt_labels = myplottools.map_bool_to_labels(df_syn_persons.get("has_pt_subscription"), yes_label="Yes", no_label="No")
    syn_counts = myplottools.compute_counts(syn_pt_labels, weights=None, categories=act_counts.index.tolist())
    syn_counts = myplottools.align_series(act_counts, syn_counts)

    title_figure = "ptsubscription"
    title_plot = "Public transport subscription comparison "
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
    title_figure += ".png"
    myplottools.plot_comparison_bar(
        context,
        imtitle=title_figure,
        plottitle=title_plot,
        ylabel="Percentage",
        xlabel="Has public transport subscription",
        lab=act_counts.index,
        actual=act_counts.values,
        synthetic=syn_counts.values,
        xticksrot=True
    )



def compute_distances_synthetic(df_syn, threshold = 25):
    df_syn["crowfly_distance"] = 0.001 * np.array(df_syn["crowfly_distance"])

    # Only consider crowfly distances shorter than <threshold> km
    df_syn_dist = df_syn[df_syn["crowfly_distance"] < threshold]
    df_syn_dist = df_syn_dist[df_syn_dist["crowfly_distance"] > 0]
    return df_syn_dist


def compute_distances_actual(df_act, threshold = 25):
    # Compute the distances
    df_act["crowfly_distance"] = 0.001 * np.sqrt(
        (df_act["origin_x"] - df_act["destination_x"])**2 + 
        (df_act["origin_y"] - df_act["destination_y"])**2
    )
    
    df_act_dist = df_act[df_act["crowfly_distance"] < threshold]
    df_act_dist = df_act_dist[df_act_dist["crowfly_distance"] > 0]
    return df_act_dist


def compare_dist_from_home(context, df_syn, df_act, target_purpose = "education", suffix = None):
    if not "origin_purpose" in df_act.columns:
        df_act.loc[:, "origin_purpose"] = df_act["preceding_purpose"]

    # select candidates
    filter_home_prim_syn = (df_syn["following_purpose"] == target_purpose) & (df_syn["preceding_purpose"] == "home")
    filter_home_prim_act = (df_act["following_purpose"] == target_purpose) & (df_act["origin_purpose"] == "home")
    filter_prim_home_syn = (df_syn["following_purpose"] == "home") & (df_syn["preceding_purpose"] == target_purpose)
    filter_prim_home_act = (df_act["following_purpose"] == "home") & (df_act["origin_purpose"] == target_purpose)


    df_syn_educ = df_syn[filter_home_prim_syn | filter_prim_home_syn].drop_duplicates(subset = ["person_id"])
    df_act_educ = df_act[filter_prim_home_act | filter_home_prim_act].drop_duplicates(subset = ["person_id"])

    pers_educ_syn = list(set(df_syn_educ["person_id"].values))
    pers_educ_act = list(set(df_act_educ["person_id"].values))

    dic_syn = {"person_id": pers_educ_syn, "dist_home_educ": [0 for i in range(len(pers_educ_syn))]}
    dic_act = {"person_id": pers_educ_act, "weight_person": [0 for i in range(len(pers_educ_act))], "dist_home_educ": [0 for i in range(len(pers_educ_act))]}

    for i in range(len(pers_educ_syn)):
        pid = pers_educ_syn[i]
        df_pers = df_syn_educ[df_syn_educ["person_id"] == pid]
        dist = 0.0
        for _, row in df_pers.iterrows():
            dist = float(row.get("crowfly_distance", 0.0))
        dic_syn["dist_home_educ"][i] = dist
            
    for i in range(len(pers_educ_act)):
        pid = pers_educ_act[i]
        df_pers = df_act_educ[df_act_educ["person_id"] == pid]
        home_x = 0.0
        home_y = 0.0
        educ_x = 0.0
        educ_y = 0.0
        have_home = False
        have_educ = False
        last_weight = 0.0
        for _, row in df_pers.iterrows():
            last_weight = float(row.get("weight_person", 0.0))
            if row.get("origin_purpose") != target_purpose:
                home_x = float(row.get("origin_x", home_x))
                home_y = float(row.get("origin_y", home_y))
                have_home = True
            elif row.get("following_purpose") != target_purpose:
                home_x = float(row.get("destination_x", home_x))
                home_y = float(row.get("destination_y", home_y))
                have_home = True
            if row.get("origin_purpose") == target_purpose:
                educ_x = float(row.get("origin_x", educ_x))
                educ_y = float(row.get("origin_y", educ_y))
                have_educ = True
            elif row.get("following_purpose") == target_purpose:
                educ_x = float(row.get("destination_x", educ_x))
                educ_y = float(row.get("destination_y", educ_y))
                have_educ = True
            if have_home and have_educ:
                break
        if have_home and have_educ:
            dist_val = 0.001 * float(np.sqrt(((home_x - educ_x) ** 2 + (home_y - educ_y) ** 2)))
            weight_val = last_weight
        else:
            dist_val = 0.0
            weight_val = 0.0
        dic_act["dist_home_educ"][i] = dist_val
        dic_act["weight_person"][i] = weight_val

    dist_df_syn = pd.DataFrame.from_dict(dic_syn)
    dist_df_act = pd.DataFrame.from_dict(dic_act)

    syn = dist_df_syn["dist_home_educ"].values
    act = dist_df_act["dist_home_educ"].values
    act_w = dist_df_act["weight_person"].values

    fig, ax = plt.subplots(1,1)
    x_data = np.array(syn, dtype=np.float64)
    x_sorted = np.argsort(x_data)
    x_weights = np.array([1.0 for i in range(len(syn))], dtype=np.float64)
    x_cdf = np.cumsum(x_weights[x_sorted])
    if len(x_cdf) >= 1:
        x_cdf /= x_cdf[-1]

    y_data = np.array(act, dtype=np.float64)
    y_sorted = np.argsort(y_data)
    y_weights = np.array(act_w, dtype=np.float64)
    y_cdf = np.cumsum(y_weights[y_sorted])
    if len(y_cdf) >= 1:
        y_cdf /= y_cdf[-1]

    ax.plot(y_data[y_sorted], y_cdf, label="Actual", color = "#A3A3A3")
    ax.plot(x_data[x_sorted], x_cdf, label="Synthetic", color="#00205B")  

    imtitle = "dist_home_"+target_purpose
    plottitle = "Distance from home to " + target_purpose
    if suffix:
        imtitle += "_" + suffix
        plottitle  += " - " + suffix 
    imtitle += ".png"

    ax.set_ylabel("Probability")
    ax.set_xlabel("Crowfly Distance [km]")
    ax.legend(loc="best")
    ax.set_title(plottitle)
    plt.savefig("%s/" % context.config("analysis_path") + imtitle)
    return syn, act, act_w


# def mode_purpose_comparison(context, df_syn, df_act, suffix = None):
#     # first in the synthetic data
#     types = df_syn.groupby(["mode","following_purpose"]).count()["person_id"]
#     syn = types / types.sum()

#     # then in the actual data
#     df_act.loc[df_act["mode"]=='car_passanger', "mode"] = 'car_passenger'
#     which = ["car","car_passenger","pt", "taxi","walk"]
#     atypes = df_act.groupby(["mode","destination_purpose"]).sum().loc[which,"weight_person"].reindex(index=which, level=0)
#     act = atypes / atypes.sum()
    
#     lista = [item for item in list(types.index.levels[0]) for i in range(len(types.index.levels[1]))]
#     listb = list(types.index.levels[1]) * len(types.index.levels[0])
#     labels = [a + " " + b for a, b in zip(lista,listb)]

#     # already ready to plot!
#     title_plot = "Synthetic and HTS Mode-Purpose Distribution"
#     title_figure = "modepurpose"
    
#     if suffix:
#         title_plot += " - " + suffix
#         title_figure += "_" + suffix
        
#     title_figure += ".png"
    
#     myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot,
#                                     ylabel = "Percentage", xlabel = "", lab = labels, 
#                                     actual = act.values.tolist(), synthetic = syn.values.tolist(), 
#                                     t = 10, xticksrot = True )



def all_the_plot_distances(context, df_act_dist, df_syn_dist, suffix = None):
    dph_title = "distance_purpose_hist"
    dmh_title = "distance_mode_hist"
    dpc_title = "distance_purpose_cdf"
    dmc_title = "distance_mode_cdf"
    
    if suffix:
        dph_title += "_" + suffix
        dmh_title += "_" + suffix
        dpc_title += "_" + suffix
        dmc_title += "_" + suffix
        
    dph_title += ".png"
    dph_title += ".png"
    dpc_title += ".png"
    dmc_title += ".png"
    
    myplottools.plot_comparison_hist_purpose(context, dph_title, df_act_dist, df_syn_dist, bins = np.linspace(0,25,120), dpi = 300, cols = 3, rows = 2)

    myplottools.plot_comparison_cdf_purpose(context, dpc_title, df_act_dist, df_syn_dist, dpi = 300, cols = 3, rows = 2)


def generate_plots(context, df_aux_act, df_aux_syn, df_act_trips, df_syn_trips, df_act_persons, df_syn_persons, df_syn_no_trip, df_act_no_trip, suffix, df_census):
    syn_CC = df_aux_syn.groupby("chain").size().reset_index(name='count')
    act_CC = df_aux_act.groupby("chain")["weight_person"].sum().reset_index(name='count')

    act_CC.columns = ["Chain", "actual Count"]
    syn_CC.columns = ["Chain", "synthetic Count"]

     # 1. ACTIVITY CHAINS
    
    # Creating the new dataframes with activity chain counts
    #syn_CC = myutils.process_synthetic_activity_chain_counts(df_syn)
    syn_CC.loc[len(syn_CC) + 1] = pd.Series({"Chain": "home", "synthetic Count": df_syn_no_trip.shape[0] })
   
    #act_CC = myutils.process_actual_activity_chain_counts(df_act, df_aux)
    act_CC.loc[len(act_CC) + 1] = pd.Series({"Chain": "home", "actual Count": np.sum(df_act_no_trip["weight_person"].values.tolist())})

    # Merging together, comparing
    all_CC = pd.merge(syn_CC, act_CC, on = "Chain", how = "outer")
    activity_chains_comparison(context, all_CC, suffix = suffix)
    
    # Number of activities    
    activity_counts_comparison(context, all_CC, suffix = suffix)
    
    # Number of activities per purposes
    activity_counts_per_purpose(context, all_CC, suffix = suffix)
    
    # Demographics comparison (include no-trip persons)
    demographics_comparison(context, df_act_persons, df_syn_persons, df_census, suffix)
    

    # 2. CROWFLY DISTANCES
    
    # 2.1. Compute the distances
    # df_syn_dist = compute_distances_synthetic(df_syn)
    # df_act_dist = compute_distances_actual(df_act) 
    
    # 2.2 Prepare for plotting
    # df_act_dist["x"] = df_act_dist["weight_person"] * df_act_dist["crowfly_distance"]

    # act = df_act_dist.groupby(["purpose"]).sum()["x"] / df_act_dist.groupby(["purpose"]).sum()["weight_person"]
    # syn = df_syn_dist.groupby(["following_purpose"]).mean()["crowfly_distance"] 

    # act_purposes = list(set(act.reset_index()["purpose"]))
    # syn = syn.reset_index()
    # for p in act_purposes:
    #     if p not in list(set(syn["following_purpose"])):
    #         syn.loc[len(syn)] = [p, 0]

    # syn = syn.groupby(["following_purpose"]).mean()["crowfly_distance"] 

    # # 2.3 Ready to plot!
    # myplottools.plot_comparison_bar(context, imtitle = "distancepurpose.png", plottitle = "Crowfly distance " + suffix, ylabel = "Mean crowfly distance [km]", xlabel = "", lab = syn.index, actual = act, synthetic = syn, t = None, xticksrot = True )
    # all_the_plot_distances(context, df_act_dist, df_syn_dist, suffix)

    # # 2.4 Distance from home to education
    # for primary_purpose in ["work", "education"]:
    #     print("INFO computing distances between home and", primary_purpose)
    #     syn_0, act_0, act_w0 = compare_dist_from_home(context, df_syn, df_act,primary_purpose, suffix = suffix)


    
def execute(context):
    pop_all = None
    suff_all = ""
    pop_selectors = [pop_all]
    suffixes      = [suff_all]

    for population_selector, suffix in list(zip(pop_selectors, suffixes)):
        df_syn_persons, df_syn_trips, df_syn_no_trip = import_data_synthetic(context, population_selector)
        df_act_persons, df_act_trips, df_act_no_trip = import_data_actual(context, population_selector)
        df_census = import_data_census(context, population_selector)
        df_aux_act, df_aux_syn = aux_data_frame(df_act_trips, df_syn_trips, df_act_persons, df_syn_persons)

        generate_plots(context, df_aux_act, df_aux_syn, df_act_trips, df_syn_trips, df_act_persons, df_syn_persons, df_syn_no_trip, df_act_no_trip, suffix, df_census)
