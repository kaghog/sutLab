import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def configure(context):
    context.config("output_path")
    context.config("data_path")
    context.config("analysis_path")
    context.config("output_prefix")
    context.stage("synthesis.output")
    context.stage("data.hts.entd.reweighted")


def import_data_synthetic(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")
    
    filepath = "%s/%strips.csv" % (output_path, output_prefix)
    df_trips = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    return df_trips   


def import_data_actual(context):
    _, _, df_act_trips = context.stage("data.hts.entd.reweighted")
    return df_act_trips

def execute(context):
    syn_trips = import_data_synthetic(context)
    hts_trips = import_data_actual(context)

        # ----- IPU synthetic -----
    syn_trips["weight"] = 1.0
    syn_trips = (
        syn_trips.groupby("mode", as_index=False)["weight"]
        .sum()
    )
    syn_trips["weight"] = syn_trips["weight"] / syn_trips["weight"].sum() * 100

    # ----- HTS output -----
    hts_trips = hts_trips.rename(columns={"trip_weight":"weight"})
    hts_trips = (
        hts_trips.groupby("mode", as_index=False)["weight"]
        .sum()
    )
    hts_trips["weight"] = hts_trips["weight"] / hts_trips["weight"].sum() * 100


    # ----- Align bins -----
    bins = sorted(
        set(syn_trips["mode"])
        | set(hts_trips["mode"])
    )

    def align(df):
        return df.set_index("mode").reindex(bins, fill_value=0)["weight"]

    ipu_w = align(syn_trips)
    output_w = align(hts_trips)

    # ----- Plot (side-by-side bars, no transparency) -----
    x = np.arange(len(bins))
    w = 0.1

    plt.figure(figsize=(11, 6))
    plt.bar(x - w, ipu_w, width=w, label="IPU synthetic")
    plt.bar(x + w, output_w, width=w, label="HTS output")

    plt.xticks(x, bins, rotation=45)
    plt.xlabel("Modes of transportation")
    plt.ylabel("Trip share (%)")
    plt.title("Modes of transportation distribution comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        f"{context.config('analysis_path')}/mode_shares.png"
    )
    plt.close()


