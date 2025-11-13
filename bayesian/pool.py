import pandas as pd
import tqdm
import numpy as np
import geopandas as gpd
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import os

from synthesis.model.params import Config as config
import synthesis.model.bayesian.model as bn
import utils.constants as cons
from synthesis.model.bayesian.model import BayesianNetwork
import synthesis.prepare.helpers as helpers




"""
This stage ..
"""

def configure(context):
    context.stage("synthesis.prepare.prepare")
    context.stage("data.dhs.households")

    context.config("output_path")
    context.config("output_prefix", "base_alchemy")

def _learn_core_structure(context, core_df, hh_type, model_name="Core BN"):
    core_for_bn = core_df.copy()
    core_structure = bn.learn_structure(
        core_for_bn,
        scoring_method=config.SCORING_METHOD,
        algorithm=config.STRUCTURE_ALGORITHM,
        start_dag=None,
        expert_knowledge=None,
        **config.STRUCTURE_KWARGS
    )
    core_bn = bn.estimate_parameters(core_structure, core_for_bn,
                                  estimator_name=config.PARAMETER_ESTIMATOR,
                                  **config.PARAM_ESTIMATOR_KWARGS)
    
    bn.print_network_edges(core_bn, name=f"{model_name} for {hh_type}")
    save_path = context.config('output_path') + context.config('output_prefix') + f"bn_structure_{model_name}_{hh_type}.png"
    bn.visualize_bn(core_bn, title=f"{model_name} for {hh_type}", save_path=save_path)
    return core_for_bn, core_bn

def _learn_children_structure(context, children_df, hh_type, model_name="Children BN"):

    print("Learning children BN...")
    
    #need to build expert knowledge and optional start DAG
    expert_child, start_child = bn.build_expert_knowledge_for_dependents(children_df, role="children")
    child_structure = bn.learn_structure(
        children_df,
        scoring_method=config.SCORING_METHOD,
        algorithm=config.STRUCTURE_ALGORITHM,
        start_dag=start_child,
        expert_knowledge=expert_child,
        **config.STRUCTURE_KWARGS
    )

    # keep only core->child style edges
    child_structure = BayesianNetwork(child_structure.edges())
    child_structure = bn.filter_bayesian_network(
        child_structure,
        household_nodes=[c for c in children_df.columns if c in cons.HOUSEHOLD_COLS],
        head_nodes=[c for c in children_df.columns if c.startswith("Head_")],
        spouse_nodes=[c for c in children_df.columns if c.startswith("Spouse_")],
        child_nodes=[c for c in children_df.columns if c.startswith("Child_")],
        other_nodes=None
    )
    child_bn = bn.estimate_parameters(child_structure, children_df,
                                    estimator_name=config.PARAMETER_ESTIMATOR,
                                    **config.PARAM_ESTIMATOR_KWARGS)
    
    
    bn.print_network_edges(child_bn, name=f"{model_name} for {hh_type}")
    save_path = context.config('output_path') + context.config('output_prefix') + f"bn_structure_{model_name}_{hh_type}.png"
    bn.visualize_bn(child_bn, title=f"{model_name} for {hh_type}", save_path=save_path)
    
    return child_bn

def _learn_others_structure(context, others_df, hh_type, model_name="Others BN"):
    print("Learning others BN...")
    expert_other, start_other = bn.build_expert_knowledge_for_dependents(others_df, role="others")
    other_structure = bn.learn_structure(
        others_df,
        scoring_method=config.SCORING_METHOD,
        algorithm=config.STRUCTURE_ALGORITHM,
        start_dag=start_other,
        expert_knowledge=expert_other,
        **config.STRUCTURE_KWARGS
    )
    other_structure = BayesianNetwork(other_structure.edges())
    other_structure = bn.filter_bayesian_network(
        other_structure,
        household_nodes=[c for c in others_df.columns if c in cons.HOUSEHOLD_COLS],
        head_nodes=[c for c in others_df.columns if c.startswith("Head_")],
        spouse_nodes=[c for c in others_df.columns if c.startswith("Spouse_")],
        child_nodes=None,
        other_nodes=[c for c in others_df.columns if c.startswith("Other_")]
    )
    other_bn = bn.estimate_parameters(other_structure, others_df,
                                    estimator_name=config.PARAMETER_ESTIMATOR,
                                    **config.PARAM_ESTIMATOR_KWARGS)
    
    bn.print_network_edges(other_bn, name=f"{model_name} for {hh_type}")
    save_path = context.config('output_path') + context.config('output_prefix') + f"bn_structure_{model_name}_{hh_type}.png"
    bn.visualize_bn(other_bn, title=f"{model_name} for {hh_type}", save_path=save_path)
    
    return other_bn
#TODO: check and combine with learn_children if needed as learn_dependents

def _sample_dependents(df_bn, dependent_df, synthetic_core, role, hh_type):
    # sample children or others per household of the synthetic core
    synthetic_df = pd.DataFrame()
    dependent_rows = []
    for idx, row in synthetic_core.iterrows():
        if role == "children":
            needed = int(row.get("num_child", 0))
        elif role == "others":
            needed = int(row.get("num_other", 0))
        else:
            raise ValueError(f"Unsupported role: {role}")
        
        if needed <= 0:
            continue
        dep = bn.sample_dependents_per_household(
            role=role,
            bn_model=df_bn,
            core_row=row,
            household_vars=[c for c in dependent_df.columns if c in cons.HOUSEHOLD_COLS],
            head_vars=[c for c in dependent_df.columns if c.startswith("Head_")],
            spouse_vars=[c for c in dependent_df.columns if c.startswith("Spouse_")],
            count_needed=needed,
            method=config.DEPENDENT_SAMPLING_METHOD,
            lws_multiplier=config.LWS_MULTIPLIER
        )
        if not dep.empty:
            dependent_rows.append(dep)
    if dependent_rows:
        synthetic_df = pd.concat(dependent_rows, ignore_index=True)
        print(f"Synthetic {role} generated for {hh_type}: {len(synthetic_df)}")
    
    return synthetic_df

def _assign_counts_with_coverage(synthetic_core: pd.DataFrame,
                                real_counts: pd.DataFrame,
                                count_col: str) -> pd.DataFrame:
    # Extract the distribution from the counts DataFrame, not from the column directly
    if count_col == "num_child":
        dist = real_counts["num_child"].value_counts(normalize=True).sort_index()
    elif count_col == "num_other":
        dist = real_counts["num_other"].value_counts(normalize=True).sort_index()
    else:
        raise ValueError(f"Unsupported count column: {count_col}")

    if dist.empty:
        synthetic_core[count_col] = 0
        return synthetic_core

    all_vals = dist.index.tolist()
    all_probs = dist.values
    n = len(synthetic_core)

    guaranteed = all_vals.copy()[:min(n, len(all_vals))]
    remaining = max(0, n - len(guaranteed))
    extra = list(np.random.choice(all_vals, size=remaining, p=all_probs, replace=True)) if remaining > 0 else []
    final_vals = guaranteed + extra
    np.random.shuffle(final_vals)

    synthetic_core[count_col] = final_vals
    synthetic_core[count_col] = synthetic_core[count_col].astype(int)
    return synthetic_core


def run_bn(context, core_df, children_df, others_df, hh_type, counts_real):
    core_for_bn, core_bn = _learn_core_structure(context, core_df, hh_type)

    synthetic_core = bn.sample_core(core_bn, n_real=len(core_for_bn),
                                 multiplier=config.SAMPLE_SIZE_MULTIPLIER,
                                 method=config.CORE_SAMPLING_METHOD)

    print(f"Synthetic core households generated: {len(synthetic_core)}")
    
    # attach required counts of children / others using real empirical distributions
    synthetic_core = _assign_counts_with_coverage(synthetic_core, counts_real, "num_child")
    synthetic_core = _assign_counts_with_coverage(synthetic_core, counts_real, "num_other")


    # dependent sampling applies for these types (children and/or others present)
    types_with_children_or_others = {
        "Couple with children and Other household members",
        "Couple with children",
        "Single parent with children",
        "Single parent with children and Other household members",
        "Head with Other household members",
        "Couple with no children and Other household members"
    }
    need_dependents = hh_type in types_with_children_or_others

    # 4) defaults so variables always exist
    synthetic_children = None
    synthetic_other = None

    if need_dependents and children_df is not None and not children_df.empty:
        child_bn = _learn_children_structure(context, children_df, hh_type)
        synthetic_children = _sample_dependents(child_bn, children_df, synthetic_core, "children", hh_type)

    if need_dependents and others_df is not None and not others_df.empty:
        other_bn = _learn_others_structure(context, others_df, hh_type)
        synthetic_other = _sample_dependents(other_bn, others_df, synthetic_core, "others", hh_type)

    return synthetic_core, synthetic_children, synthetic_other


def reverse_to_individuals(synthetic_core: pd.DataFrame,
                           synthetic_children: Optional[pd.DataFrame],
                           synthetic_others: Optional[pd.DataFrame],
                           household_type_long: str) -> pd.DataFrame:
    out = []
    hh_code = cons.HOUSEHOLD_TYPES_MAP.get(household_type_long, "UNK")

    for i, row in synthetic_core.iterrows():
        hh_id = f"{hh_code}_HH{i+1:06d}"

        # common household-level attrs (keep only the household columns for inheritance)
        hh_attrs = {k: v for k, v in row.items() if k in cons.HOUSEHOLD_COLS}

        # head
        if any(k.startswith("Head_") for k in synthetic_core.columns):
            head = {k.replace("Head_", ""): row[k] for k in synthetic_core.columns if k.startswith("Head_")}
            head.update({"Household ID": hh_id, "Person ID": f"{hh_id}_P1", "Relationship to head": "Head", **hh_attrs})
            out.append(head)
            next_pid = 2
        else:
            # For single person households without explicit Head_ prefix
            if household_type_long == "Single":
                head = {k: row[k] for k in cons.INDIVIDUAL_COLS if k in row.index}
                head.update({"Household ID": hh_id, "Person ID": f"{hh_id}_P1", "Relationship to head": "Head", **hh_attrs})
                out.append(head)
                next_pid = 2
            else:
                next_pid = 1

        # spouse
        if any(k.startswith("Spouse_") for k in synthetic_core.columns):
            sp = {k.replace("Spouse_", ""): row[k] for k in synthetic_core.columns if k.startswith("Spouse_")}
            sp.update({"Household ID": hh_id, "Person ID": f"{hh_id}_P{next_pid}", "Relationship to head": "Wife or husband", **hh_attrs})
            out.append(sp)
            next_pid += 1

        # children
        if synthetic_children is not None:
            rows = synthetic_children[synthetic_children["_household_index"] == i]
            for _, crow in rows.iterrows():
                child = {k.replace("Child_", ""): crow[k] for k in rows.columns if k.startswith("Child_")}
                child.update({"Household ID": hh_id, "Person ID": f"{hh_id}_P{next_pid}", "Relationship to head": "Son/daughter", **hh_attrs})
                out.append(child)
                next_pid += 1

        # others
        if synthetic_others is not None:
            rows = synthetic_others[synthetic_others["_household_index"] == i]
            for _, orow in rows.iterrows():
                other = {k.replace("Other_", ""): orow[k] for k in rows.columns if k.startswith("Other_")}
                other.update({"Household ID": hh_id, "Person ID": f"{hh_id}_P{next_pid}", "Relationship to head": "Other relative", **hh_attrs})
                out.append(other)
                next_pid += 1

    if not out:
        return pd.DataFrame()

    result_df = pd.DataFrame(out)

    # Add household type column
    result_df["Household type"] = household_type_long

    # Recalculate household size based on actual number of members
    household_size_counts = result_df.groupby('Household ID').size()

    # Create mapping for household size encoding (matching original encoding)
    def encode_household_size(size):
        if size <= 4:
            return size
        else:
            return 5  # "5 or more"

    household_size_encoded = household_size_counts.apply(encode_household_size)

    # Map back to the result dataframe
    result_df['Household_size'] = result_df['Household ID'].map(household_size_encoded)

    # Reorder Columns
    result_df = result_df[['Person ID', 'Household ID'] + \
                          [col for col in result_df.columns if col not in ['Person ID', 'Household ID']]]

    return result_df

def compare_distributions(original_df, synthetic_df,
                          encoding_legend: Dict[str, Dict[str, int]], save_path="bn_comparisons.png",
                          normalize: bool = False) -> None:
    """
    Decodes and compares categorical distributions between original and synthetic data.
    """

    # Filter columns that exist in both dataframes and have mappings
    columns_to_compare = [col for col in encoding_legend.keys()
                          if col in original_df.columns
                          and col in synthetic_df.columns
                         ]

    if not columns_to_compare:
        print("No valid columns to compare.")
        return None

    # Process each column
    for column in columns_to_compare:
        # Get decoded categories for each dataset
        original_categories = original_df[column]
        synthetic_categories = synthetic_df[column]

        # Compute counts for each category
        original_counts = original_categories.value_counts().sort_index()
        synthetic_counts = synthetic_categories.value_counts().sort_index()

        # Ensure both have the same categories (fill missing with 0)
        all_categories = sorted(set(original_counts.index) | set(synthetic_counts.index))
        original_counts = original_counts.reindex(all_categories, fill_value=0)
        synthetic_counts = synthetic_counts.reindex(all_categories, fill_value=0)

        # Normalize counts if requested
        if config.NORMALIZE:
            original_total = original_counts.sum()
            synthetic_total = synthetic_counts.sum()
            original_normalized = (original_counts / original_total) * 100 if original_total > 0 else original_counts
            synthetic_normalized = (synthetic_counts / synthetic_total) * 100 if synthetic_total > 0 else synthetic_counts
            merged_df = pd.DataFrame({
                'Category': all_categories,
                'Original': original_normalized.values,
                'Synthetic': synthetic_normalized.values
            })
            y_label = "Percentage (%)"
            annotation_format = '{:.1f}%'
        else:
             # Create a merged dataframe for plotting
            merged_df = pd.DataFrame({
                'Category': all_categories,
                'Original': original_counts.values,
                'Synthetic': synthetic_counts.values
            })
            y_label = "Count"
            annotation_format = '{:d}'


        # Plot overlay (grouped bar chart)
        x = range(len(merged_df))
        width = 0.40
        fig, ax = plt.subplots(figsize=(15, 8))

        bars_orig = ax.bar([i - width/2 for i in x], merged_df['Original'], width, label='Original')
        bars_syn = ax.bar([i + width/2 for i in x], merged_df['Synthetic'], width, label='Synthetic', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=45, ha='center')
        ax.set_ylabel(y_label)
        ax.set_title(f"Distribution Comparison for {column}")
        ax.legend()

        # Annotate counts/percentages on top of each bar
        for bar in bars_orig:
            height = bar.get_height()
            ax.annotate(annotation_format.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        for bar in bars_syn:
            height = bar.get_height()
            ax.annotate(annotation_format.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        plt.savefig(save_path+f"_{column}.png")

def prep_households(df_households: pd.DataFrame, hh_type) -> pd.DataFrame:
    cols = {
    "Sex of household member": "Sex",
    "Age_bin of household members": "Age",
    "Current marital status": "Marital_status",
    "Educational attainment": "Education",
    "Has vehicle": "Has_vehicle",
    "Employment status": "Employment_status",
    "Type of place of residence": "Residence_type",
    "Number of rooms_bin": "Num_rooms",
    "Main floor material": "Floor_material",
    "Main wall material": "Wall_material",
    "Main roof material": "Roof_material",
    "Wealth index combined": "Wealth_index",
    "Household Size_bin": "Household_size",
    }
    df_households = df_households.rename(columns=cols)
    return df_households[df_households["Household type"] == hh_type].copy()

def execute(context):
    household_dict, counts_real_hhs = context.stage("synthesis.prepare.prepare")

    

    results = {}

    for hh_type, df_dict in household_dict.items():
        # 'core': core_df,
        #    'children': children_df,
        #    'others': others_df
        core_df = df_dict["core"].copy()
        children_df = df_dict["children"].copy()
        others_df = df_dict["others"].copy()

        counts_real = counts_real_hhs[hh_type]

        synthetic_core, synthetic_children, synthetic_others = run_bn(context,core_df, children_df, others_df, hh_type, counts_real)

        #plot distributions for validation compare

        #reverse to individual records (roles fixed to head/spouse/child/other)
        result = reverse_to_individuals(synthetic_core=synthetic_core,
                    synthetic_children=(synthetic_children if (synthetic_children is not None and not synthetic_children.empty) else None),
                    synthetic_others=(synthetic_others if (synthetic_others is not None and not synthetic_others.empty) else None),
                    household_type_long=hh_type)
        print(f"Final individual records generated for {hh_type}: {len(result)}")

        # ToDo syn = decode_dataframe(syn, ENCODING_LEGEND)
         # decode Dataframe back to labels
        result = helpers.decode_dataframe(result, cons.ENCODING_LEGEND)
        
        if result is None or result.empty:
            print(f"No synthetic data generated for {hh_type}")
            continue
        
        print(f"Final dataframe for {hh_type} has been decoded back to labels")
        print(f"Final dataframe shape for {hh_type}: {result.shape}")
        print(f"Final dataframe head for {hh_type}:\n{result.head()}")

        results[hh_type] = result

        #validate the distribution of the bayesian pool per household structure
        df_dhs_households = context.stage("data.dhs.households")

        #prepare and filter to the household type
        df_dhs = prep_households(df_dhs_households, hh_type)
        
        os.makedirs(f"{context.config('output_path')}/{context.config('output_prefix')}/bn_comparisons/{hh_type}", exist_ok=True)  

        save_path = context.config('output_path') + context.config('output_prefix') + f"/bn_comparisons/{hh_type}/" + f"bn_comparisons_{hh_type}.png"
        
        compare_distributions(df_dhs, result, cons.ENCODING_LEGEND, save_path)

    synpool = pd.concat(results.values(), ignore_index=True)

    return synpool, results