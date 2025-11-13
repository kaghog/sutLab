import pandas as pd
import tqdm
import numpy as np
from typing import List, Tuple, Dict, Optional

from synthesis.prepare.helpers import convert_column_to_int
import synthesis.prepare.helpers as helpers
import utils.constants as cons

"""
This stage is to...
"""


def configure(context):
    context.stage("data.dhs.households")

def _select_person(df: pd.DataFrame, relationship: str, prefix: str) -> pd.DataFrame:
    # the function selects a single-role person rows and prefixes personal columns
    person_df = df[df["Relationship to head"] == relationship].copy()
    person_df.rename(columns={c: f"{prefix}_{c}" for c in cons.INDIVIDUAL_COLS}, inplace=True)
    return person_df

def _select_group(df: pd.DataFrame, relationships: List[str], prefix: str) -> pd.DataFrame:
    # the function selects group roles (children/others), prefixes columns, and keeps an id column

    # Define first the exact columns needed for a dependent group
    cols_to_keep = ["Household ID", "Person ID"] + cons.INDIVIDUAL_COLS
    
    # Filter by relationship and select only the necessary columns
    group_df = df[df["Relationship to head"].isin(relationships)][cols_to_keep].copy()
    
    # Rename individual columns with the appropriate prefix
    group_df.rename(columns={c: f"{prefix}_{c}" for c in cons.INDIVIDUAL_COLS}, inplace=True)

    group_df.rename(columns={"Person ID": f"{prefix}_ID"}, inplace=True)
    
    return group_df

def transform_single(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    head = _select_person(df, "Head", "Head")
    core = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS].copy()
    core.drop(columns=["Household ID"], inplace=True)
    return core, None, None

def transform_couple_no_children(df: pd.DataFrame):
    head = _select_person(df, "Head", "Head")
    spouse = _select_person(df, cons.SPOUSE, "Spouse")
    head_main = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS]
    spouse_main = spouse[["Household ID"] + [f"Spouse_{c}" for c in cons.INDIVIDUAL_COLS]]
    core = pd.merge(head_main, spouse_main, on="Household ID", how="inner")
    core.drop(columns=["Household ID"], inplace=True)
    return core, None, None

def transform_couple_with_children_and_others(df: pd.DataFrame):
    head = _select_person(df, "Head", "Head")
    spouse = _select_person(df, cons.SPOUSE, "Spouse")
    children = _select_group(df, cons.CHILDREN, "Child")
    others = _select_group(df, cons.OTHER_MEMBERS, "Other")

    head_main = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS]
    spouse_main = spouse[["Household ID"] + [f"Spouse_{c}" for c in cons.INDIVIDUAL_COLS]]
    couple = pd.merge(head_main, spouse_main, on="Household ID", how="inner")

    # Create core reference without duplicating household columns
    core_ref = couple[['Household ID'] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + [f"Spouse_{c}" for c in cons.INDIVIDUAL_COLS]]

    # Merge children and others with core reference (no household columns to avoid duplicates)
    if not children.empty:
        children_combined = children.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original couple dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in couple.columns:
                children_combined[hh_col] = children_combined["Household ID"].map(
                    couple.set_index("Household ID")[hh_col]
                )
        children_combined = children_combined.drop(columns=["Child_ID", "Household ID"])
    else:
        children_combined = None

    if not others.empty:
        others_combined = others.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original couple dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in couple.columns:
                others_combined[hh_col] = others_combined["Household ID"].map(
                    couple.set_index("Household ID")[hh_col]
                )
        others_combined = others_combined.drop(columns=["Other_ID", "Household ID"])
    else:
        others_combined = None

    couple.drop(columns=["Household ID"], inplace=True)
    return couple, children_combined, others_combined

def transform_couple_with_children(df: pd.DataFrame):
    head = _select_person(df, "Head", "Head")
    spouse = _select_person(df, cons.SPOUSE, "Spouse")
    children = _select_group(df, cons.CHILDREN, "Child")

    head_main = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS]
    spouse_main = spouse[["Household ID"] + [f"Spouse_{c}" for c in cons.INDIVIDUAL_COLS]]
    couple = pd.merge(head_main, spouse_main, on="Household ID", how="inner")

    # Create core reference without duplicating household columns
    core_ref = couple[['Household ID'] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + [f"Spouse_{c}" for c in cons.INDIVIDUAL_COLS]]

    if not children.empty:
        children_combined = children.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original couple dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in couple.columns:
                children_combined[hh_col] = children_combined["Household ID"].map(
                    couple.set_index("Household ID")[hh_col]
                )
        children_combined = children_combined.drop(columns=["Child_ID", "Household ID"])
    else:
        children_combined = None

    couple.drop(columns=["Household ID"], inplace=True)
    return couple, children_combined, None

def transform_couple_no_children_with_others(df: pd.DataFrame):
    head = _select_person(df, "Head", "Head")
    spouse = _select_person(df, cons.SPOUSE, "Spouse")
    others = _select_group(df, cons.OTHER_MEMBERS, "Other")

    head_main = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS]
    spouse_main = spouse[["Household ID"] + [f"Spouse_{c}" for c in cons.INDIVIDUAL_COLS]]
    couple = pd.merge(head_main, spouse_main, on="Household ID", how="inner")

    # Create core reference without duplicating household columns
    core_ref = couple[['Household ID'] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + [f"Spouse_{c}" for c in cons.INDIVIDUAL_COLS]]

    if not others.empty:
        others_combined = others.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original couple dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in couple.columns:
                others_combined[hh_col] = others_combined["Household ID"].map(
                    couple.set_index("Household ID")[hh_col]
                )
        others_combined = others_combined.drop(columns=["Other_ID", "Household ID"])
    else:
        others_combined = None

    couple.drop(columns=["Household ID"], inplace=True)
    return couple, None, others_combined

def transform_single_parent_with_children_and_others(df: pd.DataFrame):
    head = _select_person(df, "Head", "Head")
    children = _select_group(df, cons.CHILDREN, "Child")
    others = _select_group(df, cons.OTHER_MEMBERS, "Other")

    head_main = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS]

    # Create core reference without duplicating household columns
    core_ref = head_main[['Household ID'] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS]]

    if not children.empty:
        children_combined = children.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original head dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in head_main.columns:
                children_combined[hh_col] = children_combined["Household ID"].map(
                    head_main.set_index("Household ID")[hh_col]
                )
        children_combined = children_combined.drop(columns=["Child_ID", "Household ID"])
    else:
        children_combined = None

    if not others.empty:
        others_combined = others.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original head dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in head_main.columns:
                others_combined[hh_col] = others_combined["Household ID"].map(
                    head_main.set_index("Household ID")[hh_col]
                )
        others_combined = others_combined.drop(columns=["Other_ID", "Household ID"])
    else:
        others_combined = None

    head_main.drop(columns=["Household ID"], inplace=True)
    return head_main, children_combined, others_combined

def transform_single_parent_with_children(df: pd.DataFrame):
    head = _select_person(df, "Head", "Head")
    children = _select_group(df, cons.CHILDREN, "Child")

    head_main = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS]

    # Create core reference without duplicating household columns
    core_ref = head_main[['Household ID'] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS]]

    if not children.empty:
        children_combined = children.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original head dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in head_main.columns:
                children_combined[hh_col] = children_combined["Household ID"].map(
                    head_main.set_index("Household ID")[hh_col]
                )
        children_combined = children_combined.drop(columns=["Child_ID", "Household ID"])
    else:
        children_combined = None

    head_main.drop(columns=["Household ID"], inplace=True)
    return head_main, children_combined, None

def transform_head_with_others(df: pd.DataFrame):
    head = _select_person(df, "Head", "Head")
    others = _select_group(df, cons.OTHER_MEMBERS, "Other")

    head_main = head[["Household ID"] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS] + cons.HOUSEHOLD_COLS]

    # Create core reference without duplicating household columns
    core_ref = head_main[['Household ID'] + [f"Head_{c}" for c in cons.INDIVIDUAL_COLS]]

    if not others.empty:
        others_combined = others.merge(core_ref, on="Household ID", how="left")
        # Add household columns from the original head dataframe
        for hh_col in cons.HOUSEHOLD_COLS:
            if hh_col in head_main.columns:
                others_combined[hh_col] = others_combined["Household ID"].map(
                    head_main.set_index("Household ID")[hh_col]
                )
        others_combined = others_combined.drop(columns=["Other_ID", "Household ID"])
    else:
        others_combined = None

    head_main.drop(columns=["Household ID"], inplace=True)
    return head_main, None, others_combined

def transform_other(df: pd.DataFrame):
    # the "Other" bucket is left simple (no explicit roles guaranteed)
    out = df[cons.INDIVIDUAL_COLS + cons.HOUSEHOLD_COLS].copy()
    return out, None, None

def transform_households(df, hh_type_long):
    transformer = TRANSFORM_BY_TYPE[hh_type_long]
    core_df, children_df, others_df = transformer(df)
    print(f"Core households dataframe shape for {hh_type_long}: {core_df.shape}")
    print(f"Core households dataframe head for {hh_type_long}: \n{core_df.head()}")

    if core_df is None or core_df.empty:
        print(f"No core data found for {hh_type_long}")
        return pd.DataFrame()

    print(f"Core households: {len(core_df)}")
    if children_df is not None:
        print(f"Children records dataframe shape for {hh_type_long}: {children_df.shape}")
        print(f"Children records dataframe head for {hh_type_long}: \n{children_df.head()}")

    if others_df is not None:
        print(f"Others records dataframe shape for {hh_type_long}: {others_df.shape}")
        print(f"Others records dataframe head for {hh_type_long}: \n{others_df.head()}")


    return core_df, children_df, others_df

def compute_member_counts(df: pd.DataFrame) -> pd.DataFrame:
    # the function computes, per household, the number of children and other members in the real data
    counts = (
        df.groupby("Household ID")
        .apply(lambda g: pd.Series({
            "num_child": g["Relationship to head"].isin(cons.CHILDREN).sum(),
            "num_other": g["Relationship to head"].isin(cons.OTHER_MEMBERS).sum()
        }))
        .reset_index()
    )
    return counts

TRANSFORM_BY_TYPE = {
    "Single": transform_single,
    "Couple with no children": transform_couple_no_children,
    "Couple with children": transform_couple_with_children,
    "Couple with children and Other household members": transform_couple_with_children_and_others,
    "Couple with no children and Other household members": transform_couple_no_children_with_others,
    "Single parent with children": transform_single_parent_with_children,
    "Single parent with children and Other household members": transform_single_parent_with_children_and_others,
    "Head with Other household members": transform_head_with_others,
    "Other": transform_other,
}

def execute(context):
    df_households = context.stage("data.dhs.households")

    encoding_legend = cons.ENCODING_LEGEND

    df_hh_enc = helpers.encode_categorical_columns(df_households, encoding_legend)

    df = df_hh_enc[cons.REQUIRED_COLUMNS].copy()
    df.rename(columns=cons.RENAME_COLUMNS, inplace=True)
    # cast to categorical dtype as required by pgmpy
    cat_cols = cons.INDIVIDUAL_COLS + cons.HOUSEHOLD_COLS
    for c in cat_cols:
        df[c] = df[c].astype("category")

    #process household types
    bn_data = {}
    counts_real_all = {}
    for hh_type_long in df['Household type'].unique():
        df_type = df[df['Household type'] == hh_type_long].copy()
        print(f"Processing {hh_type_long} with {len(df_type)} households")
        


        core_df, children_df, others_df = transform_households(df_type, hh_type_long)
        bn_data[hh_type_long] = {
            'core': core_df,
            'children': children_df if children_df is not None else pd.DataFrame(),
            'others': others_df if others_df is not None else pd.DataFrame()
        }
        #counts are needed to attach required counts of children / others later using real empirical distribution
        counts_real = compute_member_counts(df_type)
        counts_real_all[hh_type_long] = counts_real

    return bn_data, counts_real_all