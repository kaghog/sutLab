import pandas as pd
import tqdm
import numpy as np
import geopandas as gpd
from typing import List, Tuple, Dict, Optional

from synthesis.model.params import Config as config
import synthesis.model.bayesian.model as bn
from synthesis.model.bayesian.model import BayesianNetwork
import synthesis.prepare.helpers as helpers
from synthesis.model.raking.synthesis import PopulationSynthesis

from utils.logging_config import setup_logger 



"""
This stage ..
"""

raking_population_logger = setup_logger("raking_population_logger", "raking.log", console=False)

def configure(context):
    #context.stage("synthesis.model.bayesian.output")
    context.stage("synthesis.model.raking.preprocess")
    cons = context.stage("utils.constants")

    # remove default value
    context.config("random_seed")#, 42)
    context.config("target_region")#, None)
    context.config("run_all_regions")#, False) 
    context.config("output_path")
    context.config("output_prefix")#, "base_alchemy")
    context.config("list_of_marginals")

    context.config("regions")

    #context.stage("data.dhs.households")



def execute(context):
    #synthetic_data = context.stage("synthesis.model.bayesian.pool")[0]

    #census_df = process_census(context)
    census_df, df_dhs = context.stage("synthesis.model.raking.preprocess")
    regions = context.config("regions")


    
    # build census targets from census dataframe
    synpop = pd.DataFrame()
    census_targets = {}
    
    for dhs_col in [col for col in df_dhs.columns if "_census" in col]:
        census_cols = df_dhs[dhs_col].unique().tolist()
        tmp = {}
        #print(dhs_col, census_cols)
        for c_col in census_cols:
            count = census_df[c_col].sum()
            tmp[c_col] = count
        census_targets[dhs_col] = tmp

    # Extract census targets from the census dataframe
    # census_targets = extract_census_targets_from_dataframe(
    #     census_df=census_df,
    #     target_columns=config.GR_TARGET_COLUMNS,
    #     verbose=True
    # )
    # if not census_targets:
    #     raking_population_logger.info(f"No census targets extracted for {region}. Skipping GR post-processing.")
    #     return synthetic_data

    raking_population_logger.info(f"\n=== Applying GR Post-Processing for {regions} ===")
    # do synthesis with GR model
    gr_engine = PopulationSynthesis(
        max_iterations=config.GR_MAX_ITERATIONS,
        tolerance=config.GR_TOLERANCE,
        random_seed=context.config("random_seed"),
        verbose=True,
        household_id_col='Household ID'
    )
    
    weighted_df, final_df = gr_engine.run_pipeline(
        df=df_dhs,
        census_targets=census_targets,
        area_id= regions
    )

    # Conduct diagnostic checks
    raking_population_logger.info("\nConducting Diagnostic Checks")
    raking_population_logger.info("=" * 40)

    gr_engine.diagnostic_check(weighted_df, census_targets, stage="After Raking")
    gr_engine.diagnostic_check(final_df, census_targets, stage="Final Population")

    raking_population_logger.info(f"\nGR Results Summary for {regions}:")
    raking_population_logger.info(f"   - Original households: {len(weighted_df['Household ID'].unique()):,}")
    raking_population_logger.info(f"   - Final households: {len(final_df['Household ID'].unique()):,}")
    raking_population_logger.info(f"   - Original individuals: {len(weighted_df):,}")
    raking_population_logger.info(f"   - Final individuals: {len(final_df):,}")
    raking_population_logger.info(f"   - Scale factor: {len(final_df) / len(weighted_df):.2f}x")
    synpop = pd.concat([synpop, final_df], ignore_index=True)

        
    
    return synpop