import pandas as pd
import tqdm
import numpy as np
import geopandas as gpd
from typing import List, Tuple, Dict, Optional
from raking.synthesis import PopulationSynthesis

from utils.logging_config import setup_logger 



"""
This stage ..
"""


def configure(context):
    context.stage("data.census.filtered")
    context.stage("data.hts.selected")

    # remove default value
    context.config("random_seed")#, 42)
    context.config("output_path")



def execute(context):

    census_df = context.stage("data.census.filtered")
    hts_df = context.stage("data.hts.selected")

    zones_list =census_df["commune_id"].values # define a generalisable level later

    # Variable Mappings
    # These must match columns in HTS and columns in Census
    
    # Define which variables are Household level vs Person level
    # This is CRITICAL for the algorithm to calculate sums correctly
    household_vars = ['household_size', 'num_veh_household'] #move this to config?

    # Initialize Engine
    synthesizer = PopulationSynthesis(
        max_iterations=300,
        tolerance=1e-5,
        household_id_col='household_id',
        person_id_col='person_id',
        household_vars=household_vars,
        verbose=False
    )

    final_population_list = []

    print(f"Starting Synthesis for {len(zones_list)} zones...")

    for zone_id in tqdm(zones_list):
        
        # Prepare Census Targets for this Zone
        # We need a dictionary: {'age': {1: 50, 2: 30...}, 'household_size': {1: 20...}}
        if zone_id not in census_df.index:
            print(f"Skipping Zone {zone_id}: No Census Data")
            continue
            
        zone_census = census_df.loc[zone_id]
        
        census_targets = {}
        
        # ToDo -  need to adapt this to proper column names
        
        # Household Size Targets
        # Map HTS value (1, 2, 3...) to Census Column Name ('hh_1', 'hh_2'...)
        census_targets['household_size'] = {
            1: zone_census['census_hh_size_1'],
            2: zone_census['census_hh_size_2'],
            3: zone_census['census_hh_size_3'],
            4: zone_census['census_hh_size_4_plus'] # Assuming you recoded HTS 4+ to 4
        }
        
        # 2. Employment Targets
        census_targets['employed'] = {
            1: zone_census['census_emp_yes'],
            0: zone_census['census_emp_no']
        }
        
        # 3. Vehicle Targets
        census_targets['num_veh_household'] = {
            0: zone_census['census_veh_0'],
            1: zone_census['census_veh_1'],
            2: zone_census['census_veh_2_plus']
        }

        # ---  Prepare Seed Data ---
        # OPTION 1: Use Global Seed (Recommended for small zones)
        # We use the entire HTS as the pool, but we reset weights to the initial survey weights
        seed_pool = hts_df.copy()
        
        # OPTION 2: Use Local Seed (Only if HTS is large enough per zone)
        # seed_pool = hts_df[hts_df['zone_id'] == zone_id].copy()
        
        # Remove Seed rows that have categories not present in targets (Data cleaning)
        # (The engine handles this via masks, but cleaner data helps)
        
        # --- Run IPU ---
        try:
            weighted_df = synthesizer.ipu_raking(
                df=seed_pool, 
                census_targets=census_targets,
                initial_weight_col='household_weight' # USES HTS WEIGHTS
            )
            
            # --- D. Integerize (Create Synthetic Population) ---
            final_df = synthesizer.integerize_weights(weighted_df)
            
            # Add Zone ID to result
            final_df['final_zone_id'] = zone_id
            
            final_population_list.append(final_df)
            
        except Exception as e:
            print(f"Failed execution for Zone {zone_id}: {str(e)}")

    # Combine all zones
    if final_population_list:
        full_synthetic_pop = pd.concat(final_population_list, ignore_index=True)
        return full_synthetic_pop
    else:
        return pd.DataFrame()
