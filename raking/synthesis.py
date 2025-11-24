import pandas as pd
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class PopulationSynthesis:
    """
    Enhanced Population Synthesis Pipeline with Household Structure Preservation
    """

    def __init__(self,
                 max_iterations: int, # = 300,
                 tolerance: float, # = 1e-5,
                 random_seed: Optional[int], # = 42,
                 verbose: bool, #  = True,
                 household_id_col: str = 'household_id',
                 person_id_col: str = 'person_id',
                 household_vars: List[str] = None ):

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.household_id_col = household_id_col
        self.person_id_col = person_id_col

        self.household_vars = household_vars if household_vars else []

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Dictionary to store execution time for each step
        self._performance_stats = {}

    def _log_performance(self, step_name: str, start_time: float):
        """Record and optionally print execution time for a pipeline step."""
        self._performance_stats[step_name] = time.time() - start_time
        if self.verbose:
            print(f"Time for {step_name}: {self._performance_stats[step_name]:.2f} seconds")

    def _validate_inputs(self, df: pd.DataFrame, census_targets: Dict[str, Dict[str, int]]):
        """Enhanced validation including household structure checks."""
        if df.empty:
            raise ValueError("Input dataframe is empty!")

        if self.household_id_col not in df.columns: raise ValueError(f"HH ID {self.household_id_col} missing.")

        if not census_targets:
            raise ValueError("Census targets dictionary is empty!")

        # Check for missing columns
        missing_vars = [var for var in census_targets.keys() if var not in df.columns]
        if missing_vars:
            raise ValueError(f"Missing variables in dataframe: {missing_vars}")

    def _analyze_household_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze the original household structure to preserve it during synthesis."""
        household_analysis = {}

        # Group by household
        household_groups = df.groupby(self.household_id_col)

        # Calculate household sizes
        household_sizes = household_groups.size()
        household_analysis['size_distribution'] = household_sizes.value_counts().sort_index()
        household_analysis['avg_size'] = household_sizes.mean()

        if self.verbose:
            print(f"\nHousehold Structure Analysis:")
            print(f"   Total households: {len(household_sizes):,}")
            print(f"   Average household size: {household_analysis['avg_size']:.2f}")
            print(f"   Size distribution: {dict(household_analysis['size_distribution'])}")

        return household_analysis

    def _check_missing_categories(self, df: pd.DataFrame,
                                  census_targets: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
        """Identify categories in census targets that are missing in the synthetic pool."""
        missing = defaultdict(list)

        for var, target_cats in census_targets.items():
            if var not in df.columns:
                missing[var] = list(target_cats.keys())
                continue

            present_cats = set(df[var].dropna().unique())
            missing_cats = set(target_cats.keys()) - present_cats

            if missing_cats:
                missing[var] = list(missing_cats)

        return dict(missing)

    def _create_category_masks(self, df: pd.DataFrame, census_targets: Dict) -> Dict:
        """Pre-compute boolean masks for each category of each variable."""
        masks = {}

        for var, categories in census_targets.items():
            if var not in df.columns:
                continue

            masks[var] = {}
            var_data = df[var].values

            for category in categories.keys():
                masks[var][category] = (var_data == category)

        return masks

    def ipu_raking(self,
                   df: pd.DataFrame,
                   census_targets: Dict[str, Dict[str, int]],
                   initial_weight_col: str = 'household_weight') -> pd.DataFrame:
        """
        Iterative Proportional Updating (IPU)
        """
        start_time = time.time()
        self._validate_inputs(df, census_targets)
        
        # Working copy
        df_work = df.copy()
        
        # Initialize weights using HTS weights
        # We assume every person in the HH has the same HH weight in the HTS
        if initial_weight_col in df_work.columns:
            # Normalize initial weights to match total target households (optional, but helps convergence)
            # Find a household var to get total target HHs
            total_target_hh = 0
            for var in self.household_vars:
                if var in census_targets:
                    total_target_hh = sum(census_targets[var].values())
                    break
            
            # Ensure unique weights per household are aligned
            hh_weights = df_work.groupby(self.household_id_col)[initial_weight_col].first()
            current_total_hh = hh_weights.sum()
            
            scale_factor = (total_target_hh / current_total_hh) if (total_target_hh > 0 and current_total_hh > 0) else 1.0
            
            # Map aligned weights back to persons
            df_work['weight'] = df_work[self.household_id_col].map(hh_weights) * scale_factor
        else:
            df_work['weight'] = 1.0

        # Precompute masks for speed
        masks = self._create_category_masks(df_work, census_targets)
        
        # Convert to numpy for performance
        weights = df_work['weight'].values
        # We need a way to quickly identify unique households to avoid over-summing HH weights
        # Create a mapping of row_index -> household_id to facilitate HH-level aggregation?
        # A simpler way in the loop is to handle HH vars differently.

        print(f"Starting IPU. Max Iter: {self.max_iterations}. Household Vars: {self.household_vars}")

        for iteration in range(1, self.max_iterations + 1):
            max_adjustment = 0.0
            
            for var_name, target_categories in census_targets.items():
                if var_name not in masks: continue
                
                is_household_var = var_name in self.household_vars

                for category, target_count in target_categories.items():
                    if category not in masks[var_name]: continue
                    
                    mask = masks[var_name][category]
                    
                    # --- IPU LOGIC ---
                    if is_household_var:
                        # For HH variables, we sum the weight of unique households
                        # Fast way: subset the dataframe, drop duplicates, sum weights
                        # Since all members have same weight, we can sum all weights and divide by HH size
                        # assuming 'household_size' column exists and is accurate
                        if 'household_size' in df_work.columns:
                            # Weighted Persons / HH Size = Weighted Households
                            # This avoids the slow drop_duplicates
                            current_sum = (weights[mask] / df_work.loc[mask, 'household_size'].values).sum()
                        else:
                            # Fallback (slower)
                            subset_ids = df_work.loc[mask, self.household_id_col]
                            unique_mask = ~subset_ids.duplicated()
                            current_sum = weights[mask][unique_mask].sum()
                    else:
                        # Person variable: sum of all person weights
                        current_sum = weights[mask].sum()

                    if current_sum == 0: continue

                    adjustment_factor = target_count / current_sum
                    
                    # Update weights if deviation is significant
                    if abs(adjustment_factor - 1.0) > 1e-7:
                        
                        # Apply adjustment to ALL members of the affected households
                        # Identify affected households (all members of HHs that have the attribute)
                        # NOTE: In standard IPU, if a person causes a shift, the whole HH shifts.
                        
                        affected_hh_ids = df_work.loc[mask, self.household_id_col].unique()
                        
                        # This implies we need to update weights for everyone in these HHs
                        # This step is the computational bottleneck. 
                        # Optimization: Use isin() mask on the whole array
                        affected_rows_mask = df_work[self.household_id_col].isin(affected_hh_ids)
                        weights[affected_rows_mask] *= adjustment_factor
                        
                        max_adjustment = max(max_adjustment, abs(adjustment_factor - 1.0))

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: max adj {max_adjustment:.6f}")

            if max_adjustment < self.tolerance:
                if self.verbose: print(f"Converged at iteration {iteration}")
                break
        
        df_work['weight'] = weights
        return df_work

    def generalized_raking(self,
                           df: pd.DataFrame,
                           census_targets: Dict[str, Dict[str, int]],
                           area_id: Optional[str] = None) -> pd.DataFrame:
        """Household-aware generalized raking that maintains household structures."""
        start_time = time.time()

        # Validate inputs
        self._validate_inputs(df, census_targets)

        # Analyze household structure
        household_analysis = self._analyze_household_structure(df)

        # Create working copy
        df_work = df.copy()

        # Initialize household-level weights
        household_weights = pd.Series(1.0, index=df_work[self.household_id_col].unique())
        df_work['weight'] = df_work[self.household_id_col].map(household_weights)

        if self.verbose:
            area_msg = f" for {area_id}" if area_id else ""
            # Total target population calculation (sums the first category to get this value)
            total_target = sum(census_targets.get(list(census_targets.keys())[0], {}).values())
            print(f"\nStarting household-aware raking{area_msg}...")
            print(f"Total target population: {total_target:,}")

        # Warn about missing categories
        missing = self._check_missing_categories(df_work, census_targets)
        for var, missing_cats in missing.items():
            if missing_cats:
                warnings.warn(f"Variable '{var}' missing categories: {missing_cats}")

        # Precompute masks
        masks = self._create_category_masks(df_work, census_targets)

        # Use numpy array for speed
        weights = df_work['weight'].values

        # Iterative adjustment loop
        for iteration in range(1, self.max_iterations + 1):
            max_adjustment = 0.0

            for var_name, target_categories in census_targets.items():
                if var_name not in masks:
                    continue

                target_counts = target_categories.values()

                for category, target_count in target_categories.items():
                    if category not in masks[var_name]:
                        continue

                    mask = masks[var_name][category]
                    current_sum = weights[mask].sum()

                    if current_sum == 0:
                        continue

                    adjustment_factor = target_count / current_sum

                    # Apply adjustment to entire households
                    affected_households = df_work.loc[mask, self.household_id_col].unique()
                    
                    ## RECOMMEND
                    # affected_households_mask = df_work[self.household_id_col].isin(affected_households)
                    # weights[affected_households_mask] *= adjustment_factor

                    for hh_id in affected_households:
                         hh_mask = df_work[self.household_id_col] == hh_id
                         weights[hh_mask] *= adjustment_factor

                    # Track largest change
                    max_adjustment = max(max_adjustment, abs(adjustment_factor - 1.0))

            # Show progress occasionally
            if self.verbose and (iteration <= 5 or iteration % 25 == 0):
                print(f"  Iteration {iteration:3d}: max adjustment = {max_adjustment:.6f}")

            # Check convergence
            if max_adjustment < self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
        else:
            warnings.warn(f"Raking did not converge within {self.max_iterations} iterations.")

        df_work['weight'] = weights
        self._log_performance("Household-Aware Raking", start_time)
        return df_work

    def integerize_weights(self, weighted_df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic Truncate-Replicate-Sample (TRS) to convert fractional weights to integer households.
        """
        # 1. Collapse to Household Level
        hh_data = weighted_df.groupby(self.household_id_col).agg({
            'weight': 'first',
            'household_size': 'first' # ensure this exists
        }).reset_index()
        
        # 2. Split integer and fractional parts
        hh_data['int_weight'] = np.floor(hh_data['weight']).astype(int)
        hh_data['frac_weight'] = hh_data['weight'] - hh_data['int_weight']
        
        final_households = []
        
        # 3. Probabilistic sampling for fractional parts
        # We simply treat frac_weight as probability to be selected
        random_vals = np.random.random(len(hh_data))
        hh_data['selected'] = random_vals < hh_data['frac_weight']
        hh_data['final_count'] = hh_data['int_weight'] + hh_data['selected'].astype(int)
        
        # 4. Expansion
        # Create a map of HH_ID -> Count
        expansion_map = hh_data.set_index(self.household_id_col)['final_count'].to_dict()
        
        # Filter 0 counts
        valid_hhs = {k: v for k, v in expansion_map.items() if v > 0}
        
        # 5. Reconstruct Population
        # Filter original DF to only valid households
        df_valid = weighted_df[weighted_df[self.household_id_col].isin(valid_hhs.keys())].copy()
        
        # Repeat rows based on count
        # This acts like numpy.repeat
        df_valid['repeat_count'] = df_valid[self.household_id_col].map(valid_hhs)
        final_df = df_valid.loc[df_valid.index.repeat(df_valid['repeat_count'])].copy()
        
        # Assign new unique IDs
        # Group by original HH ID and add counter
        final_df['new_hh_id'] = final_df[self.household_id_col].astype(str) + "_" + \
                                final_df.groupby([self.household_id_col, self.person_id_col]).cumcount().astype(str)
                                
        # Reset weights to 1
        final_df['weight'] = 1.0
        
        return final_df
        
    def household_aware_trs(self, weighted_df: pd.DataFrame) -> pd.DataFrame:
        """Household-aware TRS that replicates entire households as units."""
        start_time = time.time()

        if self.verbose:
            print(f"\nConverting household weights to integers...")

        # Group by household and calculate household-level statistics
        household_stats = weighted_df.groupby(self.household_id_col).agg({
            'weight': ['first', 'mean'],
            self.household_id_col: 'size'
        }).round(6)

        # Flatten column names
        household_stats.columns = ['weight', 'avg_weight', 'size']
        household_stats['household_id'] = household_stats.index

        if self.verbose:
            print(f"   Starting households: {len(household_stats):,}")
            print(f"   Target households: {household_stats['weight'].sum():,.0f}")

        # Apply TRS to households
        household_stats['int_weight'] = np.floor(household_stats['weight']).astype(int)
        household_stats['fractional_weight'] = household_stats['weight'] - household_stats['int_weight']

        # Replicate households deterministically
        replicated_households = []

        # create household id to int_weight mapping
        hh_id_to_weight = household_stats.set_index('household_id')['int_weight']
        # add int_weight value to each individual
        weighted_df['int_weight'] = weighted_df[self.household_id_col].map(hh_id_to_weight)
        # replicate all individuals in same household according to int_weight
        replicated_df = weighted_df.loc[weighted_df.index.repeat(weighted_df['int_weight'])]
        # create unique rep_num for unique household_id
        replicated_df['rep_num'] = replicated_df.groupby(self.household_id_col).cumcount()
        # create new household id by appending rep_num
        replicated_df[self.household_id_col] = (
            replicated_df[self.household_id_col] + 
            '_rep_' + 
            replicated_df['rep_num'].astype(str).str.zfill(4)
        )
        # driop helper columns
        replicated_df = replicated_df.drop(columns=['int_weight', 'rep_num'])

        # Handle fractional weights probabilistically
        target_households = int(round(household_stats['weight'].sum()))
        current_households = len(replicated_households)
        additional_needed = max(0, target_households - current_households)

        if additional_needed > 0:
            has_fractional = household_stats['fractional_weight'] > 0
            eligible_households = household_stats[has_fractional]

            if len(eligible_households) > 0:
                total_fractional = eligible_households['fractional_weight'].sum()
                sample_probs = eligible_households['fractional_weight'] / total_fractional

                n_sample = min(additional_needed, len(eligible_households))
                sampled_hh_ids = np.random.choice(
                    eligible_households.index,
                    size=n_sample,
                    replace=False,
                    p=sample_probs
                )

                for hh_id in sampled_hh_ids:
                    household_members = weighted_df[weighted_df[self.household_id_col] == hh_id].copy()
                    new_hh_id = f"{hh_id}_prob_{len(replicated_households):04d}"
                    household_members[self.household_id_col] = new_hh_id
                    replicated_households.append(household_members)

        # Combine all replicated households
        if replicated_households:
            final_population = pd.concat(replicated_households, ignore_index=True)
        else:
            final_population = pd.DataFrame(columns=weighted_df.columns)

        # Clean up and assign final properties
        final_population['weight'] = 1.0
        final_population['Person ID'] = [f'person_{i:08d}' for i in range(len(final_population))]

        # Calculate household sizes for validation
        final_hh_sizes = final_population.groupby(self.household_id_col).size()

        if self.verbose:
            print(f"   Final households: {len(final_hh_sizes):,}")
            print(f"   Final population: {len(final_population):,}")
            print(f"   Average household size: {final_hh_sizes.mean():.2f}")

        self._log_performance("Household-Aware TRS", start_time)
        return final_population

    def run_pipeline(self,
                     df: pd.DataFrame,
                     census_targets: Dict[str, Dict[str, int]],
                     method: str = "ipu",
                     area_id: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the enhanced household-aware synthesis pipeline."""
        pipeline_start = time.time()

        print("\nStarting Household-Aware Population Synthesis")
        print("=" * 60)
        print(type(df))
        print(f"Loaded {len(df):,} records from {len(df[self.household_id_col].unique()):,} households")

        if raking_method == "ipu":
            weighted_df = self.ipu_raking(df, census_targets)
            final_df = self.integerize_weights(self, weighted_df)
            return weighted_df, final_df
        # Step 2: Household-aware raking
        weighted_df = self.generalized_raking(df, census_targets, area_id)

       # Step 3: Household-aware TRS
        final_df = self.household_aware_trs(weighted_df)

        print(f"Household-aware pipeline complete in {time.time() - pipeline_start:.2f}s")
        return weighted_df, final_df

    def diagnostic_check(self,
                         df: pd.DataFrame,
                         census_targets: Dict[str, Dict[str, int]],
                         stage: str = "Post-processing") -> None:
        """Enhanced diagnostic check including household structure validation."""
        print(f"\nHousehold-Aware Diagnostic Check ({stage})")
        print("=" * 60)

        # Individual-level diagnostics
        for var_name, target_cats in census_targets.items():
            if var_name not in df.columns:
                print(f"Variable '{var_name}' missing in data")
                continue

            print(f"\n=== {var_name.upper()} ===")
            abs_errors = []

            for category, target_val in target_cats.items():
                actual_val = (df[df[var_name] == category]['weight'].sum()
                              if 'weight' in df.columns
                              else (df[var_name] == category).sum())

                abs_err = abs(target_val - actual_val)
                rel_err = abs_err / target_val * 100 if target_val != 0 else 0
                abs_errors.append(rel_err)

                print(f"{category:<15} Target: {target_val:10,.0f}  "
                      f"Actual: {actual_val:10,.0f}  "
                      f"AbsErr: {abs_err:6,.0f}  RelErr%: {rel_err:5.2f}")

            overall_abs_error_pct = np.mean(abs_errors) if abs_errors else 0
            print(f"Overall Abs Error (rounded): {overall_abs_error_pct:.2f}")