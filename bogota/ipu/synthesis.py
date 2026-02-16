import random
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PopulationSynthesis:
    """
    Enhanced Population Synthesis Pipeline with Household Structure Preservation
    """

    def __init__(
        self,
        max_iterations: int,  # = 300,
        tolerance: float,  # = 1e-5,
        random_seed: Optional[int],  # = 42,
        verbose: bool,  #  = True,
        household_id_col: str = "household_id",
        person_id_col: str = "person_id",
        household_vars: List[str] = None,
    ):
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
            print(
                f"Time for {step_name}: {self._performance_stats[step_name]:.2f} seconds"
            )

    def _validate_inputs(
        self, df: pd.DataFrame, census_targets: Dict[str, Dict[str, int]]
    ):
        """Enhanced validation including household structure checks."""
        if df.empty:
            raise ValueError("Input dataframe is empty!")

        if self.household_id_col not in df.columns:
            raise ValueError(f"HH ID {self.household_id_col} missing.")

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
        household_analysis["size_distribution"] = (
            household_sizes.value_counts().sort_index()
        )
        household_analysis["avg_size"] = household_sizes.mean()

        if self.verbose:
            print("\nHousehold Structure Analysis:")
            print(f"   Total households: {len(household_sizes):,}")
            print(f"   Average household size: {household_analysis['avg_size']:.2f}")
            print(
                f"   Size distribution: {dict(household_analysis['size_distribution'])}"
            )

        return household_analysis

    def _check_missing_categories(
        self, df: pd.DataFrame, census_targets: Dict[str, Dict[str, int]]
    ) -> Dict[str, List[str]]:
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

            for category in categories.keys():
                masks[var][category] = (df[var] == category).values

        return masks

    def ipu_raking(
        self,
        df: pd.DataFrame,
        census_targets: Dict[str, Dict[str, int]],
        initial_weight_col: str = "household_weight",
    ) -> pd.DataFrame:
        """
        Iterative Proportional Updating (IPU)
        """
        self._validate_inputs(df, census_targets)

        df_work = df.copy()

        if initial_weight_col in df_work.columns:
            total_target_hh = 0
            for var in self.household_vars:
                if var in census_targets:
                    total_target_hh = sum(census_targets[var].values())
                    break

            hh_weights = df_work.groupby(self.household_id_col)[
                initial_weight_col
            ].first()
            current_total_hh = hh_weights.sum()

            scale_factor = (
                (total_target_hh / current_total_hh)
                if (total_target_hh > 0 and current_total_hh > 0)
                else 1.0
            )

            df_work["weight"] = (
                df_work[self.household_id_col].map(hh_weights) * scale_factor
            )
        else:
            df_work["weight"] = 1.0

        masks = self._create_category_masks(df_work, census_targets)
        weights = df_work["weight"].values

        if self.verbose:
            print(
                f"Starting IPU. Max Iter: {self.max_iterations}. Household Vars: {self.household_vars}"
            )

        for iteration in range(1, self.max_iterations + 1):
            max_adjustment = 0.0

            for var_name, target_categories in census_targets.items():
                if var_name not in masks:
                    continue

                is_household_var = var_name in self.household_vars

                for category, target_count in target_categories.items():
                    if category not in masks[var_name]:
                        continue

                    mask = masks[var_name][category]

                    if is_household_var:
                        subset_ids = df_work.loc[mask, self.household_id_col]
                        unique_mask = ~subset_ids.duplicated()
                        current_sum = weights[mask][unique_mask].sum()
                    else:
                        current_sum = weights[mask].sum()

                    if current_sum == 0:
                        continue

                    adjustment_factor = target_count / current_sum

                    if abs(adjustment_factor - 1.0) > 1e-7:
                        if is_household_var:
                            affected_hh_ids = df_work.loc[
                                mask, self.household_id_col
                            ].unique()
                            affected_rows_mask = df_work[self.household_id_col].isin(
                                affected_hh_ids
                            )
                            weights[affected_rows_mask] *= adjustment_factor
                        else:
                            weights[mask] *= adjustment_factor

                        max_adjustment = max(
                            max_adjustment, abs(adjustment_factor - 1.0)
                        )

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: max adj {max_adjustment:.6f}")

            if max_adjustment < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

        df_work["weight"] = weights
        return df_work

    def generalized_raking(
        self,
        df: pd.DataFrame,
        census_targets: Dict[str, Dict[str, int]],
        area_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Household-aware generalized raking that maintains household structures."""
        start_time = time.time()

        # Validate inputs
        self._validate_inputs(df, census_targets)

        # Create working copy
        df_work = df.copy()

        # Initialize household-level weights
        household_weights = pd.Series(
            1.0, index=df_work[self.household_id_col].unique()
        )
        df_work["weight"] = df_work[self.household_id_col].map(household_weights)

        if self.verbose:
            area_msg = f" for {area_id}" if area_id else ""
            # Total target population calculation (sums the first category to get this value)
            total_target = sum(
                census_targets.get(list(census_targets.keys())[0], {}).values()
            )
            print(f"\nStarting household-aware raking{area_msg}...")
            print(f"Total target population: {total_target:,}")

        missing = self._check_missing_categories(df_work, census_targets)
        for var, missing_cats in missing.items():
            if missing_cats:
                warnings.warn(f"Variable '{var}' missing categories: {missing_cats}")

        masks = self._create_category_masks(df_work, census_targets)
        weights = df_work["weight"].values  # Iterative adjustment loop
        for iteration in range(1, self.max_iterations + 1):
            max_adjustment = 0.0

            for var_name, target_categories in census_targets.items():
                if var_name not in masks:
                    continue

                for category, target_count in target_categories.items():
                    if category not in masks[var_name]:
                        continue

                    mask = masks[var_name][category]
                    current_sum = weights[mask].sum()

                    if current_sum == 0:
                        continue

                    adjustment_factor = target_count / current_sum

                    is_household_var = var_name in self.household_vars

                    if is_household_var:
                        # Household-level: Apply adjustment to all members of affected households
                        affected_households = df_work.loc[
                            mask, self.household_id_col
                        ].unique()
                        affected_households_mask = df_work[self.household_id_col].isin(
                            affected_households
                        )
                        weights[affected_households_mask] *= adjustment_factor
                    else:
                        # Person-level: Apply adjustment ONLY to matching persons
                        weights[mask] *= adjustment_factor

                    # Track largest change
                    max_adjustment = max(max_adjustment, abs(adjustment_factor - 1.0))

            if self.verbose and (iteration <= 5 or iteration % 25 == 0):
                print(
                    f"  Iteration {iteration:3d}: max adjustment = {max_adjustment:.6f}"
                )

            if max_adjustment < self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
        else:
            warnings.warn(
                f"Raking did not converge within {self.max_iterations} iterations."
            )

        df_work["weight"] = weights
        self._log_performance("Household-Aware Raking", start_time)
        return df_work

    def integerize_weights(self, weighted_df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic Truncate-Replicate-Sample (TRS) to convert fractional weights to integer households.
        """
        hh_data = (
            weighted_df.groupby(self.household_id_col)
            .agg(
                {
                    "weight": "mean",
                    "household_size": "first",
                }
            )
            .reset_index()
        )

        hh_data["int_weight"] = np.floor(hh_data["weight"]).astype(int)
        hh_data["frac_weight"] = hh_data["weight"] - hh_data["int_weight"]

        random_vals = np.random.random(len(hh_data))
        hh_data["selected"] = random_vals < hh_data["frac_weight"]
        hh_data["final_count"] = hh_data["int_weight"] + hh_data["selected"].astype(int)

        expansion_map = hh_data.set_index(self.household_id_col)[
            "final_count"
        ].to_dict()

        valid_hhs = {k: v for k, v in expansion_map.items() if v > 0}

        df_valid = weighted_df[
            weighted_df[self.household_id_col].isin(valid_hhs.keys())
        ].copy()

        df_valid["repeat_count"] = df_valid[self.household_id_col].map(valid_hhs)
        final_df = df_valid.loc[df_valid.index.repeat(df_valid["repeat_count"])].copy()

        final_df["new_hh_id"] = (
            final_df[self.household_id_col].astype(str)
            + "_"
            + final_df.groupby([self.household_id_col, self.person_id_col])
            .cumcount()
            .astype(str)
        )

        final_df["weight"] = 1.0
        return final_df

    def household_aware_trs(self, weighted_df: pd.DataFrame) -> pd.DataFrame:
        """Household-aware TRS that replicates entire households as units."""
        start_time = time.time()

        if self.verbose:
            print("\nConverting household weights to integers...")

        household_stats = (
            weighted_df.groupby(self.household_id_col)
            .agg({"weight": ["first", "mean"], self.household_id_col: "size"})
            .round(6)
        )

        household_stats.columns = ["weight", "avg_weight", "size"]
        household_stats["household_id"] = household_stats.index

        if self.verbose:
            print(f"   Starting households: {len(household_stats):,}")
            print(f"   Target households: {household_stats['weight'].sum():,.0f}")

        household_stats["int_weight"] = np.floor(household_stats["weight"]).astype(int)
        household_stats["fractional_weight"] = (
            household_stats["weight"] - household_stats["int_weight"]
        )

        replicated_households = []

        hh_id_to_weight = household_stats.set_index("household_id")["int_weight"]
        weighted_df["int_weight"] = weighted_df[self.household_id_col].map(
            hh_id_to_weight
        )
        replicated_df = weighted_df.loc[
            weighted_df.index.repeat(weighted_df["int_weight"])
        ]
        replicated_df["rep_num"] = replicated_df.groupby(
            self.household_id_col
        ).cumcount()
        replicated_df[self.household_id_col] = (
            replicated_df[self.household_id_col]
            + "_rep_"
            + replicated_df["rep_num"].astype(str).str.zfill(4)
        )
        replicated_df = replicated_df.drop(columns=["int_weight", "rep_num"])

        target_households = int(round(household_stats["weight"].sum()))
        current_households = len(replicated_households)
        additional_needed = max(0, target_households - current_households)

        if additional_needed > 0:
            has_fractional = household_stats["fractional_weight"] > 0
            eligible_households = household_stats[has_fractional]

            if len(eligible_households) > 0:
                total_fractional = eligible_households["fractional_weight"].sum()
                sample_probs = (
                    eligible_households["fractional_weight"] / total_fractional
                )

                n_sample = min(additional_needed, len(eligible_households))
                sampled_hh_ids = np.random.choice(
                    eligible_households.index,
                    size=n_sample,
                    replace=False,
                    p=sample_probs,
                )

                for hh_id in sampled_hh_ids:
                    household_members = weighted_df[
                        weighted_df[self.household_id_col] == hh_id
                    ].copy()
                    new_hh_id = f"{hh_id}_prob_{len(replicated_households):04d}"
                    household_members[self.household_id_col] = new_hh_id
                    replicated_households.append(household_members)

        if replicated_households:
            final_population = pd.concat(replicated_households, ignore_index=True)
        else:
            final_population = pd.DataFrame(columns=weighted_df.columns)

        final_population["weight"] = 1.0
        final_population["Person ID"] = [
            f"person_{i:08d}" for i in range(len(final_population))
        ]

        final_hh_sizes = final_population.groupby(self.household_id_col).size()

        if self.verbose:
            print(f"   Final households: {len(final_hh_sizes):,}")
            print(f"   Final population: {len(final_population):,}")
            print(f"   Average household size: {final_hh_sizes.mean():.2f}")

        self._log_performance("Household-Aware TRS", start_time)
        return final_population

    def run_pipeline(
        self,
        df: pd.DataFrame,
        census_targets: Dict[str, Dict[str, int]],
        method: str = "ipu",
        area_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the enhanced household-aware synthesis pipeline."""
        pipeline_start = time.time()

        print("\nStarting Household-Aware Population Synthesis")
        print("=" * 60)
        print(type(df))
        print(
            f"Loaded {len(df):,} records from {len(df[self.household_id_col].unique()):,} households"
        )

        if method == "ipu":
            weighted_df = self.ipu_raking(df, census_targets)
            final_df = self.integerize_weights(weighted_df)
            return weighted_df, final_df

        weighted_df = self.generalized_raking(df, census_targets, area_id)
        final_df = self.household_aware_trs(weighted_df)

        print(
            f"Household-aware pipeline complete in {time.time() - pipeline_start:.2f}s"
        )
        return weighted_df, final_df

    def diagnostic_check(
        self,
        df: pd.DataFrame,
        census_targets: Dict[str, Dict[str, int]],
        stage: str = "Post-processing",
    ) -> None:
        """Enhanced diagnostic check including household structure validation."""
        print(f"\nHousehold-Aware Diagnostic Check ({stage})")
        print("=" * 60)

        for var_name, target_cats in census_targets.items():
            if var_name not in df.columns:
                print(f"Variable '{var_name}' missing in data")
                continue

            print(f"\n=== {var_name.upper()} ===")
            abs_errors = []

            for category, target_val in target_cats.items():
                actual_val = (
                    df[df[var_name] == category]["weight"].sum()
                    if "weight" in df.columns
                    else (df[var_name] == category).sum()
                )

                abs_err = abs(target_val - actual_val)
                rel_err = abs_err / target_val * 100 if target_val != 0 else 0
                abs_errors.append(rel_err)

                print(
                    f"{category:<15} Target: {target_val:10,.0f}  "
                    f"Actual: {actual_val:10,.0f}  "
                    f"AbsErr: {abs_err:6,.0f}  RelErr%: {rel_err:5.2f}"
                )

            overall_abs_error_pct = np.mean(abs_errors) if abs_errors else 0
            print(f"Overall Abs Error (rounded): {overall_abs_error_pct:.2f}")
