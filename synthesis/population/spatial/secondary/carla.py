"""
CARLA (Context-Aware Recursive Location Assignment) algorithm
for secondary location assignment.

Based on the reference implementation from MATSimPipeline.
Adapted to work with the Eqasim pipeline structure.

This is a near-exact copy of the reference implementation with minimal changes
only where necessary for Eqasim data structure compatibility.
"""

import numpy as np
import pandas as pd
import shapely.geometry as geo
import geopandas as gpd
from typing import Tuple
from scipy.spatial import cKDTree
from collections import namedtuple

from synthesis.population.spatial.secondary.problems import find_assignment_problems

# Named tuple for representing legs (matching reference implementation)
Leg = namedtuple('Leg', ['unique_leg_id', 'from_location', 'to_location', 'distance', 'to_act_type', 'to_act_identifier'])


class Helpers:
    """Helper functions copied from reference implementation."""
    
    @staticmethod
    def euclidean_distance(start: np.ndarray, end: np.ndarray) -> float:
        """Compute the Euclidean distance between two points."""
        return np.linalg.norm(end - start)
    
    @staticmethod
    def get_min_max_distance(arr):
        """Get the minimum and maximum possible distance/radius (from a fixed point) given a list of distances."""
        if len(arr) == 0:
            raise ValueError("No distances given.")
        if len(arr) == 1:
            return arr[0], arr[0]
        
        arr = np.array(arr, dtype=int)
        total_sum = sum(arr)
        
        # Is one leg longer than all others summed?
        remaining_distances = total_sum - arr
        single_leg_overshoot = max(arr - remaining_distances)
        min_diff = max(single_leg_overshoot, 0)
        
        return min_diff, total_sum
    
    @staticmethod
    def spread_distances(distance1, distance2, iteration=0, first_step=20, base=1.5):
        """Increases the difference between two distances, keeping them positive."""
        step = first_step * (base ** iteration)
        if distance1 > distance2:
            distance1 += step
            distance2 -= step
        else:
            distance1 -= step
            distance2 += step
        return max(0, distance1), max(0, distance2)
    
    @staticmethod
    def get_abs_distance_deviations(candidate_coordinates, location, wanted_distance):
        """Calculate absolute distance deviations from wanted distance."""
        # Handle single-coordinate case by reshaping
        if candidate_coordinates.ndim == 1:  # Single coordinate (1D array)
            candidate_coordinates = candidate_coordinates[np.newaxis, :]  # Make it 2D
        
        # Calculate distances
        candidate_distances = np.linalg.norm(candidate_coordinates - location, axis=1)
        return np.abs(candidate_distances - wanted_distance)


h = Helpers()


class EvaluationFunction:
    
    @staticmethod
    def evaluate_candidates(potentials: np.ndarray = None, dist_deviations: np.ndarray = None,
                            number_of_candidates: int = None) -> np.ndarray:
        """
        Scoring function collection for the candidates based on potentials and distances.

        :param potentials: Numpy array of potentials for the returned locations.
        :param dist_deviations: Distance deviations from the target (if available).
        :param number_of_candidates:
        :return: Non-normalized, absolute scores.
        """
        if dist_deviations is not None:
            return np.maximum(0, 1000000 - dist_deviations)
        else:
            if number_of_candidates is None:
                return np.full((len(potentials),), 1000000)
            return np.full((number_of_candidates,), 1000000)
    
    @classmethod
    def select_candidate_indices(
            cls,
            scores: np.ndarray,
            num_candidates: int,
            strategy: str = 'monte_carlo',
            top_portion: float = 0.5,
            coords: np.ndarray = None,
            num_cells_x: int = 20,
            num_cells_y: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the indices of candidates based on their normalized scores using Monte Carlo sampling,
        a top-n strategy, a mixed strategy, or spatial downsampling.

        :param scores: A 1D numpy array of scores corresponding to candidates.
        :param num_candidates: The number of candidates to select.
        :param strategy: Selection strategy ('monte_carlo', 'top_n', 'mixed', or 'spatial_downsample').
        :param top_portion: Portion of candidates to select from the top scores when using the 'mixed' strategy.
        :param coords: 2D numpy array of shape (n, 2) with candidate spatial coordinates (required for spatial_downsample).
        :param num_cells_x: Number of cells along the longitude (spatial_downsample).
        :param num_cells_y: Number of cells along the latitude (spatial_downsample).
        :return: A tuple containing:
                 - The selected indices of the best candidates.
                 - A 1D array of the scores corresponding to the selected indices.
        """
        assert len(scores) > 0, "The scores array cannot be empty."
        if num_candidates >= len(scores):
            return np.arange(len(scores)), scores

        if strategy == 'monte_carlo':
            normalized_scores = scores / np.sum(scores, dtype=np.float64)
            chosen_indices = np.random.choice(len(scores), num_candidates, p=normalized_scores, replace=False)

        elif strategy == 'top_n':
            chosen_indices = np.argsort(scores)[-num_candidates:][::-1]  # Top scores in descending order

        elif strategy == 'mixed':
            num_top = int(np.ceil(num_candidates * top_portion))
            num_monte_carlo = num_candidates - num_top

            sorted_indices = np.argsort(scores)[-num_top:][::-1]
            remaining_indices = np.setdiff1d(np.arange(len(scores)), sorted_indices)

            if len(remaining_indices) > 0 and num_monte_carlo > 0:
                remaining_scores = scores[remaining_indices]
                normalized_remaining_scores = remaining_scores / np.sum(remaining_scores, dtype=np.float64)
                monte_carlo_indices = np.random.choice(remaining_indices, num_monte_carlo,
                                                       p=normalized_remaining_scores, replace=False)
                chosen_indices = np.concatenate((sorted_indices, monte_carlo_indices))
            else:
                chosen_indices = sorted_indices

        elif strategy == 'spatial_downsample':
            assert coords is not None, "Coordinates (coords) are required for spatial_downsample strategy."
            chosen_indices = cls.even_spatial_downsample(
                coords, num_cells_x=num_cells_x, num_cells_y=num_cells_y
            )[:num_candidates]

        elif strategy == 'top_n_spatial_downsample':
            assert coords is not None, "Coordinates (coords) are required for top_n_spatial_downsample strategy."

            # Sort scores in descending order
            sorted_indices = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_indices]

            # Identify the cutoff score
            cutoff_score = sorted_scores[num_candidates - 1] if len(sorted_scores) >= num_candidates else sorted_scores[
                -1]

            # Find all indices with scores >= cutoff_score (this may be more than num_candidates if scores are equal)
            top_indices = np.where(scores >= cutoff_score)[0]

            # Check if spatial downsampling is needed
            if len(top_indices) > num_candidates:
                num_cells = max(1, int(np.sqrt(num_candidates)) + 1)  # Slightly above the square root of candidates
                chosen_indices = cls.even_spatial_downsample(
                    coords, num_cells_x=num_cells, num_cells_y=num_cells
                )
            else:
                # Use the sorted indices if no downsampling is needed
                chosen_indices = sorted_indices[:num_candidates]

        else:
            raise ValueError(
                "Invalid selection strategy. Use 'monte_carlo', 'top_n', 'mixed', or 'spatial_downsample'.")

        chosen_scores = scores[chosen_indices]
        return chosen_indices, chosen_scores
    
    @classmethod
    def select_candidates(
            cls,
            candidates: Tuple[np.ndarray, ...],
            scores: np.ndarray,
            num_candidates: int,
            strategy: str = 'monte_carlo',
            top_portion: float = 0.5,
            coords: np.ndarray = None,
            num_cells_x: int = 20,
            num_cells_y: int = 20
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """
        Selects a specified number of candidates based on their scores using various strategies.

        :param candidates: A tuple of arrays with the candidates.
        :param scores: A 1D array of scores corresponding to the candidates.
        :param num_candidates: The number of candidates to select.
        :param strategy: Selection strategy ('monte_carlo', 'top_n', 'mixed', or 'spatial_downsample').
        :param top_portion: Portion of candidates to select from the top scores when using the 'mixed' strategy.
        :param coords: 2D numpy array of candidate spatial coordinates (required for 'spatial_downsample'). If no
                        coordinates are provided, candidates[1] is used as coordinates.
        :param num_cells_x: Number of cells along the longitude (spatial_downsample).
        :param num_cells_y: Number of cells along the latitude (spatial_downsample).
        :return: A tuple containing:
            - A tuple of arrays with the selected candidates.
            - A 1D array of the scores corresponding to the selected candidates.
        """
        assert len(candidates[0]) == len(scores), "The number of candidates and scores must match."
        if strategy == 'keep_all':
            return candidates, scores
        if (strategy == 'spatial_downsample' or strategy == "top_n_spatial_downsample") and coords is None:
            coords = candidates[1]

        chosen_indices, chosen_scores = cls.select_candidate_indices(
            scores, num_candidates, strategy, top_portion, coords, num_cells_x, num_cells_y
        )

        selected_candidates = tuple(
            np.atleast_1d(arr[chosen_indices].squeeze()) if arr is not None else None for arr in candidates
        )

        if num_candidates == 1:
            return (
                tuple(
                    (
                        np.atleast_1d(selected_candidates[0]),  # IDs (n,)
                        np.atleast_2d(selected_candidates[1]),  # Coordinates (n, 2)
                        np.atleast_1d(selected_candidates[2]),  # Potentials (n,)
                    )
                ),
                np.atleast_1d(chosen_scores)  # Scores (n,)
            )

        return selected_candidates, chosen_scores
    
    @staticmethod
    def even_spatial_downsample(coords, num_cells_x=20, num_cells_y=20):
        """
        Downsample points and return indices of the kept points.

        Parameters:
        - coords: 2D coordinates array (n, 2)
        - num_cells_x: Number of cells along the longitude.
        - num_cells_y: Number of cells along the latitude.

        Returns:
        - A list of indices of the points that are kept after downsampling.
        """
        lats = coords[:, 0]
        lons = coords[:, 1]

        min_lat, max_lat = lats.min(), lats.max()
        min_lon, max_lon = lons.min(), lons.max()

        lat_range = max_lat - min_lat or 1e-9
        lon_range = max_lon - min_lon or 1e-9

        lat_step = lat_range / max(num_cells_y, 1)
        lon_step = lon_range / max(num_cells_x, 1)

        total_cells = num_cells_x * num_cells_y
        filled_cells = set()
        kept_indices = []

        for i in range(len(coords)):
            lat, lon = lats[i], lons[i]
            cell_x = min(int((lon - min_lon) / lon_step), num_cells_x - 1)
            cell_y = min(int((lat - min_lat) / lat_step), num_cells_y - 1)
            cell_id = cell_y * num_cells_x + cell_x

            if cell_id not in filled_cells:
                kept_indices.append(i)
                filled_cells.add(cell_id)

            # Stop early if all cells are filled
            if len(filled_cells) == total_cells:
                break

        return kept_indices


class TargetLocations:
    """
    Spatial index of activity locations split by type.
    This class is used to quickly find the nearest activity locations for a given location.
    Renamed from CandidateIndex to match reference implementation naming.
    """
    
    def __init__(self, data):
        self.data = data
        self.indices = {}
        
        for purpose, purpose_data in self.data.items():
            print("Constructing spatial index for %s ..." % purpose)
            self.indices[purpose] = cKDTree(purpose_data["locations"])
    
    def query_closest(self, type: str, location: np.ndarray, num_candidates: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest activity locations for one or more points.
        
        Parameters
        ----------
        type : str
            The activity type to query (e.g., 'shop', 'leisure', 'other')
        location : np.ndarray
            A 1D array for a single point or 2D array for multiple points
        num_candidates : int
            Number of nearest candidates to return
        
        Returns
        -------
        identifiers : np.ndarray
            IDs of the nearest candidates
        coordinates : np.ndarray
            Coordinates of the nearest candidates
        potentials : np.ndarray
            Potential values (currently set to 1.0)
        """
        # Query the KDTree
        _, indices = self.indices[type].query(location, k=num_candidates)
        
        # Retrieve data for the nearest candidates
        purpose_data = self.data[type]
        return (
            purpose_data["identifiers"][indices],
            purpose_data["locations"][indices],
            np.ones(np.atleast_1d(indices).shape)  # Placeholder potentials
        )
    
    def query_within_two_overlapping_rings(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                          radius1a: float, radius1b: float, 
                                          radius2a: float, radius2b: float,
                                          max_number_of_candidates: int = None):
        """
        Find candidates within two overlapping annuli (rings).
        
        Returns candidates that are:
        - Within outer_radius1 but outside inner_radius1 from location1 AND
        - Within outer_radius2 but outside inner_radius2 from location2
        """
        location1 = location1.reshape(1, -1) if location1.ndim == 1 else location1
        location2 = location2.reshape(1, -1) if location2.ndim == 1 else location2
        
        outer_radius1, inner_radius1 = max(radius1a, radius1b), min(radius1a, radius1b)
        outer_radius2, inner_radius2 = max(radius2a, radius2b), min(radius2a, radius2b)
        
        tree = self.indices[act_type]
        outer_indices1 = tree.query_ball_point(location1, outer_radius1)[0]
        outer_indices2 = tree.query_ball_point(location2, outer_radius2)[0]
        
        if not outer_indices1 or not outer_indices2:
            return None
        
        outer_intersection = set(outer_indices1).intersection(outer_indices2)
        if not outer_intersection:
            return None
        
        inner_indices1 = set(tree.query_ball_point(location1, inner_radius1)[0])
        inner_indices2 = set(tree.query_ball_point(location2, inner_radius2)[0])
        
        overlapping_indices = list(outer_intersection - (inner_indices1.union(inner_indices2)))
        if not overlapping_indices:
            return None
        
        if max_number_of_candidates and len(overlapping_indices) > max_number_of_candidates:
            overlapping_indices = np.random.choice(overlapping_indices, max_number_of_candidates, replace=False)
        
        purpose_data = self.data[act_type]
        overlapping_indices = np.array(overlapping_indices)
        candidate_identifiers = purpose_data["identifiers"][overlapping_indices]
        candidate_coordinates = purpose_data["locations"][overlapping_indices]
        candidate_potentials = np.ones(len(overlapping_indices))  # Placeholder
        
        return candidate_identifiers, candidate_coordinates, candidate_potentials
    
    def find_overlapping_rings_candidates(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                          radius1a: float, radius1b: float,
                                          radius2a: float, radius2b: float,
                                          min_candidates=1, max_candidates=None, max_iterations=15):
        """
        Find candidates within two overlapping rings (donuts) around two center points.
        Iteratively increase the radii until a sufficient number of candidates is found.
        """
        i = 0
        while True:
            candidates = self.query_within_two_overlapping_rings(
                act_type, location1, location2, radius1a, radius1b, radius2a, radius2b, max_candidates)
            if candidates is not None and len(candidates[0]) >= min_candidates:
                return candidates, i
            radius1a, radius1b = h.spread_distances(radius1a, radius1b, iteration=i, first_step=50)
            radius2a, radius2b = h.spread_distances(radius2a, radius2b, iteration=i, first_step=50)
            i += 1
            if i > max_iterations:
                raise RuntimeError(f"Not enough candidates found after {max_iterations} iterations.")


class CircleIntersection:
    """Handles geometric calculations for finding activity locations."""
    
    def __init__(self, target_locations: TargetLocations):
        self.target_locations = target_locations
    
    def find_circle_intersections(self, center1: np.ndarray, radius1: float, 
                                 center2: np.ndarray, radius2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the intersection points of two circles.
        
        Returns
        -------
        intersect1, intersect2 : np.ndarray or None
            The intersection points. If only one intersection, intersect2 is None.
        """
        x1, y1 = center1
        x2, y2 = center2
        r1 = radius1
        r2 = radius2
        
        # Distance between centers
        d = np.linalg.norm(center2 - center1)
        
        # Non-intersection cases
        if d < 1e-4:
            # Same center - return a point on the circle
            intersect = np.array([x1 + r1, y1])
            return intersect, None
        
        if d > (r1 + r2):
            # Circles too far apart - return point on line between them
            proportional_distance = r1 / (r1 + r2)
            point_on_line = center1 + proportional_distance * (center2 - center1)
            return point_on_line, None
        
        if d < abs(r1 - r2):
            # One circle inside the other
            if r1 > r2:
                larger_center, larger_radius = center1, r1
                smaller_center, smaller_radius = center2, r2
            else:
                larger_center, larger_radius = center2, r2
                smaller_center, smaller_radius = center1, r1
            
            proportional_distance = (d + smaller_radius + 0.5 * (larger_radius - smaller_radius - d)) / d
            midpoint = larger_center + proportional_distance * (smaller_center - larger_center)
            return midpoint, None
        
        # Calculate actual intersection points
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(max(0, r1**2 - a**2))
        
        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d
        
        if h < 1e-6:
            # Tangent circles
            return np.array([x3, y3]), None
        
        intersect1 = np.array([x3 + h * (y2 - y1) / d, y3 - h * (x2 - x1) / d])
        intersect2 = np.array([x3 - h * (y2 - y1) / d, y3 + h * (x2 - x1) / d])
        
        return intersect1, intersect2
    
    def find_circle_intersection_candidates(self, start_coord: np.ndarray, end_coord: np.ndarray, type: str,
                                            distance_start_to_act: float, distance_act_to_end: float,
                                            num_candidates: int):
        """Find candidates near circle intersection points."""
        intersect1, intersect2 = self.find_circle_intersections(
            start_coord, distance_start_to_act, end_coord, distance_act_to_end
        )
        
        # Query candidates near intersection points
        if intersect2 is not None:
            locations_to_query = np.array([intersect1, intersect2])
            candidate_ids, candidate_coords, candidate_potentials = self.target_locations.query_closest(
                type, locations_to_query, num_candidates
            )
            if num_candidates > 1:
                candidate_ids = np.concatenate(candidate_ids, axis=0)
                candidate_coords = np.concatenate(candidate_coords, axis=0)
                candidate_potentials = np.concatenate(candidate_potentials, axis=0)
        else:
            candidate_ids, candidate_coords, candidate_potentials = self.target_locations.query_closest(
                type, intersect1, num_candidates
            )
            if num_candidates == 1:
                candidate_ids = np.atleast_1d(candidate_ids)
                candidate_coords = np.atleast_2d(candidate_coords)
                candidate_potentials = np.atleast_1d(candidate_potentials)
        
        return candidate_ids, candidate_coords, candidate_potentials
    
    def get_best_circle_intersection_location(self, start_coord: np.ndarray, end_coord: np.ndarray, act_type: str,
                                              distance_start_to_act: float, distance_act_to_end: float,
                                              num_circle_intersection_candidates=None, selection_strategy='top_n',
                                              max_iterations=15, only_return_valid=False):
        """
        Place a single activity at one of the closest locations.
        Copied from reference implementation.
        
        :param start_coord: Coordinates of the start location.
        :param end_coord: Coordinates of the end location.
        :param act_type: Type of activity (e.g., 'work', 'shopping').
        :param distance_start_to_act: Distance from start location to activity.
        :param distance_act_to_end: Distance from activity to end location.
        :param num_circle_intersection_candidates: Number of candidates to consider.
        :param selection_strategy: Strategy for selecting the best candidate.
        :param max_iterations: Maximum number of iterations for finding candidates.
        :param only_return_valid: If True, only return feasible locations, else None.
        :return: Tuple containing the selected identifier, coordinates, potential, and score.
        """
        # If start and end locations are very close, use fallback
        if h.euclidean_distance(start_coord, end_coord) < 1e-4:
            if only_return_valid and abs(distance_act_to_end - distance_start_to_act) > 10:  # 10m deviation is fine
                return None, None, None, None
            # Fallback: just find nearest location
            radius1, radius2 = h.spread_distances(distance_start_to_act, distance_act_to_end)
            candidate_ids, candidate_coords, candidate_potentials = self.target_locations.query_closest(
                act_type, start_coord, num_circle_intersection_candidates or 1
            )
        else:
            # Find intersection candidates between start and end
            candidate_ids, candidate_coords, candidate_potentials = self.find_circle_intersection_candidates(
                start_coord, end_coord, act_type, distance_start_to_act, distance_act_to_end,
                num_candidates=num_circle_intersection_candidates or 1
            )
            if candidate_ids is None:
                if only_return_valid:
                    return None, None, None, None
                raise RuntimeError("Reached impossible state.")

        # Calculate distance deviations
        distance_deviations = (
                h.get_abs_distance_deviations(candidate_coords, start_coord, distance_start_to_act) +
                h.get_abs_distance_deviations(candidate_coords, end_coord, distance_act_to_end)
        )

        # Evaluate and select the best candidate
        scores = EvaluationFunction.evaluate_candidates(candidate_potentials, distance_deviations)
        best_index = EvaluationFunction.select_candidate_indices(scores, 1, selection_strategy)[0]

        # Extract the selected candidate's data
        best_id = candidate_ids[best_index][0]
        best_coord = candidate_coords[best_index][0]
        best_potential = candidate_potentials[best_index][0]
        best_score = scores[best_index][0]

        return best_id, best_coord, best_potential, best_score


def get_min_max_distance(distances: np.ndarray) -> Tuple[float, float]:
    """Calculate minimum and maximum possible distance given a chain of leg distances."""
    if len(distances) == 0:
        raise ValueError("No distances given.")
    if len(distances) == 1:
        return distances[0], distances[0]
    
    total_sum = np.sum(distances)
    remaining_distances = total_sum - distances
    single_leg_overshoot = np.max(distances - remaining_distances)
    min_distance = max(single_leg_overshoot, 0)
    
    return min_distance, total_sum


def spread_distances(distance1: float, distance2: float, iteration: int = 0, 
                    first_step: float = 50, base: float = 1.5) -> Tuple[float, float]:
    """Increase the difference between two distances, keeping them positive."""
    step = first_step * (base ** iteration)
    if distance1 > distance2:
        distance1 += step
        distance2 -= step
    else:
        distance1 -= step
        distance2 += step
    return max(0, distance1), max(0, distance2)


class CARLA:
    """CARLA algorithm - copied from reference implementation."""
    
    def __init__(self, target_locations: TargetLocations,
                 distance_distributions: dict = None, random: np.random.RandomState = None):
        self.target_locations = target_locations
        self.c_i = CircleIntersection(target_locations)
        self.distance_distributions = distance_distributions
        self.random = random
        self.visualizer = None  # Not using visualizer in Eqasim pipeline
        
        # Configuration parameters hardcoded
        self.number_of_branches = 10
        self.min_candidates_complex_case = 10
        self.candidates_two_leg_case = 30
        self.max_candidates = None
        self.anchor_strategy = "lower_middle"
        self.selection_strategy_complex_case = "top_n_spatial_downsample"
        self.selection_strategy_two_leg_case = "top_n"
        self.max_radius_reduction_factor = None
        self.max_iterations_complex_case = 15
        self.only_return_valid_persons = False
        self.leisure_correction_factor = 2.0
    
    def _get_anchor_index(self, num_legs: int) -> int:
        """Determine the anchor index based on strategy."""
        if self.anchor_strategy == "lower_middle":
            return num_legs // 2 - 1
        elif self.anchor_strategy == "upper_middle":
            return num_legs // 2
        elif self.anchor_strategy == "start":
            return 0
        elif self.anchor_strategy == "end":
            return num_legs - 1
        else:
            raise ValueError("Invalid anchor strategy.")
    
    def solve_segment(self, segment: Tuple[Leg, ...], parent_node=None) -> Tuple[Tuple[Leg, ...], float]:
        """
        Recursively solve a segment for multiple candidates.
        Copied exactly from reference implementation.
        """

        if len(segment) == 0:
            raise ValueError("No legs in segment.")
        elif len(segment) == 1:  # Base case for single leg
            assert segment[0].from_location.size > 0 and segment[0].to_location.size > 0, \
                "Start and end locations must be known."
            return segment, 0  # Score was calculated one lvl higher for single-leg segment
        elif len(segment) == 2:  # Base case for two legs
            best_loc = self.c_i.get_best_circle_intersection_location(
                segment[0].from_location, segment[1].to_location, segment[0].to_act_type,
                segment[0].distance, segment[1].distance, self.candidates_two_leg_case,
                self.selection_strategy_two_leg_case, self.max_iterations_complex_case,
                self.only_return_valid_persons
            )
            if best_loc[0] is None:
                if self.only_return_valid_persons:
                    return None, 0
                raise RuntimeError("Reached impossible state.")
            updated_leg1 = segment[0]._replace(to_location=best_loc[1], to_act_identifier=best_loc[0])
            updated_leg2 = segment[1]._replace(from_location=best_loc[1])
            if self.visualizer:
                label = f"2-leg node: {best_loc[0]}, score: {best_loc[3]:.2f}"
                self.visualizer.add_node(parent_node, label, location=best_loc[1], metadata={"score": best_loc[3]})
            return (updated_leg1, updated_leg2), best_loc[3]  # act_score

        # Recursive case
        anchor_idx = self._get_anchor_index(len(segment))
        location1 = segment[0].from_location
        location2 = segment[-1].to_location
        act_type = segment[anchor_idx].to_act_type

        # Generate candidate locations
        distances = np.array([leg.distance for leg in segment])
        distances_start_to_act = distances[:anchor_idx + 1]  # Up to and including anchor
        distances_act_to_end = distances[anchor_idx + 1:]  # From anchor + 1 to end

        # Radii describing the search area (two overlapping donuts)
        min_possible_distance1, max_possible_distance1 = h.get_min_max_distance(distances_start_to_act)
        min_possible_distance2, max_possible_distance2 = h.get_min_max_distance(distances_act_to_end)

        # Limit the search space, as the maximum radii will almost never be needed in valid trips
        if self.max_radius_reduction_factor:
            min_possible_distance1 *= self.max_radius_reduction_factor
            max_possible_distance1 *= self.max_radius_reduction_factor

        candidates, iterations = self.target_locations.find_overlapping_rings_candidates(
            act_type, location1, location2,
            min_possible_distance1, max_possible_distance1,
            min_possible_distance2, max_possible_distance2,
            self.min_candidates_complex_case, self.max_candidates,
            self.max_iterations_complex_case)
        candidate_ids, candidate_coords, candidate_potentials = candidates

        # Evaluate candidates
        if iterations > 0:  # We need to find distance deviations of each candidate to score them
            candidate_deviations = np.zeros(len(candidate_ids))
            # We only count deviations of lowest-level legs to avoid double counting (!!)
            if len(distances_start_to_act) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location1,
                                                                      distances_start_to_act)
            elif len(distances_act_to_end) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location2,
                                                                      distances_act_to_end)
            local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, candidate_deviations)
        else:  # No distance deviations expected, just score by potentials
            candidate_deviations = np.zeros(len(candidate_ids))
            if len(distances_start_to_act) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location1,
                                                                      distances_start_to_act)
            if len(distances_act_to_end) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location2,
                                                                      distances_act_to_end)
            if np.any(candidate_deviations != 0):
                raise ValueError("Total deviations should be zero.")
            local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, None,
                                                                  len(candidate_ids))

        selected_candidates, selected_scores = EvaluationFunction.select_candidates(
            candidates, local_scores, self.number_of_branches, self.selection_strategy_complex_case
        )

        # Process each candidate and split segments
        full_segs = []
        branch_scores = []
        for i in range(len(selected_candidates[0])):
            new_coord = selected_candidates[1][i]
            new_id = selected_candidates[0][i]

            if self.visualizer:
                candidate_label = f"Candidate {new_id}: Score {selected_scores[i]:.2f}"
                child_node = self.visualizer.add_node(parent_node, candidate_label, location=new_coord, metadata={"score": selected_scores[i]})
            else:
                child_node = None

            # Create updated legs (safe copies, not modifying originals)
            updated_leg1 = segment[anchor_idx]._replace(to_location=new_coord, to_act_identifier=new_id)
            updated_leg2 = segment[anchor_idx + 1]._replace(from_location=new_coord)

            # Split into subsegments with safely updated legs
            subsegment1 = (*segment[:anchor_idx], updated_leg1)
            subsegment2 = (updated_leg2, *segment[anchor_idx + 2:])

            # Recursively solve each subsegment
            located_seg1, score1 = self.solve_segment(subsegment1, child_node)
            located_seg2, score2 = self.solve_segment(subsegment2, child_node)

            if located_seg1 is None or located_seg2 is None:
                if self.only_return_valid_persons:
                    return None, 0
                raise RuntimeError("Reached impossible state.")
            # Combine results and track scores
            total_score = score1 + score2 + selected_scores[i]
            branch_scores.append(total_score)
            full_segs.append((*located_seg1, *located_seg2))

        # Return the best solution
        best_idx = np.argmax(branch_scores)
        return full_segs[best_idx], branch_scores[best_idx]
    
    def solve_problem(self, problem: dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve a single assignment problem.
        Wrapper to convert Eqasim problem format to segment format.
        
        Returns
        -------
        identifiers : np.ndarray
            Location IDs for each activity
        locations : np.ndarray
            Coordinates for each activity
        score : float
            Overall quality score
        """
        segment = self._convert_problem_to_legs(problem)
        result_segment, score = self.solve_segment(segment)
        
        if result_segment is None:
            raise RuntimeError(f"Failed to solve problem for person {problem['person_id']}")
        
        # Extract locations and identifiers from the solved segment
        locations = np.array([leg.to_location for leg in result_segment])
        identifiers = np.array([leg.to_act_identifier for leg in result_segment])
        
        return identifiers, locations, score
    
    def _convert_problem_to_legs(self, problem: dict) -> Tuple[Leg, ...]:
        """
        Convert Eqasim problem format to tuple of Leg namedtuples.
        This is the ONLY method that needs to know about Eqasim's data structure.
        
        Key insight: 
        - problem["purposes"] contains ONLY variable activities (fixed ones removed)
        - problem["modes"]/["travel_times"] contain ALL legs (including to/from fixed activities)
        - We need to reconstruct the full purpose sequence to match the number of legs
        """
        # Sample distances from distributions based on mode and travel time
        if self.distance_distributions is not None and self.random is not None:
            distances = self._sample_distances(problem)
        else:
            # Fallback: estimate distances from travel times (very rough approximation)
            speed_map = {"car": 13.9, "car_passenger": 13.9, "pt": 8.3, "bike": 4.2, "walk": 1.4}  # m/s
            distances = np.array([
                travel_time * speed_map.get(mode, 5.0) 
                for mode, travel_time in zip(problem["modes"], problem["travel_times"])
            ])
        
        # Reconstruct the full purpose sequence
        # problem["purposes"] contains ONLY variable activities (fixed purposes removed)
        # problem["modes"]/["travel_times"] contain ALL legs
        # We need to map each leg to its to_act_type
        
        has_origin = problem["origin"] is not None
        has_destination = problem["destination"] is not None
        
        # Determine the to_act_type for each leg
        to_act_types = []
        
        for i in range(len(distances)):
            # Index into the purposes array needs adjustment based on whether origin is fixed
            # If origin is fixed, first leg goes to purposes[0]
            # If origin is not fixed, purposes start from the first leg
            if has_origin:
                purpose_idx = i
            else:
                purpose_idx = i
            
            if has_destination:
                # Last leg goes to fixed destination (not in purposes list)
                if i == len(distances) - 1:
                    to_act_types.append("fixed_destination")
                elif purpose_idx < len(problem["purposes"]):
                    to_act_types.append(problem["purposes"][purpose_idx])
                else:
                    to_act_types.append("unknown")
            else:
                # No fixed destination, all legs go to variable activities
                if purpose_idx < len(problem["purposes"]):
                    to_act_types.append(problem["purposes"][purpose_idx])
                else:
                    to_act_types.append("unknown")
        
        # Create Leg namedtuples
        legs = []
        for i in range(len(distances)):
            # Determine from_location and to_location
            # Legs always use numpy arrays (either empty or filled), never None
            
            # First leg: from_location is origin (if known)
            if i == 0:
                from_loc = problem["origin"]
                if from_loc is None:
                    from_loc = np.array([])
                else:
                    from_loc = from_loc.flatten()  # Convert (1, 2) -> (2,)
            else:
                from_loc = np.array([])  # Will be filled by algorithm
            
            # Last leg: to_location is destination (if known)
            if i == len(distances) - 1:
                to_loc = problem["destination"]
                if to_loc is None:
                    to_loc = np.array([])
                else:
                    to_loc = to_loc.flatten()  # Convert (1, 2) -> (2,)
            else:
                to_loc = np.array([])  # Will be filled by algorithm
            
            leg = Leg(
                unique_leg_id=f"{problem['person_id']}_{i}",
                from_location=from_loc,
                to_location=to_loc,
                distance=float(distances[i]),
                to_act_type=to_act_types[i],
                to_act_identifier=None
            )
            legs.append(leg)
        
        return tuple(legs)
    
    def _sample_distances(self, problem: dict) -> np.ndarray:
        """Sample distances from distributions based on modes and travel times."""
        distances = np.zeros(len(problem["modes"]))
        
        # Build to_act_types to know which legs go to leisure activities
        has_origin = problem["origin"] is not None
        has_destination = problem["destination"] is not None
        to_act_types = []
        
        for i in range(len(problem["modes"])):
            purpose_idx = i if has_origin else i
            if has_destination and i == len(problem["modes"]) - 1:
                to_act_types.append(None)  # Fixed destination
            elif purpose_idx < len(problem["purposes"]):
                to_act_types.append(problem["purposes"][purpose_idx])
            else:
                to_act_types.append(None)
        
        for index, (mode, travel_time) in enumerate(zip(
            problem["modes"], problem["travel_times"]
        )):
            if mode not in self.distance_distributions:
                # Fallback to speed-based estimate
                speed_map = {"car": 13.9, "car_passenger": 13.9, "pt": 8.3, "bike": 4.2, "walk": 1.4}
                distances[index] = travel_time * speed_map.get(mode, 5.0)
                continue
            
            mode_distribution = self.distance_distributions[mode]
            bound_index = np.count_nonzero(travel_time > mode_distribution["bounds"])
            mode_distribution = mode_distribution["distributions"][bound_index]
            
            distances[index] = mode_distribution["values"][
                np.count_nonzero(self.random.random_sample() > mode_distribution["cdf"])
            ]
            
            # Apply leisure correction if configured
            if to_act_types[index] == "leisure" and self.leisure_correction_factor is not None:
                distances[index] *= self.leisure_correction_factor
        
        return distances


def process_carla(context, arguments):
    """
    CARLA algorithm for secondary location assignment.
    Entry point for the Eqasim pipeline.
    """
    df_trips, df_primary, random_seed, crs = arguments
    
    random = np.random.RandomState(random_seed)

    # Get destinations and distance distributions data
    destinations = context.data("destinations")
    distance_distributions = context.data("distance_distributions")
    target_locations = TargetLocations(destinations)
    
    # Initialize CARLA solver
    carla_solver = CARLA(target_locations, distance_distributions, random)
    
    # Process each assignment problem
    df_locations = []
    df_convergence = []
    
    # Debug counters
    total_problems = 0
    free_chain_problems = 0
    failed_problems = 0
    successful_problems = 0
    
    last_person_id = None
    
    for problem in find_assignment_problems(df_trips, df_primary):
        total_problems += 1
        starting_activity_index = problem["activity_index"]
        
        # CARLA cannot handle free chains (both origin and destination unknown)
        if problem["origin"] is None and problem["destination"] is None:
            free_chain_problems += 1
            # Generate fallback locations by sampling random facilities
            identifiers, locations = _generate_fallback_locations(
                problem, target_locations, random
            )
            
            for index, (identifier, location) in enumerate(zip(identifiers, locations)):
                df_locations.append((
                    problem["person_id"], starting_activity_index + index, identifier, geo.Point(location)
                ))
            
            df_convergence.append((False, problem["size"]))
            
            if problem["person_id"] != last_person_id:
                last_person_id = problem["person_id"]
                context.progress.update()
            continue
        
        try:
            identifiers, locations, score = carla_solver.solve_problem(problem)
            
            for index, (identifier, location) in enumerate(zip(identifiers, locations)):
                df_locations.append((
                    problem["person_id"], starting_activity_index + index, identifier, geo.Point(location)
                ))
            
            df_convergence.append((True, problem["size"]))
            successful_problems += 1
            
        except Exception as e:
            failed_problems += 1
            print(f"[CARLA] Error solving problem for person {problem['person_id']}, size={problem['size']}, purposes={problem['purposes']}: {e}")
            
            # CRITICAL: Generate fallback locations to prevent missing geometry
            identifiers, locations = _generate_fallback_locations(
                problem, target_locations, random
            )
            
            for index, (identifier, location) in enumerate(zip(identifiers, locations)):
                df_locations.append((
                    problem["person_id"], starting_activity_index + index, identifier, geo.Point(location)
                ))
            
            df_convergence.append((False, problem["size"]))
        
        if problem["person_id"] != last_person_id:
            last_person_id = problem["person_id"]
            context.progress.update()
    
    # Debug output
    print(f"\n[CARLA] Processing Summary:")
    print(f"  Total problems: {total_problems}")
    print(f"  Successful: {successful_problems} ({100*successful_problems/max(total_problems,1):.1f}%)")
    print(f"  Free chains (skipped): {free_chain_problems} ({100*free_chain_problems/max(total_problems,1):.1f}%)")
    print(f"  Failed (with fallback): {failed_problems} ({100*failed_problems/max(total_problems,1):.1f}%)")
    print(f"  Total location records generated: {len(df_locations)}")
    
    df_locations = pd.DataFrame.from_records(df_locations, columns=["person_id", "activity_index", "location_id", "geometry"])
    df_locations = gpd.GeoDataFrame(df_locations, crs=crs)
    
    # Validation check
    print(f"[CARLA] Location records shape: {df_locations.shape}")
    print(f"[CARLA] Missing geometries: {df_locations['geometry'].isna().sum()}")
    if df_locations["geometry"].isna().any():
        print(f"[CARLA] WARNING: {df_locations['geometry'].isna().sum()} locations have missing geometry!")
        print(f"[CARLA] Sample of missing geometry records:")
        print(df_locations[df_locations["geometry"].isna()].head(10))
    
    df_convergence = pd.DataFrame.from_records(df_convergence, columns=["valid", "size"])
    
    return df_locations, df_convergence


def _generate_fallback_locations(problem: dict, target_locations: TargetLocations, random: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate fallback locations when CARLA fails to solve a problem.
    Samples random facilities of the appropriate type for each activity.
    
    This ensures every activity gets a location, preventing downstream pipeline failures.
    """
    identifiers = []
    locations = []
    
    for purpose in problem["purposes"]:
        # Get all locations for this purpose type
        purpose_data = target_locations.data.get(purpose)
        
        if purpose_data is None or len(purpose_data["identifiers"]) == 0:
            # Fallback: use 'other' category if specific purpose not available
            purpose_data = target_locations.data.get("other")
            if purpose_data is None or len(purpose_data["identifiers"]) == 0:
                # Last resort: use any available category
                for fallback_purpose in ["shop", "leisure", "other"]:
                    purpose_data = target_locations.data.get(fallback_purpose)
                    if purpose_data is not None and len(purpose_data["identifiers"]) > 0:
                        break
        
        # Sample a random location
        if purpose_data is not None and len(purpose_data["identifiers"]) > 0:
            idx = random.randint(0, len(purpose_data["identifiers"]))
            identifiers.append(purpose_data["identifiers"][idx])
            locations.append(purpose_data["locations"][idx])
        else:
            # Absolute fallback: null island (should never happen)
            print(f"[CARLA] WARNING: No facilities available for purpose '{purpose}', using null island")
            identifiers.append(-999)
            locations.append(np.array([0.0, 0.0]))
    
    return np.array(identifiers), np.array(locations)
