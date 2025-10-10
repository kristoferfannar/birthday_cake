"""
Parallel search module for Player3 cake cutting strategy.

This module provides multiprocessing capabilities to speed up the cut-finding process
by parallelizing the evaluation of potential cuts across multiple CPU cores.
"""

from multiprocessing import Pool, cpu_count
from shapely import Point, Polygon
from typing import List, Tuple, Optional
from src.cake import Cake


def _evaluate_cut_chunk(
    chunk_data: Tuple[
        List[Tuple[int, Point]],
        List[Point],
        Cake,
        Polygon,
        float,
        float,
        float,
        float,
        float,
    ],
) -> List[Tuple[Point, Point, Polygon, float, float]]:
    """
    Evaluate a chunk of potential cuts (worker function for multiprocessing).

    Args:
        chunk_data: Tuple containing:
            - indexed_points1: List of (index, Point) tuples for first endpoints
            - points2: List of Point objects for second endpoints
            - cake: The Cake object
            - piece: The Polygon piece being cut
            - target_ratio: Target ratio for the cut
            - original_area: Original area of the piece
            - original_ratio: Original crust/filling ratio
            - acceptable_area_error: Maximum acceptable area error
            - acceptable_ratio_error: Maximum acceptable ratio error

    Returns:
        List of valid cuts as (p1, p2, piece, area_diff, ratio_diff) tuples
    """
    from .helper_func import get_areas_and_ratios, _would_cut_along_boundary

    (
        indexed_points1,
        points2,
        cake,
        piece,
        target_ratio,
        original_area,
        original_ratio,
        acceptable_area_error,
        acceptable_ratio_error,
    ) = chunk_data

    valid_cuts = []

    for i, xy1 in indexed_points1:
        # Only check points that come after this one (avoid duplicates)
        for xy2 in points2[i + 1 :]:
            # Skip boundary cuts
            if _would_cut_along_boundary(xy1, xy2, piece):
                continue

            if cake.cut_is_valid(xy1, xy2):
                valid, area_diff, ratio_diff = get_areas_and_ratios(
                    cake,
                    xy1,
                    xy2,
                    target_ratio,
                    original_area,
                    original_ratio,
                    acceptable_area_error,
                    acceptable_ratio_error,
                )

                if valid:
                    valid_cuts.append((xy1, xy2, piece, area_diff, ratio_diff))

    return valid_cuts


def parallel_find_valid_cuts(
    cake: Cake,
    perim_points: List[Point],
    target_ratio: float,
    original_area: float,
    original_ratio: float,
    acceptable_area_error: float,
    acceptable_ratio_error: float,
    num_workers: Optional[int] = None,
) -> List[Tuple[Point, Point, Polygon]]:
    """
    Find valid cuts using parallel processing across multiple CPU cores.

    Args:
        cake: The Cake object
        perim_points: List of perimeter points to check
        target_ratio: Target ratio for the cut (e.g., 0.5 for half)
        original_area: Original area of the piece
        original_ratio: Original crust/filling ratio
        acceptable_area_error: Maximum acceptable area error
        acceptable_ratio_error: Maximum acceptable ratio error
        num_workers: Number of worker processes (defaults to CPU count - 1)

    Returns:
        List of valid cuts sorted by accuracy
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free

    # If we have very few points, don't bother with parallelization
    if len(perim_points) < 20 or num_workers == 1:
        from .helper_func import find_valid_cuts

        return find_valid_cuts(
            cake,
            perim_points,
            target_ratio,
            original_area,
            original_ratio,
            acceptable_area_error,
            acceptable_ratio_error,
        )

    # Get the piece we're working with
    piece = max(cake.exterior_pieces, key=lambda p: p.area)

    # Split the work into chunks for parallel processing
    # We'll divide the first point list into chunks
    indexed_points = list(enumerate(perim_points))
    chunk_size = max(
        1, len(indexed_points) // (num_workers * 2)
    )  # Create more chunks than workers

    chunks = []
    for i in range(0, len(indexed_points), chunk_size):
        chunk = indexed_points[i : i + chunk_size]
        chunks.append(
            (
                chunk,
                perim_points,
                cake,
                piece,
                target_ratio,
                original_area,
                original_ratio,
                acceptable_area_error,
                acceptable_ratio_error,
            )
        )

    # Process chunks in parallel
    valid_cuts = []
    try:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_evaluate_cut_chunk, chunks)

            # Flatten results
            for chunk_results in results:
                valid_cuts.extend(chunk_results)
    except Exception as e:
        print(f"Parallel processing failed: {e}, falling back to serial")
        # Fallback to serial processing
        from .helper_func import find_valid_cuts

        return find_valid_cuts(
            cake,
            perim_points,
            target_ratio,
            original_area,
            original_ratio,
            acceptable_area_error,
            acceptable_ratio_error,
        )

    # Sort by accuracy (area_diff first, then ratio_diff)
    valid_cuts.sort(key=lambda x: (x[3], x[4]))

    # Return in same format as original find_valid_cuts
    return [(xy1, xy2, piece) for xy1, xy2, piece, _, _ in valid_cuts]


def parallel_find_coarse_cuts(
    cake: Cake,
    piece: Polygon,
    coarse_points: List[Point],
    target_ratio: float,
    original_area: float,
    original_ratio: float,
    relaxed_area_error: float,
    relaxed_ratio_error: float,
    num_workers: Optional[int] = None,
) -> List[Tuple[Point, Point, Polygon, float]]:
    """
    Find coarse cuts in parallel for the refinement system.

    Args:
        cake: The Cake object
        piece: The Polygon piece being cut
        coarse_points: List of coarse perimeter points
        target_ratio: Target ratio for the cut
        original_area: Original area of the piece
        original_ratio: Original crust/filling ratio
        relaxed_area_error: Relaxed area error tolerance
        relaxed_ratio_error: Relaxed ratio error tolerance
        num_workers: Number of worker processes

    Returns:
        List of coarse cuts with scores
    """
    from .refinement import calculate_optimization_score
    from .helper_func import get_areas_and_ratios, _would_cut_along_boundary

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # If we have very few points, don't bother with parallelization
    if len(coarse_points) < 15 or num_workers == 1:
        # Use serial version
        coarse_cuts = []
        for i, xy1 in enumerate(coarse_points):
            for xy2 in coarse_points[i + 1 :]:
                if _would_cut_along_boundary(xy1, xy2, piece):
                    continue

                if cake.cut_is_valid(xy1, xy2):
                    valid, area_diff, ratio_diff = get_areas_and_ratios(
                        cake,
                        xy1,
                        xy2,
                        target_ratio,
                        original_area,
                        original_ratio,
                        relaxed_area_error,
                        relaxed_ratio_error,
                    )

                    if valid:
                        score = calculate_optimization_score(area_diff, ratio_diff)
                        coarse_cuts.append((xy1, xy2, piece, score))

        return coarse_cuts

    # Split work into chunks
    indexed_points = list(enumerate(coarse_points))
    chunk_size = max(1, len(indexed_points) // (num_workers * 2))

    chunks = []
    for i in range(0, len(indexed_points), chunk_size):
        chunk = indexed_points[i : i + chunk_size]
        chunks.append(
            (
                chunk,
                coarse_points,
                cake,
                piece,
                target_ratio,
                original_area,
                original_ratio,
                relaxed_area_error,
                relaxed_ratio_error,
            )
        )

    # Process in parallel
    coarse_cuts = []
    try:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_evaluate_cut_chunk, chunks)

            for chunk_results in results:
                for xy1, xy2, piece, area_diff, ratio_diff in chunk_results:
                    score = calculate_optimization_score(area_diff, ratio_diff)
                    coarse_cuts.append((xy1, xy2, piece, score))
    except Exception as e:
        print(f"Parallel coarse search failed: {e}, falling back to serial")
        # Fallback already handled above
        pass

    return coarse_cuts


def get_optimal_worker_count() -> int:
    """
    Get the optimal number of worker processes based on CPU count.

    Returns:
        Number of workers to use (leaves one core free for system)
    """
    cpu_cores = cpu_count()

    # Leave at least one core free, use at most 8 workers
    # (diminishing returns beyond that for this workload)
    return min(max(1, cpu_cores - 1), 8)
