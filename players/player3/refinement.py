from shapely import Point, Polygon
from src.cake import Cake
from .helper_func import get_areas_and_ratios, _would_cut_along_boundary


def calculate_optimization_score(
    area_diff: float | None,
    ratio_diff: float | None,
    area_weight: float = 0.7,
    ratio_weight: float = 0.3,
) -> float:
    """Combined objective function for multi-criteria optimization"""
    # Handle None values by treating them as very poor scores
    if area_diff is None:
        area_diff = float("inf")
    if ratio_diff is None:
        ratio_diff = float("inf")

    return area_weight * area_diff + ratio_weight * ratio_diff


def move_point_along_perimeter(
    point: Point,
    piece: Polygon,
    distance: float,
    direction: int,  # +1 or -1
) -> Point:
    """Move a point along the piece perimeter by a given distance"""
    boundary = piece.exterior
    current_position = boundary.project(point)
    perimeter_length = boundary.length

    # Calculate new position along perimeter
    new_position = current_position + (direction * distance)

    # Handle wrapping around the perimeter
    new_position = new_position % perimeter_length

    return boundary.interpolate(new_position)


def binary_search_endpoint(
    cake: Cake,
    fixed_point: Point,
    moving_point: Point,
    piece: Polygon,
    target_area: float,
    original_ratio: float,
    max_search_distance: float,
    max_iterations: int,
    acceptable_area_error: float,
    acceptable_ratio_error: float,
) -> tuple[Point, float]:
    """
    Binary search along perimeter to optimize one endpoint.
    Returns: (optimized_point, score)
    """
    best_point = moving_point
    best_score = float("inf")

    # Binary search bounds
    left_distance = 0.0
    right_distance = max_search_distance

    for iteration in range(max_iterations):
        # Try both directions from current best point
        for direction in [-1, 1]:
            # Calculate test distance (binary search)
            test_distance = (left_distance + right_distance) / 2

            # Move point along perimeter
            test_point = move_point_along_perimeter(
                moving_point, piece, test_distance, direction
            )

            # Validate the cut
            if not cake.cut_is_valid(fixed_point, test_point):
                continue

            # Skip boundary cuts
            if _would_cut_along_boundary(fixed_point, test_point, piece):
                continue

            # Calculate areas and ratios
            valid, area_diff, ratio_diff = get_areas_and_ratios(
                cake,
                fixed_point,
                test_point,
                0.5,
                target_area,
                original_ratio,
                acceptable_area_error,
                acceptable_ratio_error,
            )

            if valid:
                # This cut meets our tolerance requirements
                score = calculate_optimization_score(area_diff, ratio_diff)
                if score < best_score:
                    best_score = score
                    best_point = test_point
                    return best_point, best_score  # Early return for valid cut

            # Calculate score even for invalid cuts to guide search
            score = calculate_optimization_score(area_diff, ratio_diff)
            if score < best_score:
                best_score = score
                best_point = test_point

                # Adjust search bounds based on direction
                if direction == 1:
                    left_distance = test_distance
                else:
                    right_distance = test_distance
            else:
                # Adjust search bounds in opposite direction
                if direction == 1:
                    right_distance = test_distance
                else:
                    left_distance = test_distance

    return best_point, best_score


def refine_cut_binary_search(
    cake: Cake,
    p1: Point,
    p2: Point,
    piece: Polygon,
    target_area: float,
    original_ratio: float,
    acceptable_area_error: float,
    acceptable_ratio_error: float,
    max_iterations: int = 15,
) -> tuple[Point, Point, float]:
    """
    Refine a cut by moving endpoints along perimeter using binary search.
    Returns: (refined_p1, refined_p2, optimization_score)
    """
    # Calculate maximum search distance (10% of perimeter length)
    max_search_distance = piece.exterior.length * 0.1

    # Try refining both endpoints
    best_p1, best_p2 = p1, p2
    best_score = float("inf")

    # Refine p1 while keeping p2 fixed
    refined_p1, score1 = binary_search_endpoint(
        cake,
        p2,
        p1,
        piece,
        target_area,
        original_ratio,
        max_search_distance,
        max_iterations,
        acceptable_area_error,
        acceptable_ratio_error,
    )

    if score1 < best_score:
        best_score = score1
        best_p1, best_p2 = refined_p1, p2

    # Refine p2 while keeping p1 fixed
    refined_p2, score2 = binary_search_endpoint(
        cake,
        p1,
        p2,
        piece,
        target_area,
        original_ratio,
        max_search_distance,
        max_iterations,
        acceptable_area_error,
        acceptable_ratio_error,
    )

    if score2 < best_score:
        best_score = score2
        best_p1, best_p2 = p1, refined_p2

    # Try refining both endpoints together (one more iteration)
    if best_score < float("inf"):
        final_p1, score_final = binary_search_endpoint(
            cake,
            best_p2,
            best_p1,
            piece,
            target_area,
            original_ratio,
            max_search_distance / 2,
            max_iterations // 2,
            acceptable_area_error,
            acceptable_ratio_error,
        )

        if score_final < best_score:
            best_score = score_final
            best_p1 = final_p1

    return best_p1, best_p2, best_score


def find_valid_cuts_with_refinement(
    cake: Cake,
    piece: Polygon,
    target_ratio: float,
    original_area: float,
    original_ratio: float,
    acceptable_area_error: float,
    acceptable_ratio_error: float,
    coarse_samples: int = 25,
    top_n_to_refine: int = 5,
    use_parallel: bool = False,
    num_workers: int = None,
) -> list[tuple[Point, Point, Polygon]]:
    """
    Two-phase cut finding:
    1. Coarse search with fewer points
    2. Refine top N candidates with binary search
    """
    # Phase 1: Coarse search with relaxed tolerances (stricter area, same ratio)
    relaxed_area_error = (
        acceptable_area_error * 2.5
    )  # 2.5x more lenient for coarse search
    relaxed_ratio_error = (
        acceptable_ratio_error * 3.0
    )  # 3x more lenient for coarse search (ratio is fine)

    # Generate coarse perimeter points
    boundary = piece.exterior
    coarse_points = [
        boundary.interpolate(i / coarse_samples, normalized=True)
        for i in range(coarse_samples)
    ]

    # Find all valid cuts with relaxed tolerances
    coarse_cuts = []
    skipped_boundary_cuts = 0

    # Use parallel search if enabled and we have enough points
    if use_parallel and len(coarse_points) >= 20:
        try:
            from .parallel_search import parallel_find_coarse_cuts

            coarse_cuts = parallel_find_coarse_cuts(
                cake,
                piece,
                coarse_points,
                target_ratio,
                original_area,
                original_ratio,
                relaxed_area_error,
                relaxed_ratio_error,
                num_workers,
            )
        except Exception as e:
            print(f"Parallel coarse search failed: {e}, using serial")
            use_parallel = False  # Fall back to serial

    # Serial version (used if parallel is disabled or failed)
    if not use_parallel or len(coarse_points) < 20:
        for i, xy1 in enumerate(coarse_points):
            for xy2 in coarse_points[i + 1 :]:
                # Skip boundary cuts
                if _would_cut_along_boundary(xy1, xy2, piece):
                    skipped_boundary_cuts += 1
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

    # Sort by score and take top N candidates
    coarse_cuts.sort(key=lambda x: x[3])
    top_candidates = coarse_cuts[:top_n_to_refine]

    # Early return if no candidates to refine
    if not top_candidates:
        print("Refinement: No coarse cuts found, returning empty list")
        return []

    # Phase 2: Refine top candidates
    refined_cuts = []
    target_area = original_area * target_ratio

    for xy1, xy2, piece, _ in top_candidates:
        # Refine this cut
        refined_p1, refined_p2, refined_score = refine_cut_binary_search(
            cake,
            xy1,
            xy2,
            piece,
            target_area,
            original_ratio,
            acceptable_area_error,
            acceptable_ratio_error,
        )

        # Validate the refined cut
        if cake.cut_is_valid(refined_p1, refined_p2):
            valid, area_diff, ratio_diff = get_areas_and_ratios(
                cake,
                refined_p1,
                refined_p2,
                target_ratio,
                original_area,
                original_ratio,
                acceptable_area_error,
                acceptable_ratio_error,
            )

            if valid:
                refined_cuts.append((refined_p1, refined_p2, piece, refined_score))
            else:
                # If refinement didn't meet strict tolerances, try with relaxed tolerances
                relaxed_area_error = (
                    acceptable_area_error * 1.5
                )  # Conservative fallback for area
                relaxed_ratio_error = (
                    acceptable_ratio_error * 2.0
                )  # Pragmatic fallback for ratio

                valid_relaxed, area_diff_relaxed, ratio_diff_relaxed = (
                    get_areas_and_ratios(
                        cake,
                        refined_p1,
                        refined_p2,
                        target_ratio,
                        original_area,
                        original_ratio,
                        relaxed_area_error,
                        relaxed_ratio_error,
                    )
                )

                if valid_relaxed:
                    # Use the original coarse cut if refinement didn't help
                    refined_cuts.append(
                        (
                            xy1,
                            xy2,
                            piece,
                            calculate_optimization_score(
                                area_diff_relaxed, ratio_diff_relaxed
                            ),
                        )
                    )

    # Sort refined cuts by score
    refined_cuts.sort(key=lambda x: x[3])

    # If no refined cuts found, fall back to using the best coarse cuts
    if not refined_cuts and top_candidates:
        print("Refinement: No refined cuts found, falling back to best coarse cuts")
        # Use the best coarse cuts with relaxed tolerances
        for xy1, xy2, piece, score in top_candidates:
            relaxed_area_error = (
                acceptable_area_error * 3.5
            )  # More conservative for area
            relaxed_ratio_error = (
                acceptable_ratio_error * 5.0
            )  # Lenient for ratio (ratio is working well)

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
                refined_cuts.append((xy1, xy2, piece, score))

    # Optional: Print optimization stats
    if skipped_boundary_cuts > 0:
        print(
            f"Refinement: Skipped {skipped_boundary_cuts} boundary cuts in coarse search"
        )

    print(
        f"Refinement: Found {len(coarse_cuts)} coarse cuts, refined {len(top_candidates)} candidates, got {len(refined_cuts)} final cuts"
    )

    # Return in same format as original find_valid_cuts
    return [(p1, p2, piece) for p1, p2, piece, _ in refined_cuts]
