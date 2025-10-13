from shapely.geometry import LineString, Point, Polygon
import math
import random
from statistics import stdev

from players.player import Player
from src.cake import Cake
from shapely.ops import split

NUMBER_ATTEMPS = 360


class Player10(Player):
    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: str | None,
        num_angle_attempts: int = NUMBER_ATTEMPS,
    ) -> None:
        super().__init__(children, cake, cake_path)
        # Binary search tolerance: area within 0.5 cm² of target
        self.target_area_tolerance = 0.005
        # Number of different angles to try (more attempts = better for complex shapes)
        self.num_angle_attempts = num_angle_attempts

    def find_line(self, position: float, piece: Polygon, angle: float):
        """Make a line at a given angle through a position that cuts the piece.

        Args:
            position: Position along the sweep direction (0 to 1)
            piece: The polygon piece to cut
            angle: Angle in degrees (0-360) where 0 is right, 90 is up
        """

        # Get bounding box of piece
        leftmost, lowest, rightmost, highest = piece.bounds
        width = rightmost - leftmost
        height = highest - lowest
        max_dim = max(width, height) * 2

        # Convert angle to radians
        angle_rad = math.radians(angle)

        # Calculate the perpendicular direction for the sweep
        sweep_angle = angle_rad + math.pi / 2

        # Start from center of bounding box
        center_x = (leftmost + rightmost) / 2
        center_y = (lowest + highest) / 2

        # Calculate sweep offset based on position (0 to 1)
        sweep_offset = (position - 0.5) * max_dim
        offset_x = sweep_offset * math.cos(sweep_angle)
        offset_y = sweep_offset * math.sin(sweep_angle)

        # Calculate point on the sweep line
        point_x = center_x + offset_x
        point_y = center_y + offset_y

        # Create a line at the given angle through this point
        dx = math.cos(angle_rad) * max_dim
        dy = math.sin(angle_rad) * max_dim

        # Create line extending in both directions
        start_point = (point_x - dx, point_y - dy)
        end_point = (point_x + dx, point_y + dy)
        cut_line = LineString([start_point, end_point])

        return cut_line

    def find_cuts(self, line: LineString, piece: Polygon):
        """This function finds the actual points where the cut line goes through cake"""
        intersection = line.intersection(piece.boundary)

        # What is the intersections geometry? - want it to be at least two points
        if intersection.is_empty or intersection.geom_type == "Point":
            return None
        points = []
        if intersection.geom_type == "MultiPoint":
            points = list(intersection.geoms)
        elif intersection.geom_type == "LineString":
            coords = list(intersection.coords)
            points = [Point(c) for c in coords]

        # Need at least 2 points for a valid cut
        if len(points) < 2:
            return None

        # return the points where the sweeping line intersects with the cake
        return (points[0], points[1])

    def calculate_piece_area(self, piece: Polygon, position: float, angle: float):
        """Determines the area of the pieces we cut.

        Args:
            piece: The polygon piece to cut
            position: Position along sweep direction (0 to 1)
            angle: Angle in degrees for the cutting line
        """
        line = self.find_line(position, piece, angle)
        pieces = split(piece, line)

        # we should get two pieces if not, line didn't cut properly
        if len(pieces.geoms) != 2:
            # if we're at the extremes
            if position <= 0.0:
                return 0.0
            elif position >= 1.0:
                return piece.area
            else:
                return 0.0

        piece1, piece2 = pieces.geoms

        # Calculate which piece is "first" based on sweep direction
        angle_rad = math.radians(angle)
        sweep_angle = angle_rad + math.pi / 2
        sweep_dir_x = math.cos(sweep_angle)
        sweep_dir_y = math.sin(sweep_angle)

        # Project centroids onto sweep direction
        centroid1 = piece1.centroid
        centroid2 = piece2.centroid

        proj1 = centroid1.x * sweep_dir_x + centroid1.y * sweep_dir_y
        proj2 = centroid2.x * sweep_dir_x + centroid2.y * sweep_dir_y

        # Return the area of the "first" piece in sweep direction
        if proj1 < proj2:
            return piece1.area
        else:
            return piece2.area

    def binary_search(self, piece: Polygon, target_area: float, angle: float):
        """Use binary search to find position along sweep that cuts off target_area.

        Args:
            piece: The polygon piece to cut
            target_area: The target area for the cut piece
            angle: Angle in degrees for the cutting line
        """

        left_pos = 0.0
        right_pos = 1.0
        best_pos = None
        best_error = float("inf")

        # try for best cut for 50 iterations
        for iteration in range(50):
            # try middle first
            mid_pos = (left_pos + right_pos) / 2

            # get the area of that prospective position
            cut_area = self.calculate_piece_area(piece, mid_pos, angle)

            if cut_area == 0:
                # Too far left, move right
                left_pos = mid_pos
                continue

            if cut_area >= piece.area:
                # Too far right, move left
                right_pos = mid_pos
                continue

            # how far away from the target value
            error = abs(cut_area - target_area)

            # Track best
            if error < best_error:
                best_error = error
                best_pos = mid_pos

            # Check if it's good enough
            if error < self.target_area_tolerance:
                return mid_pos

            # Adjust search based on distance from target area
            if cut_area < target_area:
                left_pos = mid_pos  # Need more, move right
            else:
                right_pos = mid_pos  # Too much, move left

        return best_pos

    def try_cutting_at_angle(
        self, angle: float, verbose: bool = False
    ) -> tuple[list[tuple[Point, Point]], float] | None:
        """Try cutting the cake at a specific angle and return cuts with their score.

        Args:
            angle: Angle in degrees (0-360) for cutting direction
            verbose: Whether to print debug information

        Returns:
            Tuple of (cuts, variance_score) if successful, None if failed
            Lower variance_score is better
        """
        if verbose:
            print(f"\n  Trying angle {angle:.1f} degrees...")

        target_area = self.cake.get_area() / self.children
        cuts = []
        cake_copy = self.cake.copy()

        # Try to make all n-1 cuts
        for cut_idx in range(self.children - 1):
            current_pieces = cake_copy.get_pieces()
            # Always cut the biggest piece
            cutting_piece = max(current_pieces, key=lambda pc: pc.area)

            if verbose:
                print(
                    f"    Cut {cut_idx + 1}: Cutting piece with area {cutting_piece.area:.2f}, target piece {target_area:.2f}"
                )

            # Find the best position using binary search
            position = self.binary_search(cutting_piece, target_area, angle)

            # If we can't find a position, this angle doesn't work
            if position is None:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Failed to find position")
                return None

            # Find the actual cut points
            cut_line = self.find_line(position, cutting_piece, angle)
            cut_points = self.find_cuts(cut_line, cutting_piece)

            if cut_points is None:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Failed to find cut points")
                return None

            from_p, to_p = cut_points

            # Check if the cut is valid
            is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
            if not is_valid:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Invalid - {why}")
                return None

            # Try to make the cut
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))

                # Check the resulting piece sizes
                if verbose:
                    new_pieces = cake_copy.get_pieces()
                    areas = [p.area for p in new_pieces]
                    print(
                        f"      -> Resulted in areas: {[f'{a:.2f}' for a in sorted(areas)]}"
                    )

            except Exception as e:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Exception - {e}")
                return None

        # Check if we got the right number of pieces
        if len(cake_copy.get_pieces()) != self.children:
            if verbose:
                print(
                    f"    Failed: Got {len(cake_copy.get_pieces())} pieces, expected {self.children}"
                )
            return None

        # Check piece size consistency
        areas = [p.area for p in cake_copy.get_pieces()]
        size_span = max(areas) - min(areas)

        # Per project spec: pieces within 0.5 cm² are considered same size
        # But for the sweeping algorithm, we need some tolerance
        # Use a reasonable threshold based on number of children
        max_acceptable_span = max(
            5.0, target_area * 0.15
        )  # At least 5 cm² or 15% of target

        if size_span > max_acceptable_span:
            if verbose:
                print(
                    f"    Failed: Size span {size_span:.2f} is too large (>{max_acceptable_span:.2f})"
                )
            return None

        # Calculate crust ratio variance (our score to minimize)
        ratios = cake_copy.get_piece_ratios()

        # Check if all pieces are valid (have reasonable ratios)
        if any(r < 0 or r > 1 for r in ratios):
            if verbose:
                print(f"    Failed: Invalid ratios {ratios}")
            return None

        # Calculate variance in ratios (lower is better)
        if len(ratios) > 1:
            variance = stdev(ratios)
        else:
            variance = 0.0

        if verbose:
            print(f"    Success! Variance: {variance:.4f}, Size span: {size_span:.2f}")
            print(f"    Areas: {[f'{a:.2f}' for a in sorted(areas)]}")
            print(f"    Ratios: {[f'{r:.3f}' for r in ratios]}")

        return (cuts, variance)

    def find_best_cut_at_angle(
        self,
        cutting_piece: Polygon,
        target_area: float,
        angle: float,
        cake_copy: Cake,
        is_last_cut: bool = False,
    ) -> tuple[Point, Point, float, float] | None:
        """Try cutting at a specific angle and return the cut points and resulting piece info.

        Args:
            cutting_piece: The piece to cut
            target_area: Target area for the cut piece
            angle: Angle to try
            cake_copy: Current state of the cake
            is_last_cut: Whether this is the last cut (more lenient validation)

        Returns:
            Tuple of (from_point, to_point, crust_ratio_of_new_piece, area_of_new_piece) or None if invalid
        """
        # Find the best position using binary search
        position = self.binary_search(cutting_piece, target_area, angle)

        if position is None:
            return None

        # Find the actual cut points
        cut_line = self.find_line(position, cutting_piece, angle)
        cut_points = self.find_cuts(cut_line, cutting_piece)

        if cut_points is None:
            return None

        from_p, to_p = cut_points

        # Check if the cut is valid
        is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
        if not is_valid:
            return None

        # Simulate the cut to get the new piece and its crust ratio
        test_cake = cake_copy.copy()
        try:
            # Find which piece gets created
            # pieces_before = [p.area for p in test_cake.get_pieces()]
            test_cake.cut(from_p, to_p)
            pieces_after = test_cake.get_pieces()

            # Find the new piece (smallest one, as we're cutting off target_area)
            new_piece = min(pieces_after, key=lambda p: p.area)
            new_piece_ratio = test_cake.get_piece_ratio(new_piece)
            new_piece_area = new_piece.area

            # Validate the new piece size is reasonable
            # More lenient for last cut, or when cutting small pieces
            if is_last_cut:
                tolerance = 0.5  # Very lenient for last cut
            elif cutting_piece.area < target_area * 2:
                tolerance = 0.4  # Lenient for small pieces
            else:
                tolerance = 0.3  # Standard tolerance

            if abs(new_piece_area - target_area) > target_area * tolerance:
                return None

            return (from_p, to_p, new_piece_ratio, new_piece_area)

        except Exception:
            return None

    def try_divide_and_conquer_cut(
        self, piece: Polygon, num_children: int, target_ratio: float, depth: int = 0
    ) -> tuple[list[tuple[Point, Point]], float] | None:
        """Try to cut a piece for num_children using divide-and-conquer with different ratios.

        Args:
            piece: The polygon piece to divide
            num_children: Number of children to serve from this piece
            target_ratio: Target crust ratio
            depth: Recursion depth for logging

        Returns:
            Tuple of (cuts, score) or None if failed
        """
        if num_children == 1:
            # Base case: no more cuts needed
            return ([], 0.0)

        if num_children == 2:
            # Base case: just split in half (1:1 ratio)
            target_area = piece.area / 2

            best_cut = None
            best_score = float("inf")

            # Try fewer angles for base case
            num_attempts = 10  # Much fewer attempts
            for _ in range(num_attempts):
                angle = random.uniform(0, 180)
                position = self.binary_search(piece, target_area, angle)

                if position is None:
                    continue

                cut_line = self.find_line(position, piece, angle)
                cut_points = self.find_cuts(cut_line, piece)

                if cut_points is None:
                    continue

                from_p, to_p = cut_points

                # Simulate cut and evaluate
                test_pieces = split(piece, cut_line)
                if len(test_pieces.geoms) != 2:
                    continue

                p1, p2 = test_pieces.geoms
                ratio1 = self.cake.get_piece_ratio(p1)
                ratio2 = self.cake.get_piece_ratio(p2)

                # Score based on ratio uniformity and size balance
                ratio_error = abs(ratio1 - ratio2)
                size_error = abs(p1.area - p2.area) / piece.area
                score = ratio_error * 2.0 + size_error * 1.0

                if score < best_score:
                    best_score = score
                    best_cut = (from_p, to_p)

                    # Early stopping if good enough
                    if score < 0.05:
                        break

            if best_cut:
                return ([best_cut], best_score)
            return None

        # Try random (ratio, angle) pairs together
        best_result = None
        best_total_score = float("inf")

        max_ratio_numerator = num_children // 2
        if max_ratio_numerator < 1:
            return None

        # Number of random (ratio, angle) attempts
        # Drastically reduce to prevent exponential blowup
        if depth == 0:
            num_attempts = 10  # Very limited at top level
        elif depth == 1:
            num_attempts = 5  # Even fewer at depth 1
        else:
            num_attempts = 3  # Minimal at deeper levels

        for attempt in range(num_attempts):
            # Randomly select BOTH ratio AND angle together
            split_count = random.randint(1, max_ratio_numerator)
            angle = random.uniform(0, 180)

            # Try to cut off split_count children's worth at this angle
            target_area = piece.area * split_count / num_children
            remaining_count = num_children - split_count

            position = self.binary_search(piece, target_area, angle)
            if position is None:
                continue

            cut_line = self.find_line(position, piece, angle)
            cut_points = self.find_cuts(cut_line, piece)
            if cut_points is None:
                continue

            from_p, to_p = cut_points

            # Simulate the cut
            test_pieces = split(piece, cut_line)
            if len(test_pieces.geoms) != 2:
                continue

            p1, p2 = test_pieces.geoms

            # Identify which piece is for split_count children
            if p1.area < p2.area:
                small_piece, large_piece = p1, p2
                small_count, large_count = split_count, remaining_count
            else:
                small_piece, large_piece = p2, p1
                small_count, large_count = split_count, remaining_count

            # Quick score for this cut
            ratio1 = self.cake.get_piece_ratio(p1)
            ratio2 = self.cake.get_piece_ratio(p2)
            cut_score = abs(ratio1 - target_ratio) + abs(ratio2 - target_ratio)

            # Recursively divide both pieces
            result1 = self.try_divide_and_conquer_cut(
                small_piece, small_count, target_ratio, depth + 1
            )
            result2 = self.try_divide_and_conquer_cut(
                large_piece, large_count, target_ratio, depth + 1
            )

            if result1 is None or result2 is None:
                continue

            cuts1, score1 = result1
            cuts2, score2 = result2

            # Combine results
            all_cuts = [(from_p, to_p)] + cuts1 + cuts2
            total_score = cut_score + score1 + score2

            if total_score < best_total_score:
                best_total_score = total_score
                best_result = (all_cuts, total_score)

                # Early stopping: if we found a very good solution, stop searching
                if total_score < 0.1 * num_children:  # Good enough threshold
                    break

        return best_result

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - greedy approach with random (ratio, angle) pairs"""
        print(f"__________Cutting for {self.children} children_______")

        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        print(f"TARGET AREA: {target_area:.2f} cm²")
        print(f"TARGET CRUST RATIO: {target_ratio:.3f}")
        # print(f"Strategy: Greedy cutting with random ratio+angle exploration\n")

        return self._greedy_ratio_angle_cutting(target_area, target_ratio)

    def _greedy_ratio_angle_cutting(
        self, target_area: float, target_ratio: float
    ) -> list[tuple[Point, Point]]:
        """
        TRUE divide-and-conquer approach:
        - Track how many children each piece is for (e.g., 1/n, 2/n, 3/n...)
        - Iteratively divide pieces until all are for 1 child
        - For each piece with n>1 children, try random (split_ratio, angle) pairs
        """
        cake_copy = self.cake.copy()
        all_cuts = []

        # Initialize: the whole cake is for all children
        # pieces_queue: list of (piece_polygon, num_children_for_this_piece)
        pieces_queue = [(cake_copy.exterior_shape, self.children)]

        cut_number = 0
        while cut_number < self.children - 1:
            # Find a piece that needs to be divided (num_children > 1)
            cutting_piece = None
            cutting_num_children = 0
            cutting_index = -1

            for i, (piece, num_children) in enumerate(pieces_queue):
                if num_children > 1:
                    # Prefer larger pieces or pieces with more children
                    if num_children > cutting_num_children:
                        cutting_piece = piece
                        cutting_num_children = num_children
                        cutting_index = i
                    elif num_children == cutting_num_children and piece.area > (
                        cutting_piece.area if cutting_piece else 0
                    ):
                        cutting_piece = piece
                        cutting_num_children = num_children
                        cutting_index = i

            if cutting_piece is None:
                # All pieces are for 1 child
                break

            # Remove the piece from queue
            pieces_queue.pop(cutting_index)

            print(f"\n=== Cut {cut_number + 1}/{self.children - 1} ===")
            print(
                f"Dividing piece for {cutting_num_children} children (area: {cutting_piece.area:.2f})"
            )

            # Try different split ratios: split n children into (k, n-k)
            # where k ranges from 1 to floor(n/2) for balanced divide-and-conquer
            min_split = 1
            max_split = max(1, cutting_num_children // 2)
            print(
                f"Exploring split ratios: 1/{cutting_num_children} to {max_split}/{cutting_num_children}"
            )

            # Two-phase strategy:
            # Phase 1: Try all split ratios with cardinal angles + random sampling to find best ratio
            # Phase 2: Focus on best split ratio, only vary angles
            num_attempts = self.num_angle_attempts
            cardinal_angles = [0, 90, 180, 270]

            best_cut = None
            best_score = float("inf")
            best_split_ratio = None
            valid_attempts = 0

            # Track best score for each split ratio
            split_ratio_scores = {}
            for split_children in range(min_split, max_split + 1):
                split_ratio_scores[split_children] = float("inf")

            # Build list of (split_ratio, angle) to try
            attempts_to_try = []

            # First: Try all split ratios with all cardinal angles
            for split_children in range(min_split, max_split + 1):
                for angle in cardinal_angles:
                    attempts_to_try.append((split_children, angle, "phase1"))

            # Phase 1: Random sample all split ratios (first half)
            phase1_attempts = num_attempts // 2
            for _ in range(phase1_attempts):
                split_children = random.randint(min_split, max_split)
                angle = random.uniform(0, 180)
                attempts_to_try.append((split_children, angle, "phase1"))

            # Try phase 1 attempts
            for split_children, angle, phase in attempts_to_try:
                remaining_children = cutting_num_children - split_children

                # Calculate target area for this split
                target_cut_area = target_area * split_children

                # Find the cut position using binary search
                position = self.binary_search(cutting_piece, target_cut_area, angle)
                if position is None:
                    continue

                cut_line = self.find_line(position, cutting_piece, angle)
                cut_points = self.find_cuts(cut_line, cutting_piece)
                if cut_points is None:
                    continue

                from_p, to_p = cut_points

                # Simulate the cut to get the two pieces
                test_pieces = split(cutting_piece, cut_line)
                if len(test_pieces.geoms) != 2:
                    continue

                p1, p2 = test_pieces.geoms

                # Determine which piece is for split_children
                if abs(p1.area - target_cut_area) < abs(p2.area - target_cut_area):
                    small_piece, large_piece = p1, p2
                else:
                    small_piece, large_piece = p2, p1

                # Get crust ratios
                ratio1 = self.cake.get_piece_ratio(small_piece)
                ratio2 = self.cake.get_piece_ratio(large_piece)

                valid_attempts += 1

                # Score this cut
                size_error = abs(small_piece.area - target_cut_area)
                ratio_error = abs(ratio1 - target_ratio) + abs(ratio2 - target_ratio)
                score = size_error * 3.0 + ratio_error * 1.0

                # Track best score for this split ratio
                if score < split_ratio_scores[split_children]:
                    split_ratio_scores[split_children] = score

                if score < best_score:
                    best_score = score
                    best_cut = (
                        from_p,
                        to_p,
                        small_piece,
                        large_piece,
                        ratio1,
                        ratio2,
                        angle,
                    )
                    best_split_ratio = (split_children, remaining_children)

            # Phase 2: Use the best split ratio found, only vary angles
            if split_ratio_scores:
                # Find the split ratio with the best score
                best_ratio_from_phase1 = min(
                    split_ratio_scores.keys(), key=lambda k: split_ratio_scores[k]
                )
                phase2_attempts = num_attempts - phase1_attempts

                print(
                    f"  Phase 1 complete. Best split ratio: {best_ratio_from_phase1}/{cutting_num_children}"
                )
                print(
                    f"  Phase 2: Trying {phase2_attempts} more angles with best ratio..."
                )

                remaining_children_phase2 = (
                    cutting_num_children - best_ratio_from_phase1
                )
                target_cut_area_phase2 = target_area * best_ratio_from_phase1

                # In phase 2, try cardinal angles again with the best ratio, then random
                phase2_angles = list(cardinal_angles) + [
                    random.uniform(0, 180)
                    for _ in range(phase2_attempts - len(cardinal_angles))
                ]

                for angle in phase2_angles:
                    # Only vary angle, keep the best split ratio
                    split_children = best_ratio_from_phase1
                    remaining_children = remaining_children_phase2

                    # Find the cut position using binary search
                    position = self.binary_search(
                        cutting_piece, target_cut_area_phase2, angle
                    )
                    if position is None:
                        continue

                    cut_line = self.find_line(position, cutting_piece, angle)
                    cut_points = self.find_cuts(cut_line, cutting_piece)
                    if cut_points is None:
                        continue

                    from_p, to_p = cut_points

                    # Simulate the cut to get the two pieces
                    test_pieces = split(cutting_piece, cut_line)
                    if len(test_pieces.geoms) != 2:
                        continue

                    p1, p2 = test_pieces.geoms

                    # Determine which piece is for split_children
                    if abs(p1.area - target_cut_area_phase2) < abs(
                        p2.area - target_cut_area_phase2
                    ):
                        small_piece, large_piece = p1, p2
                    else:
                        small_piece, large_piece = p2, p1

                    # Get crust ratios
                    ratio1 = self.cake.get_piece_ratio(small_piece)
                    ratio2 = self.cake.get_piece_ratio(large_piece)

                    valid_attempts += 1

                    # Score this cut
                    size_error = abs(small_piece.area - target_cut_area_phase2)
                    ratio_error = abs(ratio1 - target_ratio) + abs(
                        ratio2 - target_ratio
                    )
                    score = size_error * 3.0 + ratio_error * 1.0

                    if score < best_score:
                        best_score = score
                        best_cut = (
                            from_p,
                            to_p,
                            small_piece,
                            large_piece,
                            ratio1,
                            ratio2,
                            angle,
                        )
                        best_split_ratio = (split_children, remaining_children)

            if best_cut is None:
                print(f"  No valid cut found after {num_attempts} attempts!")
                # Put the piece back for now (shouldn't happen often)
                pieces_queue.append((cutting_piece, cutting_num_children))
                continue

            from_p, to_p, small_piece, large_piece, ratio1, ratio2, used_angle = (
                best_cut
            )
            split_children, remaining_children = best_split_ratio

            # Make the cut on the actual cake
            cake_copy.cut(from_p, to_p)
            all_cuts.append((from_p, to_p))
            cut_number += 1

            # Add the two new pieces to the queue with their child counts
            pieces_queue.append((small_piece, split_children))
            pieces_queue.append((large_piece, remaining_children))

            # Print info
            print(f"  Best cut (tried {valid_attempts} valid attempts)")
            print(
                f"  Split ratio: {split_children}/{cutting_num_children} and {remaining_children}/{cutting_num_children}, angle={used_angle:.1f}°"
            )
            print(
                f"  Piece 1 ({split_children} children): size={small_piece.area:.2f} (target={split_children * target_area:.2f}), crust_ratio={ratio1:.3f}"
            )
            print(
                f"  Piece 2 ({remaining_children} children): size={large_piece.area:.2f} (target={remaining_children * target_area:.2f}), crust_ratio={ratio2:.3f}"
            )

            # Show current queue status
            total_in_queue = sum(nc for _, nc in pieces_queue)
            print(
                f"  Queue: {len(pieces_queue)} pieces for {total_in_queue} total children"
            )

        # Final summary
        print(f"\n{'=' * 50}")
        print(f"FINAL RESULT: {len(all_cuts)}/{self.children - 1} cuts completed")

        pieces = cake_copy.get_pieces()
        areas = [p.area for p in pieces]
        ratios = cake_copy.get_piece_ratios()

        print(f"\nPiece areas: {[f'{a:.2f}' for a in sorted(areas)]}")
        print(
            f"  Min: {min(areas):.2f}, Max: {max(areas):.2f}, Span: {max(areas) - min(areas):.2f}"
        )

        print(f"\nCrust ratios: {[f'{r:.3f}' for r in ratios]}")
        if len(ratios) > 1:
            ratio_variance = stdev(ratios)
            print(f"  Variance: {ratio_variance:.4f}")
            print(
                f"  Min: {min(ratios):.3f}, Max: {max(ratios):.3f}, Span: {max(ratios) - min(ratios):.3f}"
            )
        print(f"{'=' * 50}\n")

        return all_cuts

    def _sequential_cutting(self) -> list[tuple[Point, Point]]:
        """Fallback: Sequential per-piece cutting with angle selection"""
        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area

        cuts = []
        cake_copy = self.cake.copy()

        for cut_idx in range(self.children - 1):
            print(f"=== Cut {cut_idx + 1}/{self.children - 1} ===")

            current_pieces = cake_copy.get_pieces()
            cutting_piece = max(current_pieces, key=lambda pc: pc.area)

            print(f"Cutting piece area: {cutting_piece.area:.2f}")

            is_last_cut = cut_idx == self.children - 2

            best_cut = None
            best_score = float("inf")
            best_angle = None

            num_attempts = (
                self.num_angle_attempts * 2 if is_last_cut else self.num_angle_attempts
            )
            valid_attempts = 0

            for attempt in range(num_attempts):
                angle = random.uniform(0, 180)

                result = self.find_best_cut_at_angle(
                    cutting_piece, target_area, angle, cake_copy, is_last_cut
                )

                if result is not None:
                    from_p, to_p, piece_ratio, piece_area = result

                    size_error = abs(piece_area - target_area) / target_area
                    ratio_error = abs(piece_ratio - target_ratio)
                    score = size_error * 3.0 + ratio_error * 1.0

                    valid_attempts += 1

                    if score < best_score:
                        best_score = score
                        best_cut = (from_p, to_p)
                        best_angle = angle
                        best_ratio = piece_ratio
                        best_size = piece_area

            if best_cut is None:
                print("  Failed: No valid cut found")
                continue

            from_p, to_p = best_cut
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))

                areas = [p.area for p in cake_copy.get_pieces()]
                size_error = abs(best_size - target_area)
                ratio_error = abs(best_ratio - target_ratio)
                print(
                    f"  Best angle: {best_angle:.1f}° (tried {valid_attempts} valid angles)"
                )
                print(
                    f"  Piece: size={best_size:.2f} (target={target_area:.2f}, err={size_error:.2f}), ratio={best_ratio:.3f} (target={target_ratio:.3f}, err={ratio_error:.3f})"
                )
                print(f"  Current areas: {[f'{a:.2f}' for a in sorted(areas)]}\n")

            except Exception as e:
                print(f"  Error applying cut: {e}")
                break

        return cuts
