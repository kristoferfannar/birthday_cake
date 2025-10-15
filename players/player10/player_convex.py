from shapely.geometry import LineString, Point, Polygon
import math
import random
from statistics import stdev

from players.player import Player
from src.cake import Cake
from shapely.ops import split


class Player10(Player):
    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: str | None,
        num_angle_attempts: int = None,  # Auto-scale based on children
    ) -> None:
        super().__init__(children, cake, cake_path)
        self.target_area_tolerance = 0.005

        # FIX 1: Adaptive attempts based on number of children and cake complexity
        if num_angle_attempts is None:
            # Fewer children = more attempts per cut affordable
            # More children = need to reduce attempts
            if children <= 3:
                self.num_angle_attempts = 360
            elif children <= 5:
                self.num_angle_attempts = 360
            elif children <= 8:
                self.num_angle_attempts = 360
            else:
                self.num_angle_attempts = 360  # Minimum for many children
        else:
            self.num_angle_attempts = num_angle_attempts

        print(f"Initialized with {self.num_angle_attempts} angle attempts per cut")

    def find_line(self, position: float, piece: Polygon, angle: float):
        """Make a line at a given angle through a position that cuts the piece."""
        leftmost, lowest, rightmost, highest = piece.bounds
        width = rightmost - leftmost
        height = highest - lowest
        max_dim = max(width, height) * 3  # FIX: Longer lines for concave shapes

        angle_rad = math.radians(angle)
        sweep_angle = angle_rad + math.pi / 2

        center_x = (leftmost + rightmost) / 2
        center_y = (lowest + highest) / 2

        sweep_offset = (position - 0.5) * max_dim
        offset_x = sweep_offset * math.cos(sweep_angle)
        offset_y = sweep_offset * math.sin(sweep_angle)

        point_x = center_x + offset_x
        point_y = center_y + offset_y

        dx = math.cos(angle_rad) * max_dim
        dy = math.sin(angle_rad) * max_dim

        start_point = (point_x - dx, point_y - dy)
        end_point = (point_x + dx, point_y + dy)
        cut_line = LineString([start_point, end_point])

        return cut_line

    def find_cuts(self, line: LineString, piece: Polygon):
        """Find exactly two points where the cut line intersects the cake boundary."""
        print("Finding cuts...")
        intersection = line.intersection(piece.boundary)

        points = []
        if intersection.is_empty:
            return None
        if intersection.geom_type == "Point":
            points = [intersection]
        elif intersection.geom_type == "MultiPoint":
            points = list(intersection.geoms)
        elif intersection.geom_type == "LineString":
            coords = list(intersection.coords)
            points = [Point(coords[0]), Point(coords[-1])]
        elif intersection.geom_type == "GeometryCollection":
            for geom in intersection.geoms:
                if geom.geom_type == "Point":
                    points.append(geom)
                elif geom.geom_type == "LineString":
                    coords = list(geom.coords)
                    points.extend([Point(coords[0]), Point(coords[-1])])

        # Remove duplicates
        unique_points = []
        for p in points:
            if not any(p.equals(q) for q in unique_points):
                unique_points.append(p)
        points = unique_points

        if len(points) < 2:
            return None

        # If more than 2 points, select the two farthest apart
        if len(points) > 2:

            def proj(pt):
                return line.project(pt)

            points_sorted = sorted(points, key=proj)
            cut_points = (points_sorted[0], points_sorted[-1])
        else:
            cut_points = (points[0], points[1])

        return cut_points

    def calculate_piece_area(self, piece: Polygon, position: float, angle: float):
        """Determines the area of the pieces we cut."""
        line = self.find_line(position, piece, angle)

        try:
            pieces = split(piece, line)
        except Exception:
            return 0.0

        if len(pieces.geoms) != 2:
            if position <= 0.0:
                return 0.0
            elif position >= 1.0:
                return piece.area
            else:
                return 0.0

        piece1, piece2 = pieces.geoms

        angle_rad = math.radians(angle)
        sweep_angle = angle_rad + math.pi / 2
        sweep_dir_x = math.cos(sweep_angle)
        sweep_dir_y = math.sin(sweep_angle)

        centroid1 = piece1.centroid
        centroid2 = piece2.centroid

        proj1 = centroid1.x * sweep_dir_x + centroid1.y * sweep_dir_y
        proj2 = centroid2.x * sweep_dir_x + centroid2.y * sweep_dir_y

        if proj1 < proj2:
            return piece1.area
        else:
            return piece2.area

    def binary_search(self, piece: Polygon, target_area: float, angle: float):
        """Use binary search to find position along sweep that cuts off target_area."""
        left_pos = 0.0
        right_pos = 1.0
        best_pos = None
        best_error = float("inf")

        # FIX 2: Reduced iterations for speed (50 -> 30)
        for iteration in range(50):
            mid_pos = (left_pos + right_pos) / 2

            cut_area = self.calculate_piece_area(piece, mid_pos, angle)

            if cut_area == 0:
                left_pos = mid_pos
                continue

            if cut_area >= piece.area:
                right_pos = mid_pos
                continue

            error = abs(cut_area - target_area)

            if error < best_error:
                best_error = error
                best_pos = mid_pos

            if error < self.target_area_tolerance:
                return mid_pos

            if cut_area < target_area:
                left_pos = mid_pos
            else:
                right_pos = mid_pos

        return best_pos

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - optimized greedy approach"""
        print(f"__________Cutting for {self.children} children_______")

        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        print(f"TARGET AREA: {target_area:.2f} cm²")
        print(f"TARGET CRUST RATIO: {target_ratio:.3f}")

        return self._optimized_greedy_cutting(target_area, target_ratio)

    def _optimized_greedy_cutting(
        self, target_area: float, target_ratio: float
    ) -> list[tuple[Point, Point]]:
        """Optimized divide-and-conquer with smart heuristics for concave shapes."""
        print("Starting optimized greedy cutting...")
        cake_copy = self.cake.copy()
        all_cuts = []

        pieces_queue = [(cake_copy.exterior_shape, self.children)]
        cut_number = 0

        while cut_number < self.children - 1:
            # Find piece to cut
            cutting_piece = None
            cutting_num_children = 0
            cutting_index = -1

            for i, (piece, num_children) in enumerate(pieces_queue):
                if num_children > 1:
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
                break

            pieces_queue.pop(cutting_index)

            print(f"\n=== Cut {cut_number + 1}/{self.children - 1} ===")
            print(
                f"Dividing piece for {cutting_num_children} children (area: {cutting_piece.area:.2f})"
            )

            # FIX 3: Smart split ratio selection
            # For large n, prefer balanced splits (close to 50/50)
            # For small n, can try more ratios
            if cutting_num_children > 6:
                # Large piece: only try balanced splits
                split_ratios = [cutting_num_children // 2]
                print(f"Using balanced split: {split_ratios[0]}/{cutting_num_children}")
            elif cutting_num_children > 3:
                # Medium piece: try 2-3 ratios around the middle
                mid = cutting_num_children // 2
                split_ratios = [mid - 1, mid, mid + 1]
                split_ratios = [
                    s for s in split_ratios if 1 <= s < cutting_num_children
                ]
                print(f"Trying split ratios: {split_ratios}")
            else:
                # Small piece: try all ratios
                split_ratios = list(range(1, cutting_num_children // 2 + 1))
                print(f"Trying all split ratios: {split_ratios}")

            # FIX 4: Smarter angle selection
            # Start with cardinal angles, then try angles based on piece orientation
            cardinal_angles = [0, 45, 90, 135]

            # Estimate piece's principal axis
            leftmost, lowest, rightmost, highest = cutting_piece.bounds
            width = rightmost - leftmost
            height = highest - lowest

            # Add angles aligned with piece shape
            if width > height * 1.5:
                oriented_angles = [0, 180]  # Horizontal cuts for wide pieces
            elif height > width * 1.5:
                oriented_angles = [90, 270]  # Vertical cuts for tall pieces
            else:
                oriented_angles = [45, 135]  # Diagonal for square-ish pieces

            # Combine and deduplicate
            priority_angles = list(set(cardinal_angles + oriented_angles))

            # Calculate how many random angles to try
            num_priority = len(priority_angles) * len(split_ratios)
            num_random = max(5, self.num_angle_attempts - num_priority)

            best_cut = None
            best_score = float("inf")
            best_split_ratio = None
            valid_attempts = 0

            # Try all (split_ratio, angle) combinations
            attempts = []

            # Priority: Split ratios × priority angles
            for split_children in split_ratios:
                for angle in priority_angles:
                    attempts.append((split_children, angle))

            # Add random attempts
            for _ in range(num_random):
                split_children = random.choice(split_ratios)
                angle = random.uniform(0, 180)
                attempts.append((split_children, angle))

            print(f"  Trying {len(attempts)} total attempts...")

            for split_children, angle in attempts:
                remaining_children = cutting_num_children - split_children
                target_cut_area = target_area * split_children

                position = self.binary_search(cutting_piece, target_cut_area, angle)
                if position is None:
                    print(position)
                    continue

                cut_line = self.find_line(position, cutting_piece, angle)
                cut_points = self.find_cuts(cut_line, cutting_piece)
                if cut_points is None:
                    continue

                from_p, to_p = cut_points

                # Quick validation before expensive split
                try:
                    test_pieces = split(cutting_piece, cut_line)
                    if len(test_pieces.geoms) != 2:
                        continue

                    p1, p2 = test_pieces.geoms

                    # Determine which piece is which
                    if abs(p1.area - target_cut_area) < abs(p2.area - target_cut_area):
                        small_piece, large_piece = p1, p2
                    else:
                        small_piece, large_piece = p2, p1

                    # FIX 5: Simplified scoring (faster)
                    size_error = abs(small_piece.area - target_cut_area) / target_area

                    # Only calculate ratios for promising cuts
                    if size_error < 0.5:  # Within 50% of target
                        ratio1 = self.cake.get_piece_ratio(small_piece)
                        ratio2 = self.cake.get_piece_ratio(large_piece)
                        ratio_error = abs(ratio1 - target_ratio) + abs(
                            ratio2 - target_ratio
                        )
                        score = size_error * 3.0 + ratio_error * 1.0
                    else:
                        score = size_error * 3.0  # Skip ratio calculation for bad cuts

                    valid_attempts += 1

                    if score < best_score:
                        best_score = score
                        best_cut = (from_p, to_p, small_piece, large_piece, angle)
                        best_split_ratio = (split_children, remaining_children)

                        # FIX 6: Early stopping for good cuts
                        if score < 0.001:  # Good enough
                            print(
                                f"  Found good cut early (score={score:.3f}), stopping search"
                            )
                            break

                except Exception:
                    continue

            if best_cut is None:
                print(f"  No valid cut found after trying {len(attempts)} attempts!")
                pieces_queue.append((cutting_piece, cutting_num_children))
                continue

            from_p, to_p, small_piece, large_piece, used_angle = best_cut
            split_children, remaining_children = best_split_ratio

            # Make the cut
            cake_copy.cut(from_p, to_p)
            all_cuts.append((from_p, to_p))
            cut_number += 1

            # Add pieces to queue
            pieces_queue.append((small_piece, split_children))
            pieces_queue.append((large_piece, remaining_children))

            print(
                f"  Best cut: {split_children}/{cutting_num_children} at {used_angle:.1f}° (tried {valid_attempts} valid)"
            )
            print(
                f"  Piece 1: {small_piece.area:.2f} cm² for {split_children} children"
            )
            print(
                f"  Piece 2: {large_piece.area:.2f} cm² for {remaining_children} children"
            )

        # Final summary
        print(f"\n{'=' * 50}")
        print(f"FINAL: {len(all_cuts)}/{self.children - 1} cuts completed")

        pieces = cake_copy.get_pieces()
        areas = [p.area for p in pieces]
        ratios = cake_copy.get_piece_ratios()

        print(f"Piece areas: {[f'{a:.2f}' for a in sorted(areas)]}")
        print(f"  Span: {max(areas) - min(areas):.2f}")

        if len(ratios) > 1:
            ratio_variance = stdev(ratios)
            print(f"Crust ratios: variance={ratio_variance:.4f}")
        print(f"{'=' * 50}\n")

        return all_cuts
