from shapely import Point, LineString, MultiPoint
from shapely.geometry import Polygon
from shapely.ops import split
from typing import cast
from math import radians, sin, cos, atan2, degrees, sqrt, copysign
import src.constants as c
from players.player import Player
from players.random_player import RandomPlayer
from src.cake import Cake, extend_line, copy_geom


class CrustOptimizingPlayer(Player):
    """
    Strategy:
    - Samples random points along the cake crust (outer edge)
    - Computes crust density to find crust-heavy regions
    - Picks starting point p1 from densest crust points
    - Chooses p2 and evaluates cut quality using weighted scoring
    - Includes an angle-based fallback for precision
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.random_player = RandomPlayer(children, cake, cake_path)
        self.valid_line_pair = dict()
        self.tolerance = 0.05
        # FIX: Added the missing attribute initialization
        self.target_area_tolerance = 0.0005
        self.max_crust_points = 3
        self.angle_increment = 15  # Degrees

    # ---------- Utility methods ----------

    def _get_pieces_after_temp_cut(
        self, piece: Polygon, from_p: Point, to_p: Point
    ) -> list[Polygon]:
        """
        Simulates a cut on a given piece and returns the resulting pieces.
        This function replaces the non-existent method in the Cake class.
        """
        piece_copy = copy_geom(piece)

        # FIX: Ensure from_p and to_p are Shapely Point objects
        if not isinstance(from_p, Point):
            from_p = Point(from_p)
        if not isinstance(to_p, Point):
            to_p = Point(to_p)

        bound = piece_copy.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = LineString([a, b])
        line = extend_line(line)

        try:
            split_piece = split(piece_copy, line)
            split_pieces: list[Polygon] = [
                cast(Polygon, geom) for geom in split_piece.geoms
            ]
            return split_pieces
        except Exception:
            return []

    def get_crust_length(self, piece: Polygon) -> float:
        """Length of piece boundary that lies on the exterior crust."""
        return piece.boundary.intersection(self.cake.exterior_shape.boundary).length

    def get_crust_density(self, piece: Polygon, p: Point):
        """Approximate local crust density near a boundary point."""
        crust_section = self.cake.exterior_shape.boundary.buffer(c.CRUST_SIZE * 2)
        nearby = piece.boundary.intersection(crust_section)
        return nearby.length

    def get_piece(self, p1: Point, p2: Point) -> Polygon | None:
        """Find which piece both points belong to."""
        for piece in self.cake.get_pieces():
            if self.cake.point_lies_on_piece_boundary(
                p1, piece
            ) and self.cake.point_lies_on_piece_boundary(p2, piece):
                return piece
        return None

    def find_cuts(self, line: LineString, piece: Polygon) -> tuple[Point, Point] | None:
        """
        Finds the first two valid intersection points between a line and a piece's boundary.
        Returns the two points or None if a valid two-point cut cannot be formed.
        """
        intersection = line.intersection(piece.boundary)

        if intersection.is_empty:
            return None

        if isinstance(intersection, Point):
            return None

        if isinstance(intersection, LineString):
            return None

        if isinstance(intersection, MultiPoint):
            points = list(intersection.geoms)
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    p1 = points[i]
                    p2 = points[j]

                    is_valid, _ = self.cake.cut_is_valid(p1, p2)

                    if is_valid:
                        return (p1, p2)

        return None

    def check_precision(
        self,
        p1: Point,
        p2: Point,
        Area_list: list[float],
        piece: Polygon,
        crust_ratio: float,
    ) -> tuple[float, float]:
        """Returns precision metrics using the new utility method."""
        split_pieces = self._get_pieces_after_temp_cut(piece, p1, p2)

        if len(split_pieces) < 2:
            return float("inf"), float("inf")

        if split_pieces[0].area < split_pieces[1].area:
            Area_piece = split_pieces[0].area
            new_piece_crust_ratio = self.cake.get_piece_ratio(split_pieces[0])
        else:
            Area_piece = split_pieces[1].area
            new_piece_crust_ratio = self.cake.get_piece_ratio(split_pieces[1])

        closest_target_area = min(Area_list, key=lambda a: abs(a - Area_piece))
        cake_precision = (
            abs(Area_piece - closest_target_area) / self.cake.exterior_shape.area
        )
        crust_precision = abs(new_piece_crust_ratio - crust_ratio)

        return cake_precision, crust_precision

    def get_weight(
        self, p1: Point, p2: Point, piece: Polygon, Goal_ratio: float
    ) -> float:
        """Compute weight for a given cut — based on area ratio deviation."""
        split_pieces = self._get_pieces_after_temp_cut(piece, p1, p2)
        if len(split_pieces) != 2:
            return 0.0
        r1 = self.cake.get_piece_ratio(split_pieces[0])
        r2 = self.cake.get_piece_ratio(split_pieces[1])
        weight = 0.5 * (1 - abs(Goal_ratio - r1)) + 0.5 * (1 - abs(Goal_ratio - r2))
        return weight

    def _get_line_at_angle(
        self, center_x: float, center_y: float, angle_degrees: float, length: float
    ) -> LineString:
        """Creates a line of a given length, angle, and center point."""
        angle_rad = radians(angle_degrees)
        half_len = length / 2
        p1_x = center_x - half_len * cos(angle_rad)
        p1_y = center_y - half_len * sin(angle_rad)
        p2_x = center_x + half_len * cos(angle_rad)
        p2_y = center_y + half_len * sin(angle_rad)
        return LineString([(p1_x, p1_y), (p2_x, p2_y)])

    def _calculate_area_at_offset(
        self, piece: Polygon, offset: float, angle_degrees: float
    ) -> float:
        """Calculates the area of one side of a cut at a specific offset and angle."""
        bounds = piece.bounds
        search_range = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2

        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        angle_rad = radians(angle_degrees + 90)
        cut_center_x = center_x + offset * cos(angle_rad)
        cut_center_y = center_y + offset * sin(angle_rad)

        line = self._get_line_at_angle(
            cut_center_x, cut_center_y, angle_degrees, search_range
        )

        split_pieces = self._get_pieces_after_temp_cut(
            piece, line.coords[0], line.coords[1]
        )
        if len(split_pieces) < 2:
            return 0.0
        return min(p.area for p in split_pieces)

    def _find_all_cuts_by_angle(
        self, piece: Polygon, Area_list: list[float], crust_ratio: float
    ) -> list[tuple]:
        """Iterates through angles and tests every cut to find those that meet precision requirements."""
        all_found_cuts = []

        # Iterate through angles
        for angle in range(0, 180, self.angle_increment):
            bounds = piece.bounds
            search_range = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2

            # Brute-force offsets
            num_offsets = 500
            step_size = search_range / num_offsets

            for i in range(num_offsets):
                offset = -search_range / 2 + i * step_size

                cut_points = self._get_cut_points_from_offset(piece, offset, angle)

                if cut_points:
                    from_p, to_p = cut_points
                    valid, _ = self.cake.cut_is_valid(from_p, to_p)
                    if valid:
                        cake_p, crust_p = self.check_precision(
                            from_p, to_p, Area_list, piece, crust_ratio
                        )

                        # Test against all possible target areas to see if the precision is good enough
                        if cake_p < self.target_area_tolerance:
                            all_found_cuts.append((cake_p, from_p, to_p, crust_p))

        return all_found_cuts

    def _get_cut_points_from_offset(self, piece, offset, angle):
        bounds = piece.bounds
        search_range = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        angle_rad = radians(angle + 90)
        cut_center_x = center_x + offset * cos(angle_rad)
        cut_center_y = center_y + offset * sin(angle_rad)
        line = self._get_line_at_angle(cut_center_x, cut_center_y, angle, search_range)
        return self.find_cuts(line, piece)

    # ---------- Crawl Methods ----------

    def calculate_initial_angle_offset(
        self, piece: Polygon, p1: Point, p2: Point
    ) -> tuple[float, float]:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        angle = degrees(atan2(dy, dx)) % 180

        line_center_x = (p1.x + p2.x) / 2
        line_center_y = (p1.y + p2.y) / 2

        bounds = piece.bounds
        box_center_x = (bounds[0] + bounds[2]) / 2
        box_center_y = (bounds[1] + bounds[3]) / 2

        offset_vec_x = line_center_x - box_center_x
        offset_vec_y = line_center_y - box_center_y
        offset_dist = sqrt(offset_vec_x**2 + offset_vec_y**2)

        offset_angle_rad = radians(angle + 90)
        offset_dir_x = cos(offset_angle_rad)
        offset_dir_y = sin(offset_angle_rad)

        dot_product = offset_vec_x * offset_dir_x + offset_vec_y * offset_dir_y

        signed_offset = copysign(offset_dist, dot_product)

        return angle, signed_offset

    def crawl(
        self,
        piece: Polygon,
        Area_list: list[float],
        crust_ratio: float,
        starting_cut: tuple,
    ) -> tuple:
        initial_angle, initial_offset = self.calculate_initial_angle_offset(
            piece, starting_cut[1], starting_cut[2]
        )

        current_angle = initial_angle
        current_offset = initial_offset
        best_found_cut = starting_cut

        bounds = piece.bounds
        max_dimension = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        step_angle = 3
        step_offset = max_dimension / 200

        for i in range(15):
            best_neighbor_score = (best_found_cut[3] + 0.001) * (
                best_found_cut[0] + 0.1
            )

            for angle_delta in [-step_angle, 0, step_angle]:
                for offset_delta in [-step_offset, 0, step_offset]:
                    if angle_delta == 0 and offset_delta == 0:
                        continue

                    test_angle = current_angle + angle_delta
                    test_offset = current_offset + offset_delta

                    cut_points = self._get_cut_points_from_offset(
                        piece, test_offset, test_angle
                    )
                    if not cut_points:
                        continue

                    from_p, to_p = cut_points
                    is_valid, _ = self.cake.cut_is_valid(from_p, to_p)

                    if is_valid:
                        cake_p, crust_p = self.check_precision(
                            from_p, to_p, Area_list, piece, crust_ratio
                        )
                        if cake_p == float("inf"):
                            continue

                        new_score = (crust_p + 0.001) * (cake_p + 0.1)

                        if new_score < best_neighbor_score:
                            best_neighbor_score = new_score
                            best_found_cut = (cake_p, from_p, to_p, crust_p)
                            current_angle = test_angle
                            current_offset = test_offset

            step_angle /= 2
            step_offset /= 2

            if (
                best_found_cut[0] < self.target_area_tolerance
                and best_found_cut[3] < 0.01
            ):
                break

        return best_found_cut

    # ---------- Main Algorithm ----------

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        piece = max(self.cake.get_pieces(), key=lambda p: p.area)
        crust_ratio = self.cake.get_piece_ratio(piece)
        Total_Area = self.cake.exterior_shape.area
        Area_list = [(Total_Area / self.children) * i for i in range(1, self.children)]

        for k in range(self.children - 1):
            print("-------------------")
            print(f"Cut {k + 1}/{self.children - 1}")
            best_line_list = []
            best_line = [100, None, None, 100]
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            piece_boundary = piece.boundary

            num_candidates = 200
            step_size = piece_boundary.length / num_candidates
            candidates = [
                piece_boundary.interpolate(i * step_size) for i in range(num_candidates)
            ]

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    p1 = candidates[i]
                    p2 = candidates[j]

                    good, _ = self.cake.does_line_cut_piece_well(
                        LineString((p1, p2)), piece
                    )
                    if not good:
                        continue

                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        continue

                    cake_precision, crust_precision = self.check_precision(
                        p1, p2, Area_list, piece, crust_ratio
                    )
                    if cake_precision == float("inf"):
                        continue

                    if best_line[0] > cake_precision:
                        best_line = [cake_precision, p1, p2, crust_precision]

                    if cake_precision < self.target_area_tolerance:
                        best_line_list.append((cake_precision, p1, p2, crust_precision))

            print(f"Found {len(best_line_list)} candidates from the first algorithm.")

            run_angle_fallback = False
            if not best_line_list:
                print("No optimal candidates found. Running angle-based fallback.")
                run_angle_fallback = True
            else:
                # Check the best cut's cake precision from the primary search
                best_primary_cut = min(best_line_list, key=lambda x: x[3])
                if best_primary_cut[3] > 0.02:
                    print(
                        f"Primary search found cuts, but the best precision ({best_primary_cut[3]:.6f}) is not optimal. Running angle-based fallback."
                    )
                    run_angle_fallback = True

            if run_angle_fallback:
                all_found_cuts = self._find_all_cuts_by_angle(
                    piece, Area_list, crust_ratio
                )
                if all_found_cuts:
                    best_line_list.extend(all_found_cuts)
                    print(
                        f"Angle-based search found {len(all_found_cuts)} suitable cuts."
                    )
                else:
                    print("Angle-based search failed to find a suitable cut.")

            bestline = None
            if best_line_list:
                best_line_list.sort(key=lambda x: (x[3] + 0.001) * (x[0] + 0.1))
                bestline = best_line_list[0]
                print("Using best line from candidates or angle search.")
                print(
                    f"Cake Precision: {bestline[0]:.6f}, Crust Precision: {bestline[3]:.6f}"
                )
            elif best_line[1] is not None:
                print("No optimal crust found, using best-effort cut.")
                bestline = best_line

            if bestline:
                print(
                    f"Found initial candidate. CakeP: {bestline[0]:.6f}, CrustP: {bestline[3]:.6f}. Refining..."
                )

                bestline = self.crawl(piece, Area_list, crust_ratio, bestline)

                print("Crawl finished.")
                print(
                    f"Final Cut -> Cake Precision: {bestline[0]:.6f}, Crust Precision: {bestline[3]:.6f}"
                )
            else:
                print("Random strategy as a last resort.")
                a, b = self.random_player.find_random_cut()
                bestline = [100, a, b, 100]

            moves.append((bestline[1], bestline[2]))
            self.cake.cut(bestline[1], bestline[2])

        return moves
