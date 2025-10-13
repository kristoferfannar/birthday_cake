from shapely import Point, LineString
from shapely.ops import split

# import random
from players.player import Player
from players.random_player import RandomPlayer
from src.cake import Cake
import src.constants as c
import math


class AngleBSTPlayer(Player):
    """
    Strategy:
    - Samples random points along the cake crust (outer edge)
    - Computes crust density to find crust-heavy regions
    - Picks starting point p1 from densest crust points
    - Chooses p2 and evaluates cut quality using weighted scoring
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.random_player = RandomPlayer(children, cake, cake_path)
        self.valid_line_pair = dict()
        self.tolerance = 0.05
        self.max_crust_points = 3  # how many top crust-heavy points to consider

    # ---------- Utility metrics ----------

    def get_crust_length(self, piece):
        """Length of piece boundary that lies on the exterior crust."""
        return piece.boundary.intersection(self.cake.exterior_shape.boundary).length

    def get_crust_density(self, piece, p: Point):
        """Approximate local crust density near a boundary point. This is only meaningful for the initial cut."""
        crust_section = self.cake.exterior_shape.boundary.buffer(c.CRUST_SIZE * 2)
        nearby = piece.boundary.intersection(crust_section)
        return nearby.length

    def get_piece(self, p1, p2):
        """Find which piece both points belong to."""
        for piece in self.cake.get_pieces():
            if self.cake.point_lies_on_piece_boundary(
                p1, piece
            ) and self.cake.point_lies_on_piece_boundary(p2, piece):
                return piece
        return None

    # Inside CrustOptimizingPlayer class

    def check_precision(self, p1, p2, Area_list, piece, crust_ratio):
        """Return a tuple containing the absolute difference between the piece area and the closest
        target area, and the absolute difference between the new crust ratio and the target crust ratio."""

        split_pieces = self.cake.cut_piece(piece, p1, p2)
        if len(split_pieces) < 2:
            return float("inf"), float("inf")  # Return a tuple for invalid cuts

        if split_pieces[0].area < split_pieces[1].area:
            Area_piece = split_pieces[0].area
            closest_target_area = min(Area_list, key=lambda a: abs(a - Area_piece))
            cake_precision = (
                abs(Area_piece - closest_target_area) / self.cake.exterior_shape.area
            )

            new_piece1_crust_ratio = self.cake.get_piece_ratio(split_pieces[0])

            crust_precision = abs(new_piece1_crust_ratio - crust_ratio)
            return cake_precision, crust_precision
        else:
            Area_piece = split_pieces[1].area
            closest_target_area = min(Area_list, key=lambda a: abs(a - Area_piece))
            cake_precision = (
                abs(Area_piece - closest_target_area) / self.cake.exterior_shape.area
            )

            new_piece1_crust_ratio = self.cake.get_piece_ratio(split_pieces[1])

            crust_precision = abs(new_piece1_crust_ratio - crust_ratio)
            return cake_precision, crust_precision

    def get_weight(self, p1, p2, piece, Goal_ratio):
        """Compute weight for a given cut — based on area ratio deviation."""
        split_pieces = self.cake.cut_piece(piece, p1, p2)
        if len(split_pieces) != 2:
            return 0
        r1 = self.cake.get_piece_ratio(split_pieces[0])
        r2 = self.cake.get_piece_ratio(split_pieces[1])
        weight = 0.5 * (1 - abs(Goal_ratio - r1)) + 0.5 * (1 - abs(Goal_ratio - r2))
        return weight

    # ---------- Binary Search Algorithm ----------

    def point_side(self, p1, p2, p3):
        # Computes if p3 is on the left or right of line p1-p2
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        if cross > 0:
            return "left"
        elif cross < 0:
            return "right"
        else:
            return "collinear"

    def line_bound_given_PA(self, mid, piece, degree):
        left_bound, below_bound, right_bound, above_bound = piece.bounds
        # print("l, b, r, a ", left_bound, below_bound, right_bound, above_bound)
        upper_left = (left_bound, above_bound)
        width = right_bound - left_bound
        height = above_bound - below_bound
        bar_dimension = (width**2 + height**2) ** 0.5

        # First, we find the boundary of the cake, then starting from
        # the upper left, we find the distance to the lower right,
        # this is the bar_dimension varaible. At this point, we try to cut
        # at mid*bar_dimension starting from the upper right point
        # with the correct degree.

        rad = math.radians(degree)
        rad_perp = rad + math.pi / 2

        # Initial point for our line
        point_x = upper_left[0] + mid * bar_dimension * math.cos(rad_perp)
        point_y = upper_left[1] - mid * bar_dimension * math.sin(rad_perp)

        # These give the endpoints, Note that since we multiply by bar_dimension, this will be a huge line
        offx = math.cos(rad) * bar_dimension
        offy = math.sin(rad) * bar_dimension

        p1 = (point_x - offx, point_y - offy)
        p2 = (point_x + offx, point_y + offy)
        cut_line = LineString([p1, p2])
        return cut_line, p1, p2

    def find_actual_cuts(self, line, piece):
        intersection = line.intersection(piece.boundary)

        if intersection.is_empty:
            return False, 0, 0
        elif intersection.geom_type == "Point":
            return False, 0, 0

        points = []
        if intersection.geom_type == "LineString":
            return False, 0, 0
        elif intersection.geom_type == "MultiPoint":
            points = list(intersection.geoms)

        if len(points) < 2:
            return False, 0, 0

        return True, points[0], points[1]

    def get_area_with_scaler(self, mid, piece, degree):
        line, p1, p2 = self.line_bound_given_PA(mid, piece, degree)
        # This first one find a "big sweeping cut" line that just cuts at the right degree and mid percentage

        found, p1, p2 = self.find_actual_cuts(line, piece)
        # Given that first line, we find the actual endpoints

        if not found:
            return False, 0, 0, 0

        pieces = split(piece, line)

        if len(pieces.geoms) != 2:
            return False, 0, 0, 0

        piece1, piece2 = pieces.geoms

        centroid1 = piece1.centroid

        if self.point_side(Point(p1), Point(p2), centroid1) == "left":
            return True, piece1.area, Point(p1), Point(p2)
        else:
            return True, piece2.area, Point(p1), Point(p2)

    def Search_for_angle_area(self, degree, targt_area, piece, Area_list, crust_ratio):
        # Search for a cut with with valid area and cut made at set degree

        lower_scaler = 0
        upper_scaler = 1

        for _ in range(15):
            # Mid is the "percent" if the way through we cut the cake that we set our angle
            mid = (lower_scaler + upper_scaler) / 2
            valid_BS, area, p1, p2 = self.get_area_with_scaler(mid, piece, degree)

            if not valid_BS:
                lower_scaler = mid
            if area - targt_area < 0.0001:
                break
            if area < targt_area:
                lower_scaler = mid
            elif area > targt_area:
                upper_scaler = mid
            else:
                break

        # Serch for a piece with the set midpoint. If we find one, we run the same code from the standard algorithm
        valid_BS, area, p1, p2 = self.get_area_with_scaler(mid, piece, degree)
        if not valid_BS:
            return False, 0

        valid, _ = self.cake.cut_is_valid(p1, p2)
        if not valid:
            # print(p1, p2)
            return False, 0
        # print(p1,p2)

        cake_precision, crust_precision = self.check_precision(
            p1, p2, Area_list, piece, crust_ratio
        )
        if cake_precision == float("inf"):
            return False, 0

        if cake_precision < 0.0005:
            # print(cake_precision)
            return True, (cake_precision, p1, p2, crust_precision)
        return False, 0

    def Binary_Search(self, Area_list, piece, crust_ratio):
        degree_list = []
        step = 0.1
        for i in range(int((360 - 271) / step)):
            degree_list += [271 + step * i]

        valid_cuts = []
        for degree in degree_list:
            for area in Area_list:
                # Search for a cut with with valid area and cut made at set degree
                found, cut = self.Search_for_angle_area(
                    degree, area, piece, Area_list, crust_ratio
                )
                if found:
                    valid_cuts += [cut]
        return valid_cuts

    # ---------- Main Algorithm ----------

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        # num_p1 = 125
        # num_p2 = 300

        piece = max(self.cake.get_pieces(), key=lambda p: p.area)
        crust_ratio = self.cake.get_piece_ratio(piece)

        Total_Area = self.cake.exterior_shape.area
        Area_list = []
        for i in range(1, self.children):
            Area_list += [(Total_Area / self.children) * i]
        for k in range(self.children - 1):
            print(f"Cut {k + 1}/{self.children - 1}")

            best_line_list = []
            best_line = [100, None, None, 100]
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            piece_boundary = piece.boundary

            num_candidates = 3  # You can adjust this number
            step_size = piece_boundary.length / num_candidates
            candidates = [
                piece_boundary.interpolate(i * step_size) for i in range(num_candidates)
            ]

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    p1 = candidates[i]
                    p2 = candidates[j]
                    # print(p1, p2)
                    good, _ = self.cake.does_line_cut_piece_well(
                        LineString((p1, p2)), piece
                    )

                    if not good:
                        # print(_)
                        continue

                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        # print(p1, p2)
                        continue
                    # print(p1,p2)
                    cake_precision, crust_precision = self.check_precision(
                        p1, p2, Area_list, piece, crust_ratio
                    )
                    if cake_precision == float("inf"):
                        continue

                    if best_line[0] > cake_precision:
                        best_line = [cake_precision, p1, p2, crust_precision]

                    if cake_precision < 0.0005:
                        best_line_list.append((cake_precision, p1, p2, crust_precision))

            # print(best_line)
            print(len(best_line_list))

            best_line_found = False

            # if len(best_line_list) > 0:
            #     print("lol")
            #     best_line_list.sort(key=lambda x: (x[3] + 0.001) * (x[0] + 0.1))
            #     bestline = best_line_list[0]
            #     best_line_found = True
            # elif best_line[1] is not None:
            #     print("No optimal crust found")
            #     bestline = best_line
            #     best_line_found = True

            if not best_line_found:
                print("Trying Binary Search")
                best_line_list = self.Binary_Search(Area_list, piece, crust_ratio)

            if not best_line_found and len(best_line_list) > 0:
                print("BS worked")
                best_line_list.sort(key=lambda x: (x[3] + 0.001) * (x[0] + 0.1))
                bestline = best_line_list[0]
                best_line_found = True

            if not best_line_found:
                print("random strategy")
                a, b = self.random_player.find_random_cut()
                bestline = [100, a, b, 100]
            # --- Step 4: Execute cut ---
            # print(bestline)
            moves.append((bestline[1], bestline[2]))
            self.cake.cut(bestline[1], bestline[2])

        return moves
