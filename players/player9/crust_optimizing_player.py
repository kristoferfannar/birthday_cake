from shapely import Point, LineString
import random
from players.player import Player
from players.random_player import RandomPlayer
from src.cake import Cake
import src.constants as c


class CrustOptimizingPlayer(Player):
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
            if self.cake.point_lies_on_piece_boundary(p1, piece) and \
               self.cake.point_lies_on_piece_boundary(p2, piece):
                return piece
        return None

# Inside CrustOptimizingPlayer class

    def check_precision(self, p1, p2, Area_list, piece, crust_ratio):
        """Return a tuple containing the absolute difference between the piece area and the closest
        target area, and the absolute difference between the new crust ratio and the target crust ratio."""

        split_pieces = self.cake.cut_piece(piece, p1, p2)
        if len(split_pieces) < 2:
            return float("inf"), float("inf") # Return a tuple for invalid cuts

        Area_piece = split_pieces[0].area
        closest_target_area = min(Area_list, key=lambda a: abs(a - Area_piece))
        cake_precision = abs(Area_piece - closest_target_area) / self.cake.exterior_shape.area
        
        new_piece1_crust_ratio = self.cake.get_piece_ratio(split_pieces[0])
        new_piece2_crust_ratio = self.cake.get_piece_ratio(split_pieces[1])
        
        crust_precision = abs(new_piece1_crust_ratio - crust_ratio) + abs(new_piece2_crust_ratio - crust_ratio)
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

    # ---------- Main Algorithm ----------

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        num_p1 = 100 
        num_p2 = 200
        
        piece = max(self.cake.get_pieces(), key=lambda p: p.area)
        crust_ratio = self.cake.get_piece_ratio(piece)
        

        Total_Area = self.cake.exterior_shape.area
        Area_list = []
        for i in range(1, self.children):
            Area_list += [(Total_Area / self.children) * i]
        for k in range(self.children - 1):
            print(f"Cut {k+1}/{self.children-1}")

            best_line_list = []
            best_line = [100, None, None, 100]
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            piece_boundary = piece.boundary

            step_size = piece_boundary.length / num_p1
            p1_candidates = [piece_boundary.interpolate(i * step_size) for i in range(num_p1)]
            step_size = piece_boundary.length / num_p2
            p2_candidates = [piece_boundary.interpolate(i * step_size) for i in range(num_p1)]
            for p1 in p1_candidates:
                for p2 in p2_candidates:
                    good, _ = self.cake.does_line_cut_piece_well(
                        LineString((p1, p2)), piece
                    )
                    
                    if not good:
                        continue

                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        continue

                    cake_precision, crust_precision = self.check_precision(p1, p2, Area_list, piece, crust_ratio)
                    if cake_precision == float("inf"):
                        continue

                    if best_line[0] > cake_precision:
                        bestline = [cake_precision, p1, p2, crust_precision]
                    
                    if cake_precision < 0.05:
                        best_line_list.append((cake_precision, p1, p2, crust_precision))

            if len(best_line_list) > 0:
                best_line_list.sort(key=lambda x: (x[3] + 0.000001) * (x[0] + 0.000001))
                best_line = best_line_list[0][1], best_line_list[0][2]
            else:
                print("No crust found")
                best_line = bestline[1], bestline[2]

            # --- Step 4: Execute cut ---
            print(best_line)
            moves.append(best_line)
            self.cake.cut(best_line[0], best_line[1])

        return moves