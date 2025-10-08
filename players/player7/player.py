from shapely import Point, wkb

from players.player import Player, PlayerException
from src.cake import Cake

def copy_geom(g):
    return wkb.loads(wkb.dumps(g))
class Player7(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

        self.target_area = cake.get_area() / children

        total_crust_area = cake.get_area() - cake.interior_shape.area
        self.target_crust_ratio = total_crust_area / cake.get_area()

        self.moves: list[tuple[Point, Point]] = []
        
        # Configurable parameters
        self.top_k_cuts = 5  # Number of top cuts to optimize
        self.optimization_iterations = 100  # Number of optimization iterations
        self.max_area_deviation = 0.25  # Maximum area deviation tolerance
        self.sample_step = 1  # Step size for sample points

    def copy_ext(self, cake):
        new = object.__new__(Cake)
        new.exterior_shape = self.cake.exterior_shape
        new.interior_shape = self.cake.interior_shape
        new.exterior_pieces = [copy_geom(p) for p in self.cake.exterior_pieces]

        return new
        
    def evaluate_cut(self, from_p: Point, to_p: Point) -> float:
        """Evaluate how good a cut is by measuring deviation from the target crust ratio.
        Only considers valid cuts where the resulting pieces are within tolerance of the target area.
        Returns the sum of squared differences from the target crust ratio."""

        # Copy necessary to avoid mutating the original cake
        cake_copy = self.copy_ext(self.cake)

        try:
            cake_copy.cut(from_p, to_p)

            # First, enforce area tolerance for all resulting pieces.
            for piece in cake_copy.exterior_pieces:
                piece_size = piece.area
                area_multiple = round(piece_size / self.target_area)
                nearest_area_multiple = area_multiple * self.target_area
                area_deviation = abs(piece_size - nearest_area_multiple)
                if area_deviation > self.max_area_deviation:
                    return float("inf")

            # If all pieces are within tolerance, score using crust ratio deviation
            ratio_deviation_total = 0.0
            for piece in cake_copy.exterior_pieces:
                interior_ratio = cake_copy.get_piece_ratio(piece)
                piece_crust_ratio = 1 - interior_ratio
                crust_ratio_deviation = piece_crust_ratio - self.target_crust_ratio
                ratio_deviation_total += crust_ratio_deviation ** 2

            return ratio_deviation_total

        except Exception:
            return float("inf")

    def get_sample_points(self, piece, step: float = None) -> list[Point]:
        """Get sample points along the piece boundary.
        For each edge: include the two vertices, the midpoint, and then every `step` cm along the edge.
        Also includes points from previous cuts that lie on this piece's boundary.
        """
        if step is None:
            step = self.sample_step
            
        coords = list(piece.exterior.coords[:-1])  # Exclude the duplicate last point
        raw_points: list[tuple[float, float]] = []
        
        # Add existing cut endpoints that lie on this piece's boundary
        for move in self.moves:
            for point in move:
                if self.cake.point_lies_on_piece_boundary(point, piece):
                    raw_points.append((point.x, point.y))

        for i in range(len(coords)):
            next_i = (i + 1) % len(coords)
            x1, y1 = coords[i]
            x2, y2 = coords[next_i]

            # Add the starting vertex of the edge
            raw_points.append((x1, y1))

            dx = x2 - x1
            dy = y2 - y1
            length = (dx * dx + dy * dy) ** 0.5

            # Add midpoint
            mx = x1 + 0.5 * dx
            my = y1 + 0.5 * dy
            raw_points.append((mx, my))

            # Add points every `step` cm along the edge, excluding endpoints
            if step > 0 and length > 0:
                k = 1
                while k * step < length:
                    t = (k * step) / length
                    px = x1 + t * dx
                    py = y1 + t * dy
                    raw_points.append((px, py))
                    k += 1

        # Deduplicate points that may coincide (e.g., when midpoint aligns with a step)
        seen = set()
        sample_points: list[Point] = []
        for x, y in raw_points:
            key = (round(x, 6), round(y, 6))
            if key in seen:
                continue
            seen.add(key)
            sample_points.append(Point(x, y))

        return sample_points

    def find_best_cut(self) -> tuple[Point, Point]:
        """Find the cut that minimizes deviation from target crust area by optimizing top 3 cuts."""
        pieces = self.cake.get_pieces()
        if not pieces:
            raise PlayerException("no pieces available to cut")
        
        piece = max(pieces, key=lambda p: p.area)

        # Get sample points along the piece boundary
        sample_points = self.get_sample_points(piece)
        print(f"Found {len(sample_points)} sample points")
        
        min_len = 1.0
        # Collect all valid cuts with their scores
        candidate_cuts = []
        for i in range(len(sample_points)):
            for j in range(i + 1, len(sample_points)):
                if (sample_points[i].distance(sample_points[j]) < min_len):
                    continue  # Skip cuts that are too short

                from_p = sample_points[i]
                to_p = sample_points[j]

                score = self.evaluate_cut(from_p, to_p)
                if score != float("inf"):  # Only consider valid cuts
                    candidate_cuts.append((score, from_p, to_p))

        if not candidate_cuts:
            raise PlayerException("could not find a valid cut")

        # Sort by score and take top k
        print(f"Found {len(candidate_cuts)} candidate cuts")
        candidate_cuts.sort(key=lambda x: x[0])
        top_cuts = candidate_cuts[:self.top_k_cuts]

        # Optimize each of the top cuts
        best_optimized_score = float("inf")
        best_optimized_cut = None
        
        for original_score, from_p, to_p in top_cuts:
            optimized_from_p, optimized_to_p = self.optimize_cut(from_p, to_p, iterations=self.optimization_iterations)
            optimized_score = self.evaluate_cut(optimized_from_p, optimized_to_p)
            
            if optimized_score < best_optimized_score:
                best_optimized_score = optimized_score
                best_optimized_cut = (optimized_from_p, optimized_to_p)

        return best_optimized_cut

    def get_boundary_direction(self, piece, point: Point) -> tuple[float, float]:
        """Get the direction vector along the piece boundary at the given point."""
        boundary = piece.boundary

        # Find the closest point on the boundary
        closest_point = boundary.interpolate(boundary.project(point))

        # Get a small offset along the boundary in both directions
        distance = boundary.project(closest_point)
        offset = 0.01

        # Get points slightly before and after
        before_point = boundary.interpolate((distance - offset) % boundary.length)
        after_point = boundary.interpolate((distance + offset) % boundary.length)

        # Calculate direction vector
        dx = after_point.x - before_point.x
        dy = after_point.y - before_point.y

        # Normalize the direction vector
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            return dx / length, dy / length
        else:
            return 0.0, 0.0

    def optimize_cut(
        self, from_p: Point, to_p: Point, iterations: int = 20
    ) -> tuple[Point, Point]:
        """Optimize a cut by moving points along the boundary direction."""
        best_cut = (from_p, to_p)
        best_score = self.evaluate_cut(from_p, to_p)

        # If the initial cut is invalid, return it as-is
        if best_score == float("inf"):
            return best_cut

        # If the initial score is 0 (perfect), skip optimization
        if best_score == 0:
            return best_cut

        # Find the piece that this cut would affect
        cuttable_piece, _ = self.cake.get_cuttable_piece(from_p, to_p)
        if not cuttable_piece:
            return best_cut

        current_from = Point(from_p.x, from_p.y)
        current_to = Point(to_p.x, to_p.y)

        # Step size for optimization (start larger, decrease over time)
        initial_step_size = self.sample_step/2

        for iteration in range(iterations):
            # Calculate step size (decreases over iterations)
            step_size = initial_step_size * (1 - iteration / iterations)

            improved = False

            # Try moving each point along the boundary in both directions
            for direction in [-1, 1]:
                # Get boundary direction at current from point
                from_dx, from_dy = self.get_boundary_direction(
                    cuttable_piece, current_from
                )

                # Move the from point along the boundary
                new_from = Point(
                    current_from.x + direction * step_size * from_dx,
                    current_from.y + direction * step_size * from_dy,
                )

                new_score = self.evaluate_cut(new_from, current_to)

                if new_score < best_score:
                    best_score = new_score
                    best_cut = (new_from, current_to)
                    current_from = new_from
                    improved = True
                    break

                # Get boundary direction at current to point
                to_dx, to_dy = self.get_boundary_direction(cuttable_piece, current_to)

                # Move the to point along the boundary
                new_to = Point(
                    current_to.x + direction * step_size * to_dx,
                    current_to.y + direction * step_size * to_dy,
                )

                new_score = self.evaluate_cut(current_from, new_to)

                if new_score < best_score:
                    best_score = new_score
                    best_cut = (current_from, new_to)
                    current_to = new_to
                    improved = True
                    break

            # If no improvement found, continue with next iteration
            if not improved:
                continue

        return best_cut

    def get_cuts(self) -> list[tuple[Point, Point]]:
        self.moves.clear()  # Reset moves list

        for cut in range(self.children - 1):
            print(f"Finding cut number {cut + 1}")
            optimized_from_p, optimized_to_p = self.find_best_cut()
            
            self.moves.append((optimized_from_p, optimized_to_p))

            # Simulate the cut on our cake to maintain accurate state
            self.cake.cut(optimized_from_p, optimized_to_p)

        return self.moves
