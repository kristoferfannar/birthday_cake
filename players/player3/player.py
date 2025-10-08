from shapely import Point

from players.player import Player
from src.cake import Cake
from .helper_func import find_valid_cuts


class Player3(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        # NEW: Configuration flags for refinement system
        self.use_refinement = True  # Toggle for new optimization
        self.coarse_samples = 60  # For coarse search phase (increased for better coverage)
        self.top_n_refine = 10  # How many candidates to refine (more candidates for better optimization)
        
        # NEW: Parallel processing configuration
        self.use_parallel = True  # Toggle for parallel processing
        self.num_workers = None  # None = auto-detect optimal worker count
        
        # OLD: Keep for backward compatibility
        self.num_samples = 70  # Number of perimeter points to sample (high precision)
        self.cuts = []
        self.original_ratio = cake.get_piece_ratio(cake.get_pieces()[0])  # store original ratio once

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Greedily generate cuts to divide cake into equal pieces."""
        return self._greedy_cutting_strategy()

    def _greedy_cutting_strategy(self) -> list[tuple[Point, Point]]:
        """Main greedy cutting algorithm."""
        cuts = []
        working_cake = self.cake.copy()
        remaining_children = self.children

        while remaining_children > 1:
            # Find the largest piece to cut
            largest_piece = max(working_cake.get_pieces(), key=lambda piece: piece.area)
            target_ratio = 1.0 / remaining_children
            best_cut = self._find_best_cut_for_piece(
                working_cake, largest_piece, target_ratio
            )

            if best_cut is None:
                break  # No valid cut found

            cuts.append(best_cut)
            working_cake.cut(best_cut[0], best_cut[1])
            remaining_children -= 1

        return cuts

    def _find_best_cut_for_piece(
        self, cake: Cake, piece, desired_cut_ratio: float
    ) -> tuple[Point, Point] | None:
        """Find the best cut for a specific piece using optimized perimeter approach."""
        piece_area = piece.area

        # NEW: Use refinement system if enabled
        if self.use_refinement:
            from .refinement import find_valid_cuts_with_refinement
            
            # Try with different tolerances if no cuts found (strict area for homogeneity, relaxed ratio)
            for tolerance_area in [0.15, 0.25, 0.35, 0.5]:
                for tolerance_ratio in [0.03, 0.05, 0.1]:
                    valid_cuts = find_valid_cuts_with_refinement(
                        cake,
                        piece,
                        desired_cut_ratio,
                        piece_area,
                        self.original_ratio,
                        acceptable_area_error=tolerance_area,
                        acceptable_ratio_error=tolerance_ratio,
                        coarse_samples=self.coarse_samples,
                        top_n_to_refine=self.top_n_refine,
                        use_parallel=self.use_parallel,
                        num_workers=self.num_workers
                    )
                    if valid_cuts:
                        break
                if valid_cuts:
                    break
        else:
            # EXISTING: Keep old code path unchanged
            perimeter_points = self._get_perimeter_points_for_piece(piece)
            
            # Use find_valid_cuts with configurable tolerance
            # Try with different tolerances if no cuts found (strict area for homogeneity, relaxed ratio)
            for tolerance_area in [0.15, 0.25, 0.35, 0.5]:
                for tolerance_ratio in [0.03, 0.05, 0.1]:
                    # Use parallel search if enabled
                    if self.use_parallel:
                        from .parallel_search import parallel_find_valid_cuts
                        valid_cuts = parallel_find_valid_cuts(
                            cake,
                            perimeter_points,
                            desired_cut_ratio,
                            piece_area,
                            self.original_ratio,
                            acceptable_area_error=tolerance_area,
                            acceptable_ratio_error=tolerance_ratio,
                            num_workers=self.num_workers
                        )
                    else:
                        valid_cuts = find_valid_cuts(
                            cake,
                            perimeter_points,
                            desired_cut_ratio,
                            piece_area,
                            self.original_ratio,
                            acceptable_area_error=tolerance_area,
                            acceptable_ratio_error=tolerance_ratio
                        )
                    if valid_cuts:
                        break

        if not valid_cuts:
            return None

        # Return the best cut (already sorted by accuracy)
        return valid_cuts[0][:2]  # Return (Point, Point) tuple

    def _get_perimeter_points_for_piece(self, piece) -> list[Point]:
        """Get perimeter points for a specific piece."""
        boundary = piece.exterior
        points = [
            boundary.interpolate(i / self.num_samples, normalized=True)
            for i in range(self.num_samples)
        ]
        return points
