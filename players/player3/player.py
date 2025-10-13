from shapely import Point

from players.player import Player
from src.cake import Cake
from .helper_func import find_valid_cuts_binary_search


class Player3(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.num_samples = 500
        self.original_ratio = cake.get_piece_ratio(cake.get_pieces()[0])
        self.target_area = sum(p.area for p in self.cake.get_pieces()) / self.children

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Greedily generate cuts to divide cake into equal pieces."""
        return self._greedy_cutting_strategy()

    def _greedy_cutting_strategy(self) -> list[tuple[Point, Point]]:
        """Main greedy cutting algorithm."""
        cuts = []
        working_cake = self.cake.copy()
        remaining_children = self.children

        while remaining_children > 1:
            print(f"Remaining children: {remaining_children}")

            largest_piece = max(working_cake.get_pieces(), key=lambda piece: piece.area)
            target_ratio = 1.0 / remaining_children
            best_cut = self._find_best_cut_for_piece(
                working_cake, largest_piece, target_ratio
            )

            if best_cut is None:
                break

            cuts.append(best_cut)
            working_cake.cut(best_cut[0], best_cut[1])
            remaining_children -= 1

        return cuts

    def _find_best_cut_for_piece(
        self, cake: Cake, piece, desired_cut_ratio: float
    ) -> tuple[Point, Point] | None:
        """Find the best cut for a specific piece using binary search."""
        perimeter_points = self._get_perimeter_points_for_piece(piece)

        valid_cuts = find_valid_cuts_binary_search(
            cake,
            perimeter_points,
            self.target_area,
            self.original_ratio,
        )

        if not valid_cuts:
            return None

        return valid_cuts[0]

    def _get_perimeter_points_for_piece(self, piece) -> list[Point]:
        """Get perimeter points for a specific piece."""
        boundary = piece.exterior
        points = [
            boundary.interpolate(i / self.num_samples, normalized=True)
            for i in range(self.num_samples)
        ]
        return points
