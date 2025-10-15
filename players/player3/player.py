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
        return self._bucket_cutting_strategy()

    def _greedy_cutting_strategy(self) -> list[tuple[Point, Point]]:
        """Main greedy cutting algorithm."""
        cuts = []
        working_cake = self.cake.copy()
        remaining_children = self.children

        while remaining_children > 1:
            print(f"=========================================================")
            print(f"Remaining children: {remaining_children}")

            largest_piece = max(working_cake.get_pieces(), key=lambda piece: piece.area)
            # If this is the first cut, let's try to cut in as close to half as possible
            print(f"Largest piece area: {largest_piece}")
            if remaining_children == self.children:
                potential_ratios = [
                    i / remaining_children
                    for i in range(1, (remaining_children // 2) + 1)
                ]
                target_ratio = max(potential_ratios)
            else:
                # Cut one piece at a time
                target_ratio = self.target_area / largest_piece.area

            best_cut = self._find_best_cut_for_piece(
                working_cake, largest_piece, target_ratio
            )

            if best_cut is None:
                break

            cuts.append(best_cut)
            print(f"Cutting at: {best_cut[0]}, {best_cut[1]}")
            working_cake.cut(best_cut[0], best_cut[1])
            # print("Cake pieces", working_cake.get_pieces())
            print("Piece areas", [p.area for p in working_cake.get_pieces()])
            remaining_children -= 1

        return cuts

    def _bucket_cutting_strategy(self) -> list[tuple[Point, Point]]:
        """Main bucket cutting algorithm using recursive splitting (area fairness only)."""
        cuts = []
        largest_piece = max(self.cake.get_pieces(), key=lambda piece: piece.area)
        self._recursive_bucket_cut(self.cake, largest_piece, self.children, cuts)
        return cuts

    def _recursive_bucket_cut(
        self, full_cake, cake_piece, num_children: int, cuts: list[tuple[Point, Point]]
    ):
        """Recursively divide a cake piece for a given number of children based on area fairness."""
        if num_children <= 1:
            return

        working_cake = full_cake.copy()
        # Only need to test up to half since ratios > 0.5 are symmetric
        potential_ratios = [i / num_children for i in range(1, (num_children // 2) + 1)]
        potential_ratios.sort(reverse=True)
        best_cut = None
        best_ratio_diff = float("inf")
        best_split = None
        best_ratio = None
        print("=======================================================")
        print("Current Piece Area:", cake_piece.area)
        print("Potential Ratios for this piece:", potential_ratios)
        print("Number of Children for this piece:", num_children)

        for ratio in potential_ratios:
            # Get the largets piece each iteration
            print("Current Ratio: ", ratio)
            cut = self._find_best_cut_for_piece(working_cake, cake_piece, ratio)
            print("Is there a cut for RATIO:", ratio, "| CUT:", cut)
            if cut is None:
                continue

            split_pieces = working_cake.cut_piece(cake_piece, cut[0], cut[1])
            if len(split_pieces) != 2:
                # Likely the lsa
                best_cut = cut
                continue

            # Compute the absolute difference from target area ratio
            areas = [piece.area for piece in split_pieces]
            print("This is the ", ratio, "ratio best cut area split ", areas)

            ratios = [working_cake.get_piece_ratio(piece) for piece in split_pieces]
            ratio_diffs = [abs(ratio - self.original_ratio) for ratio in ratios]
            ratio_diff = min(ratio_diffs)

            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_cut = cut
                best_split = split_pieces
                best_ratio = ratio

        if best_cut is None:
            print("No valid cut found for any ratio, stopping.")
            return

        # Assign children proportional to the chosen ratio
        left_children = int(round(best_split[0].area / self.target_area))
        right_children = num_children - left_children
        print("left children: ", left_children)
        print("right children: ", right_children)

        # Record the chosen cut
        cuts.append(best_cut)
        areas = [piece.area for piece in best_split]
        working_cake.cut(best_cut[0], best_cut[1])
        print("BEST CUT ", best_cut, " and AREA: ", areas, "| RATIO: ", best_ratio)

        # Recursively cut each subpiece
        self._recursive_bucket_cut(working_cake, best_split[0], left_children, cuts)
        # make cuts from the left piece for the working cake using cuts
        print("Working Cake Pieces", working_cake.get_pieces())

        # If valid cut, cut the working cake (So update the left half to the right half)
        for cut in cuts:
            not_already_cut, _ = working_cake.cut_is_valid(cut[0], cut[1])
            if not_already_cut:
                working_cake.cut(cut[0], cut[1])

        set_pieces = working_cake.get_pieces()
        print("Working Cake Areas", [p.area for p in set_pieces])

        self._recursive_bucket_cut(working_cake, best_split[1], right_children, cuts)

    def _find_best_cut_for_piece(
        self, cake: Cake, piece, desired_cut_ratio: float
    ) -> tuple[Point, Point] | None:
        """Find the best cut for a specific piece using binary search."""
        perimeter_points = self._get_perimeter_points_for_piece(piece)

        valid_cuts = find_valid_cuts_binary_search(
            cake,
            perimeter_points,
            # self.target_area * self.children * desired_cut_ratio,
            piece.area * desired_cut_ratio,
            self.original_ratio,
            piece
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
