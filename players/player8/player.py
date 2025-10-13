from shapely.geometry import Point, LineString
from shapely.ops import split
from players.player import Player
from src.cake import Cake
import src.constants as c


class Player8(Player):
    """
    Player8 - Adaptive boundary search strategy with global crust balance feedback.
    Inspired by Player9 logic.

    This strategy adaptively selects cuts around the cake’s boundary to balance
    both area and crust ratio across pieces. It evaluates possible cuts based
    on a weighted combination of area accuracy, crust difference, and global
    balance to achieve fair divisions.
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        """
        Initializes Player8 with weighting preferences for area and crust fairness.

        Args:
            children: Number of cake recipients.
            cake: The cake object to be divided.
            cake_path: Optional path for serialized cake state.
        """
        super().__init__(children, cake, cake_path)
        self.area_weight = 0.7
        self.crust_weight = 0.23

    def _refine_cut(self, piece, p1, p2, t_area, delta=0.01, steps=4):
        """
        Fine-tunes the position of a proposed cut by making small local
        adjustments to its endpoints. The goal is to minimize the difference
        between the smaller resulting piece’s area and the target area.

        Args:
            piece: The piece being divided.
            p1, p2: The initial proposed cut endpoints.
            t_area: Target area for one of the resulting pieces.
            delta: Step size for each boundary movement.
            steps: Number of iterations for refinement.

        Returns:
            Tuple of the optimized endpoints (p1, p2).
        """
        boundary = piece.boundary

        def move(pt, step):
            s = boundary.project(pt)
            s_new = min(max(0.0, s + step * delta * boundary.length), boundary.length)
            return boundary.interpolate(s_new)

        best = (
            p1,
            p2,
            abs(
                min(split(piece, LineString([p1, p2])).geoms, key=lambda g: g.area).area
                - t_area
            ),
        )

        for _ in range(steps):
            improved = False
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    q1, q2 = move(best[0], s1), move(best[1], s2)
                    parts = split(piece, LineString([q1, q2]))
                    if len(parts.geoms) != 2:
                        continue
                    small = min(parts.geoms[0].area, parts.geoms[1].area)
                    err = abs(small - t_area)
                    if err < best[2]:
                        best = (q1, q2, err)
                        improved = True
            if not improved:
                break
        return best[0], best[1]

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """
        Determines the sequence of cuts to divide the cake into fair pieces.
        The method iteratively selects boundary points that minimize a composite
        score based on area fairness, crust ratio difference, and global crust balance.

        Returns:
            A list of (Point, Point) tuples representing all executed cuts.
        """
        total_area = self.cake.exterior_shape.area
        target_area = total_area / self.children
        target_areas = [target_area for _ in range(self.children - 1)]
        moves: list[tuple[Point, Point]] = []

        for cut_index, t_area in enumerate(target_areas):
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            boundary = piece.boundary
            boundary_len = boundary.length

            # Adjust candidate density based on iteration stage
            if cut_index == 0:
                num_candidates = min(500, max(200, int(boundary_len / 0.4)))
            else:
                num_candidates = max(150, min(220, int(boundary_len / 1.2)))

            step = boundary.length / num_candidates
            points = [boundary.interpolate(i * step) for i in range(num_candidates)]

            best_score = float("inf")
            best_cut = None

            perfect_thresh = 0.001
            area_prune_thresh = 0.3

            # Evaluate global crust balance among all existing pieces
            ratios = [self.cake.get_piece_ratio(p) for p in self.cake.get_pieces()]
            global_mean = sum(ratios) / len(ratios)
            global_weight = 0.25

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    p1 = points[i]
                    p2 = points[j]

                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        continue

                    line = LineString([p1, p2])
                    parts = split(piece, line)
                    if len(parts.geoms) != 2:
                        continue

                    a1, a2 = parts.geoms
                    a1_area, a2_area = a1.area, a2.area
                    small = min(a1_area, a2_area)

                    if abs(small - t_area) > t_area * area_prune_thresh:
                        continue
                    if a1_area < c.MIN_PIECE_AREA or a2_area < c.MIN_PIECE_AREA:
                        continue

                    area_error = abs(small - t_area) / t_area
                    r1 = self.cake.get_piece_ratio(a1)
                    r2 = self.cake.get_piece_ratio(a2)
                    crust_diff = abs(r1 - r2)
                    crust_diff = min(crust_diff, 1.0)

                    # Adjust weighting dynamically as cuts progress
                    alpha = cut_index / max(1, self.children - 1)
                    base_area_w = getattr(
                        self, "area_weight_override", self.area_weight
                    )
                    base_crust_w = getattr(
                        self, "crust_weight_override", self.crust_weight
                    )

                    area_weight = base_area_w + (1 - alpha) * 0.1
                    crust_weight = base_crust_w * (1 + 0.5 * alpha)

                    r_small = r1 if a1_area <= a2_area else r2
                    global_balance_penalty = abs(r_small - global_mean)

                    # Composite score combining area, crust, and global balance
                    score = (
                        area_weight * area_error
                        + crust_weight * crust_diff
                        + global_weight * global_balance_penalty
                    )

                    if cut_index == 0 and small > t_area:
                        score *= 1.25
                    if area_error > 0.3 or crust_diff > 0.6:
                        continue

                    if score < best_score:
                        best_score = score
                        best_cut = (p1, p2)

                        if area_error < perfect_thresh and crust_diff < 0.01:
                            break
                if best_score < float("inf") and best_score < perfect_thresh:
                    break

            if best_cut is None:
                print(f"[WARN] No valid cut found at step {cut_index + 1}")
                break

            best_cut = self._refine_cut(piece, best_cut[0], best_cut[1], t_area)

            # Apply final chosen cut
            self.cake.cut(best_cut[0], best_cut[1])
            moves.append(best_cut)

            # print(
            #     f"[CUT {cut_index + 1}] "
            #     f"Points=({round(best_cut[0].x, 3)}, {round(best_cut[0].y, 3)}) → "
            #     f"({round(best_cut[1].x, 3)}, {round(best_cut[1].y, 3)}) | "
            #     f"Score={best_score:.4f}, "
            #     f"Target={t_area:.3f}, "
            #     f"PieceArea={piece.area:.3f}, "
            #     f"GlobalMeanRatio={global_mean:.3f}"
            # )

        return moves
