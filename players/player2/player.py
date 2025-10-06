from __future__ import annotations
from typing import List, Tuple, Optional
from math import inf

from shapely import Point, Polygon

from players.player import Player, PlayerException
import src.constants as c
from src.cake import Cake


class Player2(Player):
    """
    Player2: Divide-&-Conquer Backtracking (beam search + pruning)
    with greedy fill to guarantee exactly (children - 1) cuts.

    - Primary objective: equal areas per child.
    - Secondary (soft): interior proportion closeness to cake average.
    - Strategy:
        1) Search: recursive split with preferred even/odd child counts; keep top-K cuts.
        2) Fill: if search returns fewer than needed, iteratively split the largest
           remaining piece with the best approximate equal-area cut until we have n-1 cuts.
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        """
        Initialize Player2 with the number of children, the cake object, and optional parameters.

        Args:
            children (int): Number of children to divide the cake for.
            cake (Cake): The cake object to be divided.
            cake_path (Optional[str]): Path to the cake file (if any).
            beam_width (int): Number of top candidates to keep during beam search.
            debug (bool): Whether to enable debug logging.
        """
        super().__init__(children, cake, cake_path)

        # Calculate total area, target area per child, and average interior ratio
        base_poly = self.cake.get_pieces()[0]
        self.total_area = base_poly.area
        self.target_area = self.total_area / children
        self.avg_ratio = self.cake.interior_shape.area / self.total_area

        self.beam_width = 5
        self.debug = False

        # Soft tolerances for area and ratio errors
        self.area_tol = getattr(c, "PIECE_SPAN_TOL", 0.5)  # cm^2 if present
        self.ratio_tol = 0.05

    # ---------- public ----------
    def get_cuts(self) -> list[tuple[Point, Point]]:
        """
        Generate the cuts required to divide the cake into `children` pieces.

        Returns:
            List[Tuple[Point, Point]]: List of cuts as tuples of points.
        """
        # Work on a local copy of the cake
        work = self.cake.copy()
        root_piece = work.get_pieces()[0]

        # Perform recursive search to generate a cutting plan
        _, planned = self._search(
            self._wrap_subcake(work, root_piece),
            pieces_needed=self.children,
            best_so_far=inf,
            depth=0,
        )

        cuts: list[tuple[Point, Point]] = list(planned)

        # If the plan is incomplete, use greedy filling to add remaining cuts
        while len(cuts) < self.children - 1:
            remaining_children = self.children - len(cuts)
            best = self._best_greedy_cut_for(work, remaining_children)
            if not best:
                break  # No further valid cuts found
            from_p, to_p = best
            cuts.append((from_p, to_p))
            work.cut(from_p, to_p)  # Update the working cake

        # If still incomplete, use fallback random valid cuts
        if len(cuts) != self.children - 1:
            if self.debug:
                print(f"[Player2] Had {len(cuts)} cuts; attempting any-valid fallback.")
            if not self._fill_with_any_valid(work, cuts):
                raise PlayerException(
                    f"Player2: could not assemble {self.children - 1} cuts (got {len(cuts)})."
                )

        return cuts

    # ---------- beam-search backtracking ----------
    def _search(
        self,
        cake: Cake,
        pieces_needed: int,
        best_so_far: float,
        depth: int,
    ) -> Tuple[float, List[Tuple[Point, Point]]]:
        """
        Perform recursive beam-search backtracking to find the best cuts.

        Args:
            cake (Cake): The current cake state.
            pieces_needed (int): Number of pieces needed.
            best_so_far (float): Best score found so far.
            depth (int): Current recursion depth.

        Returns:
            Tuple[float, List[Tuple[Point, Point]]]: Best score and corresponding cuts.
        """
        piece = cake.get_pieces()[0]
        piece_area = piece.area

        if self.debug:
            print(
                f"{'  ' * depth}[search] depth={depth} need={pieces_needed} area={piece_area:.3f}"
            )

        # Base case: if only one piece is needed, score the leaf
        if pieces_needed == 1:
            return self._leaf_score(cake), []

        # Generate preferred and neighboring split counts
        m = pieces_needed
        preferred = [(m // 2, m - (m // 2))]
        neighbors: List[Tuple[int, int]] = []
        if m > 2:
            a0 = preferred[0][0]
            neighbors.append((max(1, a0 - 1), m - max(1, a0 - 1)))
            neighbors.append((min(m - 1, a0 + 1), m - min(m - 1, a0 + 1)))

        splits: List[Tuple[int, int]] = []
        seen = set()
        for a, b in preferred + neighbors:
            if a <= 0 or b <= 0:
                continue
            key = (min(a, b), max(a, b))
            if key not in seen:
                splits.append((a, b))
                seen.add(key)

        # Generate candidate cuts based on perimeter vertices and edge midpoints
        candidate_points = self._candidate_points(piece)

        cand: List[
            Tuple[float, Tuple[Point, Point], Tuple[Polygon, Polygon], Tuple[int, int]]
        ] = []
        for i in range(len(candidate_points)):
            for j in range(i + 1, len(candidate_points)):
                pA, pB = candidate_points[i], candidate_points[j]
                ok, _ = cake.cut_is_valid(pA, pB)
                if not ok:
                    continue
                split = cake.cut_piece(piece, pA, pB)
                if len(split) != 2:
                    continue
                P, Q = split
                aP, aQ = P.area, Q.area

                for a, b in splits:
                    tP = a * self.target_area
                    tQ = b * self.target_area
                    # Calculate error based on area and ratio
                    err = (aP - tP) ** 2 + (aQ - tQ) ** 2
                    rP = self._ratio_v_parent(cake, P)
                    rQ = self._ratio_v_parent(cake, Q)
                    err += 0.1 * (
                        (rP - self.avg_ratio) ** 2 + (rQ - self.avg_ratio) ** 2
                    )
                    cand.append((err, (pA, pB), (P, Q), (a, b)))

        if not cand:
            return inf, []

        # Keep the top-K candidates based on beam width
        cand.sort(key=lambda x: x[0])
        cand = cand[: self.beam_width]
        if self.debug:
            print(f"{'  ' * depth}[search] candidates={len(cand)} kept={len(cand)}")

        best_score = best_so_far
        best_seq: List[Tuple[Point, Point]] = []

        # Recursively evaluate each candidate
        for cut_err, (from_p, to_p), (P, Q), (a, b) in cand:
            if cut_err >= best_score:
                continue

            # Prune extreme imbalance
            tP = a * self.target_area
            tQ = b * self.target_area
            if (
                abs(P.area - tP) > 10 * self.area_tol
                and abs(Q.area - tQ) > 10 * self.area_tol
            ):
                continue

            cakeP = self._wrap_subcake(cake, P)
            cakeQ = self._wrap_subcake(cake, Q)

            scoreP, seqP = self._search(cakeP, a, best_score - cut_err, depth + 1)
            if cut_err + scoreP >= best_score:
                continue

            scoreQ, seqQ = self._search(
                cakeQ, b, best_score - cut_err - scoreP, depth + 1
            )
            total = cut_err + scoreP + scoreQ

            if total < best_score:
                best_score = total
                best_seq = [(from_p, to_p)] + seqP + seqQ
                if self.debug:
                    print(
                        f"{'  ' * depth}[search] new best={best_score:.3f} cuts={len(best_seq)}"
                    )

        return best_score, best_seq

    # ---------- greedy fill helpers ----------
    def _best_greedy_cut_for(
        self, cake: Cake, remaining_children: int
    ) -> Optional[Tuple[Point, Point]]:
        """
        Find the best greedy cut for the largest piece to match the target area.

        Args:
            cake (Cake): The current cake state.
            remaining_children (int): Number of remaining children.

        Returns:
            Optional[Tuple[Point, Point]]: The best cut or None if no valid cut is found.
        """
        # Pick the largest piece to split
        target_piece = max(cake.get_pieces(), key=lambda p: p.area)
        m = remaining_children
        a = m // 2
        b = m - a
        tP = a * self.target_area
        tQ = b * self.target_area

        best = None
        best_err = inf

        # Evaluate all candidate cuts
        pts = self._candidate_points(target_piece)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                pA, pB = pts[i], pts[j]
                ok, _ = cake.cut_is_valid(pA, pB)
                if not ok:
                    continue
                split = cake.cut_piece(target_piece, pA, pB)
                if len(split) != 2:
                    continue
                P, Q = split
                err = (P.area - tP) ** 2 + (Q.area - tQ) ** 2
                if err < best_err:
                    best_err = err
                    best = (pA, pB)
        return best

    def _fill_with_any_valid(self, cake: Cake, cuts: List[Tuple[Point, Point]]) -> bool:
        """
        Add any valid cuts to the largest piece until the required number of cuts is reached.

        Args:
            cake (Cake): The current cake state.
            cuts (List[Tuple[Point, Point]]): Current list of cuts.

        Returns:
            bool: True if successful, False otherwise.
        """
        while len(cuts) < self.children - 1:
            target_piece = max(cake.get_pieces(), key=lambda p: p.area)
            pts = self._candidate_points(target_piece)
            found = False
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    a, b = pts[i], pts[j]
                    ok, _ = cake.cut_is_valid(a, b)
                    if not ok:
                        continue
                    cuts.append((a, b))
                    cake.cut(a, b)
                    found = True
                    break
                if found:
                    break
            if not found:
                return False
        return True

    # ---------- scoring / geometry ----------
    def _leaf_score(self, cake: Cake) -> float:
        """
        Calculate the score for a leaf node (single piece).

        Args:
            cake (Cake): The current cake state.

        Returns:
            float: The score for the leaf node.
        """
        piece = cake.get_pieces()[0]
        area = piece.area
        ratio = self._ratio_v_parent(cake, piece)
        area_err = (area - self.target_area) ** 2
        ratio_err = (ratio - self.avg_ratio) ** 2
        return area_err + 0.1 * ratio_err

    def _candidate_points(self, poly: Polygon) -> List[Point]:
        """
        Generate candidate points for cuts (vertices and edge midpoints).

        Args:
            poly (Polygon): The polygon to generate points for.

        Returns:
            List[Point]: List of candidate points.
        """
        verts = list(poly.exterior.coords[:-1])
        pts = [Point(v) for v in verts]
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            pts.append(Point((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return pts

    def _wrap_subcake(self, parent: Cake, poly: Polygon) -> Cake:
        """
        Create a lightweight sub-cake for recursive processing.

        Args:
            parent (Cake): The parent cake.
            poly (Polygon): The polygon representing the sub-cake.

        Returns:
            Cake: The sub-cake object.
        """
        new = object.__new__(Cake)
        new.exterior_shape = poly
        new.interior_shape = parent.interior_shape
        new.exterior_pieces = [poly]
        new.sandbox = True
        return new

    def _ratio_v_parent(self, parent: Cake, poly: Polygon) -> float:
        """
        Calculate the interior ratio of a polygon relative to its parent.

        Args:
            parent (Cake): The parent cake.
            poly (Polygon): The polygon to calculate the ratio for.

        Returns:
            float: The interior ratio.
        """
        if poly.is_empty or poly.area <= 0:
            return 0.0
        inter = poly.intersection(parent.interior_shape)
        return 0.0 if inter.is_empty else inter.area / poly.area
