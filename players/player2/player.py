# players/player2.py
from __future__ import annotations
from typing import List, Tuple, Optional
from math import inf
from shapely import Point
from shapely.geometry import LineString, Polygon

from players.player import Player, PlayerException
import src.constants as c
from src.cake import Cake, extend_line
import time


class Player2(Player):
    """
    divide-&-Conquer (beam) planner + robust sequential

    plan phase (on subcakes):
      - recursively split target piece counts (prefer near-halves)
      - evaluate chords built from rich boundary candidates
      - keep top-K (beam) by squared error vs. target areas
      - return an ordered list of cuts (root cut, then cuts for each subpiece)

    phase (on a working copy of the real cake):
      - apply cuts in that order
      - for each planned pair, SNAP both endpoints to the boundary of the
        actual piece it should cut (try all pieces, choose the first that yields a
        valid split under simulator rules)
      - if a planned cut can't be realized, use a greedy best cut on the
        largest current piece; if that fails, use any-valid fallback.

    Focus: equal areas. (Crust ratio only lightly penalized in scoring;
    you can dial that in later.)
    """

    # ------------------ init ------------------

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

        base_poly = self.cake.get_pieces()[0]
        self.total_area = base_poly.area
        self.target_area = self.total_area / max(1, self.children)
        self.avg_ratio = self.cake.interior_shape.area / max(self.total_area, 1e-9)

        self.beam_width = 5  # widen for better search, narrow for speed
        self.sample_count = 64  # boundary samples per piece (in addition to vertices)
        self.area_tol = getattr(c, "PIECE_SPAN_TOL", 0.5)  # cm^2
        self.ratio_w = 0.10  # small weight on ratio to break ties (can set 0)

        self.max_area_deviation = 0.25
        print("Using Player2 v4 with two-stage scoring (area-first, ratio-second)")
        # print(f"Target area per piece: {self.target_area:.2f} cm²")
        # print(f"Max area deviation tolerance: {self.max_area_deviation} cm²")

    # ------------------ public ------------------

    def get_cuts(self) -> List[Tuple[Point, Point]]:
        start_time = time.time()
        if self.children <= 1:
            return []

        # on a sandbox copy (subcakes)
        plan_cake = self.cake.copy()
        root_piece = plan_cake.get_pieces()[0]
        _, planned = self._search(
            self._wrap_subcake(plan_cake, root_piece),
            pieces_needed=self.children,
            best_so_far=inf,
            depth=0,
        )

        # planned list sequentially on a fresh working cake,
        # snapping each cut to the current geometry
        realized = self._realize_sequential(self.cake.copy(), planned)

        # Debug: check if planning was complete
        print(
            f"Planning phase found {len(planned)} cuts, need {self.children - 1} total"
        )
        if len(planned) < self.children - 1:
            print(
                f"PLANNING INCOMPLETE: Missing {self.children - 1 - len(planned)} cuts, falling back to greedy"
            )

        # if we didn't get enough, greedily add equal-area cuts
        while len(realized) < self.children - 1:
            add = self._best_greedy_cut_for(self.cake.copy(), realized)
            if add is None:
                break
            realized.append(add)

        # if still short, fill with any valid cuts
        if len(realized) != self.children - 1:
            if not self._fill_with_any_valid(self.cake.copy(), realized):
                raise PlayerException(
                    f"Player2: could not assemble {self.children - 1} cuts (got {len(realized)})."
                )
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        return realized

    # ------------------ search (plan) ------------------

    def _search(
        self,
        cake: Cake,  # sandbox subcake with a single piece
        pieces_needed: int,
        best_so_far: float,
        depth: int,
    ) -> tuple[float, List[Tuple[Point, Point]]]:
        piece = cake.get_pieces()[0]
        if pieces_needed == 1:
            return self._leaf_score(cake), []

        m = pieces_needed
        preferred = [(m // 2, m - (m // 2))]
        neighbors: list[tuple[int, int]] = []
        if m > 2:
            a0 = preferred[0][0]
            neighbors.append((max(1, a0 - 1), m - max(1, a0 - 1)))
            neighbors.append((min(m - 1, a0 + 1), m - min(m - 1, a0 + 1)))

        splits: list[tuple[int, int]] = []
        seen = set()
        for a, b in preferred + neighbors:
            if a <= 0 or b <= 0:
                continue
            key = (min(a, b), max(a, b))
            if key not in seen:
                splits.append((a, b))
                seen.add(key)

        # rich candidate endpoints on THIS piece boundary
        cand_pts = self._candidate_points(piece)

        cand: list[
            tuple[float, Tuple[Point, Point], Tuple[Polygon, Polygon], Tuple[int, int]]
        ] = []

        for i in range(len(cand_pts)):
            for j in range(i + 1, len(cand_pts)):
                pA, pB = cand_pts[i], cand_pts[j]
                ok, _ = cake.cut_is_valid(pA, pB)
                if not ok:
                    continue
                parts = cake.cut_piece(piece, pA, pB)
                if len(parts) != 2:
                    continue
                P, Q = parts

                # score each desired split (a,b)
                for a, b in splits:
                    tP = a * self.target_area
                    tQ = b * self.target_area

                    area_deviation_P = abs(P.area - tP)
                    area_deviation_Q = abs(Q.area - tQ)

                    if (
                        area_deviation_P > self.max_area_deviation
                        or area_deviation_Q > self.max_area_deviation
                    ):
                        err = (area_deviation_P + area_deviation_Q) * 1000
                    else:
                        ratio_P = self._ratio_v_parent(cake, P)
                        ratio_Q = self._ratio_v_parent(cake, Q)
                        ratio_deviation_P = ratio_P - self.avg_ratio
                        ratio_deviation_Q = ratio_Q - self.avg_ratio
                        err = ratio_deviation_P**2 + ratio_deviation_Q**2

                    cand.append((err, (pA, pB), (P, Q), (a, b)))

        if not cand:
            print(
                f"  Search failed at depth {depth}: no valid cuts found for {pieces_needed} pieces"
            )
            return inf, []

        # Debug: check how many cuts meet area requirements
        area_acceptable = sum(
            1 for err, _, _, _ in cand if err < 1000
        )  # < 1000 means no heavy penalty
        print(
            f"  Found {len(cand)} total cuts, {area_acceptable} meet area requirements"
        )

        cand.sort(key=lambda x: x[0])
        cand = cand[: self.beam_width]

        best_score = best_so_far
        best_seq: List[Tuple[Point, Point]] = []

        for cut_err, (from_p, to_p), (P, Q), (a, b) in cand:
            if cut_err >= best_score:
                continue

            # prune very poor fits - use same tolerance as two-stage scoring
            tP = a * self.target_area
            tQ = b * self.target_area
            if (
                abs(P.area - tP) > self.max_area_deviation
                or abs(Q.area - tQ) > self.max_area_deviation
            ):
                continue

            cakeP = self._wrap_subcake(cake, P)
            cakeQ = self._wrap_subcake(cake, Q)

            sP, seqP = self._search(cakeP, a, best_score - cut_err, depth + 1)
            if cut_err + sP >= best_score:
                continue

            sQ, seqQ = self._search(cakeQ, b, best_score - cut_err - sP, depth + 1)
            total = cut_err + sP + sQ

            if total < best_score:
                best_score = total
                best_seq = [(from_p, to_p)] + seqP + seqQ

        # if best_seq:
        #     print(f"  Search at depth {depth}: found solution with {len(best_seq)} cuts for {pieces_needed} pieces")
        # else:
        #     print(f"  Search at depth {depth}: no solution found for {pieces_needed} pieces")

        return best_score, best_seq

    # ------------------ realization (snap & apply) ------------------

    def _realize_sequential(
        self, sim: Cake, planned: List[Tuple[Point, Point]]
    ) -> List[Tuple[Point, Point]]:
        realized: List[Tuple[Point, Point]] = []

        for cut_idx, (p_raw, q_raw) in enumerate(planned):
            snapped = self._snap_to_some_piece(sim, p_raw, q_raw)
            if snapped is None:
                # planned cut can't be realized on current geometry; try greedy instead
                print(
                    f"Cut {cut_idx + 1}: Planned cut failed to snap, trying greedy..."
                )
                g = self._best_greedy_cut_for(sim, realized)
                if g is None:
                    # try any valid
                    g = self._any_valid_on(sim)
                    print(
                        f"Cut {cut_idx + 1}: Using ANY-VALID cut (planning failed badly)"
                    )
                else:
                    print(f"Cut {cut_idx + 1}: Using GREEDY cut (planning failed)")
                if g is None:
                    # give up on this slot; continue (fill later)
                    print(
                        f"Cut {cut_idx + 1}: Giving up on this slot; continue (fill later)"
                    )
                    continue
                p_use, q_use = g
            else:
                p_use, q_use = snapped
                print(f"Cut {cut_idx + 1}: Planned cut succeeded")

            realized.append((p_use, q_use))
            try:
                sim.cut(p_use, q_use)
            except Exception:
                # if simulator still rejects, skip but keep count
                pass

        return realized

    def _snap_to_some_piece(
        self, sim: Cake, p: Point, q: Point
    ) -> Optional[Tuple[Point, Point]]:
        """
        Try every current piece: project p and q to that piece boundary.
        If extend_line(boundary-snapped segment) validly splits the piece into two
        >= MIN_PIECE_AREA polygons, return snapped endpoints.
        """
        for piece in sim.get_pieces():
            bound = piece.boundary
            a = bound.interpolate(bound.project(p))
            b = bound.interpolate(bound.project(q))

            # Build the extended line and test piece-level split predicate
            line = extend_line(LineString([a, b]))
            good, _ = sim.does_line_cut_piece_well(line, piece)
            if not good:
                continue

            # Also make sure global validity passes (single-piece, within cake)
            ok, _ = sim.cut_is_valid(a, b)
            if not ok:
                continue

            return a, b
        return None

    # ------------------ greedy & fallbacks ------------------

    def _best_greedy_cut_for(
        self, original: Cake, already_realized: List[Tuple[Point, Point]]
    ) -> Optional[Tuple[Point, Point]]:
        """
        Given a fresh copy of the original cake and the cuts already realized,
        simulate those cuts to reproduce current geometry, then find the best
        approximate equal-area cut on the largest current piece.
        """
        sim = original.copy()
        # replay already_realized to mirror current state
        for a, b in already_realized:
            try:
                sim.cut(a, b)
            except Exception:
                # Try one snap pass if it fails
                snapped = self._snap_to_some_piece(sim, a, b)
                if snapped is None:
                    return None
                sim.cut(snapped[0], snapped[1])

        target_piece = max(sim.get_pieces(), key=lambda p: p.area)

        piece_children = max(1, round(target_piece.area / self.target_area))
        a = max(1, piece_children // 2)
        b = max(1, piece_children - a)

        # Calculate target areas for the two pieces we're creating
        tP = a * self.target_area
        tQ = b * self.target_area

        print(
            f"  Greedy: piece {target_piece.area:.1f} cm² should serve {piece_children} children"
        )
        print(
            f"  Greedy: splitting into {a} and {b} children, targets={tP:.1f} and {tQ:.1f} cm²"
        )

        best = None
        best_err = inf

        pts = self._candidate_points(target_piece)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                pA, pB = pts[i], pts[j]
                ok, _ = sim.cut_is_valid(pA, pB)
                if not ok:
                    continue
                parts = sim.cut_piece(target_piece, pA, pB)
                if len(parts) != 2:
                    continue
                P, Q = parts

                area_deviation_P = abs(P.area - tP)
                area_deviation_Q = abs(Q.area - tQ)

                if (
                    area_deviation_P > self.max_area_deviation
                    or area_deviation_Q > self.max_area_deviation
                ):
                    err = (area_deviation_P + area_deviation_Q) * 1000
                else:
                    ratio_P = self._ratio_v_parent(sim, P)
                    ratio_Q = self._ratio_v_parent(sim, Q)
                    ratio_deviation_P = ratio_P - self.avg_ratio
                    ratio_deviation_Q = ratio_Q - self.avg_ratio
                    err = ratio_deviation_P**2 + ratio_deviation_Q**2

                if err < best_err:
                    best_err = err
                    best = (pA, pB)
                    # Debug: show area deviations for best greedy cut
                    if (
                        area_deviation_P > self.max_area_deviation
                        or area_deviation_Q > self.max_area_deviation
                    ):
                        print(
                            f"  Greedy cut with area violations: P={P.area:.1f} (target={tP:.1f}, dev={area_deviation_P:.1f}), Q={Q.area:.1f} (target={tQ:.1f}, dev={area_deviation_Q:.1f})"
                        )

        return best

    def _any_valid_on(self, sim: Cake) -> Optional[Tuple[Point, Point]]:
        """Pick any valid cut on the largest current piece, preferring area-balanced cuts."""
        piece = max(sim.get_pieces(), key=lambda p: p.area)
        pts = self._candidate_points(piece)

        # Try to find a cut that's reasonably balanced
        best_cut = None
        best_area_balance = float("inf")

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                ok, _ = sim.cut_is_valid(a, b)
                if ok:
                    # Test the cut to see area balance
                    try:
                        parts = sim.cut_piece(piece, a, b)
                        if len(parts) == 2:
                            P, Q = parts
                            area_balance = abs(P.area - Q.area)
                            if area_balance < best_area_balance:
                                best_area_balance = area_balance
                                best_cut = (a, b)
                    except Exception:
                        continue

        return best_cut

    def _fill_with_any_valid(
        self, original: Cake, cuts: List[Tuple[Point, Point]]
    ) -> bool:
        """Keep adding any valid cuts until we reach n-1."""
        sim = original.copy()
        for a, b in cuts:
            try:
                sim.cut(a, b)
            except Exception:
                snapped = self._snap_to_some_piece(sim, a, b)
                if snapped is None:
                    continue
                sim.cut(snapped[0], snapped[1])

        while len(cuts) < self.children - 1:
            g = self._any_valid_on(sim)
            if g is None:
                return False
            cuts.append(g)
            try:
                sim.cut(g[0], g[1])
            except Exception:
                # if even this fails, try snap once
                snapped = self._snap_to_some_piece(sim, g[0], g[1])
                if snapped is None:
                    return False
                sim.cut(snapped[0], snapped[1])
        return True

    # ------------------ points / scoring / subcakes ------------------

    def _candidate_points(self, poly: Polygon) -> List[Point]:
        """
        Vertices + uniform samples along boundary length.
        This avoids 'corner fan' and provides many chords that truly cross the piece.
        """
        pts: List[Point] = [Point(xy) for xy in list(poly.exterior.coords[:-1])]
        L = poly.boundary.length
        if L <= c.TOL:
            return pts
        step = L / self.sample_count
        # start at small offset to avoid degeneracy with actual vertices
        offset = 0.5 * step
        for k in range(self.sample_count):
            s = (offset + k * step) % L
            pts.append(poly.boundary.interpolate(s))
        return pts

    def _leaf_score(self, cake: Cake) -> float:
        piece = cake.get_pieces()[0]

        # Stage 1: Check area tolerance
        area_deviation = abs(piece.area - self.target_area)
        if area_deviation > self.max_area_deviation:
            return area_deviation * 1000  # Heavy penalty for area violations

        # Stage 2: If area is acceptable, score by interior ratio deviation
        interior_ratio = self._ratio_v_parent(cake, piece)
        ratio_deviation = interior_ratio - self.avg_ratio
        return ratio_deviation**2

    def _wrap_subcake(self, parent: Cake, poly: Polygon) -> Cake:
        new = object.__new__(Cake)
        new.exterior_shape = poly
        new.interior_shape = parent.interior_shape
        new.exterior_pieces = [poly]
        new.sandbox = True
        return new

    def _ratio_v_parent(self, parent: Cake, poly: Polygon) -> float:
        if poly.is_empty or poly.area <= 0:
            return 0.0
        inter = poly.intersection(parent.interior_shape)
        return 0.0 if inter.is_empty else inter.area / poly.area
