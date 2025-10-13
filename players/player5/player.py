from shapely import Point, Polygon
from shapely.ops import split as geom_split
from shapely.geometry import LineString as LS
from players.player import Player, PlayerException
from src.cake import Cake
import src.constants as c
import math
from typing import List, Tuple, Optional


class Player5(Player):
    # Sweeps
    NUM_DIRECTIONS = 64  # directions over [0, π)
    OFFSETS_PER_DIRECTION = 240  # offsets per direction
    MAX_VERTEX_ENUM = 240
    REFINE_ITERS = 60
    REFINE_EPS = 1e-9

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        print(f"I am {self}")

        self.same_size_tol = getattr(c, "SAME_SIZE_TOLERANCE", 0.5)
        self.num_tol = getattr(c, "NUM_TOL", 2e-3)
        self.area_tol = max(1e-6, self.same_size_tol / 2.0 - self.num_tol / 2.0)
        self.min_seg_len = 1e-5

    def get_cuts(self) -> List[Tuple[Point, Point]]:
        if not self._validate_constraints():
            raise PlayerException("Cake does not meet constraints")

        temp_cake = self.cake.copy()
        moves: List[Tuple[Point, Point]] = []

        total_area = temp_cake.get_area()
        if self.children <= 0:
            raise PlayerException("Invalid number of children")
        A_star = total_area / self.children
        S = 0.0  # running deviation: sum(produced) - m * A_star

        for i in range(self.children - 1):
            piece = max(temp_cake.get_pieces(), key=lambda p: p.area)
            piece = self._clean(piece)

            r = self.children - i
            desired = A_star - (S / r)

            # Clamp to feasible (strict interior)
            eps = max(1e-6, 1e-4 * piece.area)
            desired = max(eps, min(desired, piece.area - eps))

            # Tighter tolerance on the final cut (guarantees last two pieces land within spread)
            tol = (
                self.area_tol
                if i < self.children - 2
                else max(5e-4, self.area_tol * 0.8)
            )

            best = self._best_segment_for_target(temp_cake, piece, desired, tol)
            if best is None:
                raise PlayerException(f"Could not find valid cut for piece {i + 1}")

            a, b, produced_area = best
            S += produced_area - A_star

            moves.append((a, b))
            temp_cake.cut(a, b)

        areas = [p.area for p in temp_cake.get_pieces()]
        if not areas:
            raise PlayerException("No pieces after cutting?")
        spread = max(areas) - min(areas)
        if spread > (self.same_size_tol + self.num_tol):
            raise PlayerException(
                f"Size tolerance breached: spread={spread:.4f} > {self.same_size_tol} (+{self.num_tol} tol)"
            )

        return moves

    # Segment search

    def _best_segment_for_target(
        self, temp_cake_obj: Cake, polygon: Polygon, target_area: float, area_tol: float
    ) -> Optional[Tuple[Point, Point, float]]:
        poly = self._clean(polygon)
        if poly.is_empty or poly.area <= 0:
            return None

        best_tuple: Optional[Tuple[Point, Point, float]] = None
        best_err = float("inf")

        # Try vertex–vertex chords first on small polygons (often perfect early hits)
        for seg in self._vertex_pair_chords(poly):
            cand = self._eval_segment(temp_cake_obj, poly, seg, target_area)
            if cand is None:
                continue
            a, b, produced_area, err = cand
            if err <= area_tol:
                return (a, b, produced_area)
            if err < best_err:
                best_err = err
                best_tuple = (a, b, produced_area)

        # Directional sweeps with bracketed refinement
        for ux, uy in self._direction_set(poly):
            cand = self._search_direction_with_refine(
                temp_cake_obj, poly, (ux, uy), target_area, area_tol
            )
            if cand is None:
                continue
            a, b, produced_area, err = cand
            if err <= area_tol:
                return (a, b, produced_area)
            if err < best_err:
                best_err = err
                best_tuple = (a, b, produced_area)

        return best_tuple

    def _direction_set(self, poly: Polygon) -> List[Tuple[float, float]]:
        """Evenly spaced directions over [0, π)."""
        dirs: List[Tuple[float, float]] = []
        for k in range(self.NUM_DIRECTIONS):
            th = (math.pi * k) / self.NUM_DIRECTIONS
            dirs.append((math.cos(th), math.sin(th)))
        return dirs

    def _search_direction_with_refine(
        self,
        temp_cake_obj: Cake,
        poly: Polygon,
        direction: Tuple[float, float],
        target_area: float,
        area_tol: float,
    ) -> Optional[Tuple[Point, Point, float, float]]:
        """Sweep offsets; when two neighboring offsets bracket target, refine by bisection."""
        ux, uy = self._unit(direction)
        nx, ny = -uy, ux

        minx, miny, maxx, maxy = poly.bounds
        cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
        diameter = max(1e-6, math.hypot(maxx - minx, maxy - miny))
        L = diameter * 8.0

        verts = list(poly.exterior.coords[:-1])
        if not verts:
            return None
        dots = [(vx - cx) * nx + (vy - cy) * ny for (vx, vy) in verts]
        s_min = min(dots) - diameter * 0.5
        s_max = max(dots) + diameter * 0.5

        best: Optional[Tuple[Point, Point, float, float]] = None
        best_err = float("inf")

        prev_s = None
        prev_err = None

        for j in range(self.OFFSETS_PER_DIRECTION):
            t = (
                j / (self.OFFSETS_PER_DIRECTION - 1)
                if self.OFFSETS_PER_DIRECTION > 1
                else 0.5
            )
            s = s_min * (1.0 - t) + s_max * t

            x0, y0 = cx + s * nx, cy + s * ny
            a_inf = Point(x0 - ux * L, y0 - uy * L)
            b_inf = Point(x0 + ux * L, y0 + uy * L)
            inf_line = LS([a_inf, b_inf])

            # Evaluate all chords on this line; pick the best legal one
            chord_cand = self._best_chord_on_line(
                temp_cake_obj, poly, inf_line, target_area
            )
            if chord_cand is None:
                continue
            a, b, produced_area, err = chord_cand

            # Track best
            if err <= area_tol:
                return (a, b, produced_area, err)
            if err < best_err:
                best_err = err
                best = (a, b, produced_area, err)

            # inside bracket
            signed = produced_area - target_area
            if prev_s is not None and prev_err is not None:
                prev_signed = prev_err  # signed error there
                if signed == 0 or prev_signed == 0 or (signed > 0) != (prev_signed > 0):
                    # Refine between prev_s and s
                    ref = self._refine_between_offsets(
                        temp_cake_obj,
                        poly,
                        (ux, uy),
                        (nx, ny),
                        cx,
                        cy,
                        L,
                        prev_s,
                        s,
                        target_area,
                        area_tol,
                    )
                    if ref is not None:
                        ra, rb, rA, rerr = ref
                        if rerr <= area_tol:
                            return (ra, rb, rA, rerr)
                        if rerr < best_err:
                            best_err = rerr
                            best = (ra, rb, rA, rerr)

            prev_s = s
            prev_err = signed

        return best

    def _refine_between_offsets(
        self,
        temp_cake_obj: Cake,
        poly: Polygon,
        u: Tuple[float, float],
        nvec: Tuple[float, float],
        cx: float,
        cy: float,
        L: float,
        s_lo: float,
        s_hi: float,
        target_area: float,
        area_tol: float,
    ) -> Optional[Tuple[Point, Point, float, float]]:
        """Binary search on s in [s_lo, s_hi] to hit target area using exact chord splits."""
        ux, uy = u
        nx, ny = nvec

        def eval_at(s: float) -> Optional[Tuple[Point, Point, float, float]]:
            x0, y0 = cx + s * nx, cy + s * ny
            a_inf = Point(x0 - ux * L, y0 - uy * L)
            b_inf = Point(x0 + ux * L, y0 + uy * L)
            inf_line = LS([a_inf, b_inf])
            return self._best_chord_on_line(temp_cake_obj, poly, inf_line, target_area)

        lo, hi = s_lo, s_hi
        best: Optional[Tuple[Point, Point, float, float]] = None
        best_err = float("inf")

        for _ in range(self.REFINE_ITERS):
            if abs(hi - lo) <= self.REFINE_EPS:
                break
            mid = 0.5 * (lo + hi)
            cand = eval_at(mid)
            if cand is None:
                # If invalid geometry
                hi = mid
                continue
            a, b, produced_area, err = cand
            if err <= area_tol:
                return (a, b, produced_area, err)
            if err < best_err:
                best_err = err
                best = (a, b, produced_area, err)
            # Move side based on sign of (produced - target)
            if (produced_area - target_area) > 0:
                hi = mid
            else:
                lo = mid

        return best

    def _best_chord_on_line(
        self, temp_cake_obj: Cake, poly: Polygon, inf_line: LS, target_area: float
    ) -> Optional[Tuple[Point, Point, float, float]]:
        """Intersect poly with inf_line; among the chords, return best legal split."""
        try:
            inter = poly.intersection(inf_line)
        except Exception:
            return None

        chords: List[LS] = []
        if isinstance(inter, LS):
            chords = [inter]
        elif hasattr(inter, "geoms"):
            chords = [g for g in inter.geoms if isinstance(g, LS)]

        best: Optional[Tuple[Point, Point, float, float]] = None
        best_err = float("inf")

        for chord in chords:
            if chord.length < self.min_seg_len:
                continue
            cand = self._eval_segment(temp_cake_obj, poly, chord, target_area)
            if cand is None:
                continue
            a, b, produced_area, err = cand
            if err < best_err:
                best_err = err
                best = (a, b, produced_area, err)
        return best

    def _eval_segment(
        self, temp_cake_obj: Cake, poly: Polygon, seg: LS, target_area: float
    ) -> Optional[Tuple[Point, Point, float, float]]:
        """Validate seg on current cake, split poly with it, and return errors."""
        if not isinstance(seg, LS) or seg.length < self.min_seg_len:
            return None
        (xa, ya), (xb, yb) = list(seg.coords)[0], list(seg.coords)[-1]
        a, b = Point(xa, ya), Point(xb, yb)

        # Validate on current cake
        if not self._valid_on_current_cake(temp_cake_obj, a, b):
            return None

        try:
            parts = geom_split(poly, seg)
        except Exception:
            return None
        if not hasattr(parts, "geoms") or len(parts.geoms) != 2:
            return None

        A0 = parts.geoms[0].area
        A1 = parts.geoms[1].area
        produced_area = A0 if abs(A0 - target_area) <= abs(A1 - target_area) else A1
        err = abs(produced_area - target_area)
        return (a, b, produced_area, err)

    def _vertex_pair_chords(self, polygon: Polygon) -> List[LS]:
        out: List[LS] = []
        try:
            coords = list(polygon.exterior.coords[:-1])
        except Exception:
            return out

        n = len(coords)
        if n < 2 or n > self.MAX_VERTEX_ENUM:
            return out

        poly = self._clean(polygon)
        for i in range(n):
            ai = coords[i]
            for j in range(i + 1, n):
                bj = coords[j]
                seg = LS([ai, bj])
                if seg.length < self.min_seg_len:
                    continue
                try:
                    if poly.covers(seg):
                        out.append(seg)
                except Exception:
                    # fallback: keep if seg is not leaving poly
                    try:
                        if seg.difference(poly).is_empty:
                            out.append(seg)
                    except Exception:
                        continue
        return out

    def _valid_on_current_cake(self, cake_obj: Cake, a: Point, b: Point) -> bool:
        if a.distance(b) < self.min_seg_len:
            return False
        try:
            ok, _ = cake_obj.cut_is_valid(a, b)
            return ok
        except Exception:
            return False

    def _validate_constraints(self) -> bool:
        area_per_child = self.cake.get_area() / self.children
        if not (
            c.MIN_PIECE_AREA_PER_CHILD <= area_per_child <= c.MAX_PIECE_AREA_PER_CHILD
        ):
            return False
        try:
            interior_ratio = (
                self.cake.interior_shape.area / self.cake.exterior_shape.area
            )
        except Exception:
            return False
        if interior_ratio < getattr(c, "MIN_CAKE_INTERIOR_RATIO", 0.5):
            return False
        # Min interior angle ≥ 15°
        mind = self._min_interior_angle_deg(self.cake.exterior_shape)
        return (mind is not None) and (mind >= 15.0)

    def _min_interior_angle_deg(self, poly: Polygon) -> Optional[float]:
        try:
            coords = list(poly.exterior.coords[:-1])
            n = len(coords)
            if n < 3:
                return None

            def ang(a, b, c):
                bax, bay = a[0] - b[0], a[1] - b[1]
                bcx, bcy = c[0] - b[0], c[1] - b[1]
                na, nc = math.hypot(bax, bay), math.hypot(bcx, bcy)
                if na == 0 or nc == 0:
                    return math.pi
                cosv = (bax * bcx + bay * bcy) / (na * nc)
                cosv = max(-1.0, min(1.0, cosv))
                return math.acos(cosv)

            mins = []
            for i in range(n):
                mins.append(ang(coords[(i - 1) % n], coords[i], coords[(i + 1) % n]))
            return min(mins) * (180.0 / math.pi)
        except Exception:
            return None

    def _unit(self, v: Tuple[float, float]) -> Tuple[float, float]:
        x, y = v
        n = math.hypot(x, y)
        if n == 0:
            return (1.0, 0.0)
        return (x / n, y / n)

    def _clean(self, poly: Polygon) -> Polygon:
        try:
            return poly if poly.is_valid else poly.buffer(0)
        except Exception:
            return poly
