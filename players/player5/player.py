from shapely import Point, Polygon
from shapely.geometry import LineString as LS
from players.player import Player, PlayerException
from src.cake import Cake
import src.constants as c
import math
import time
from typing import List, Tuple, Optional


class Player5(Player):
    NUM_DIRECTIONS = 48
    OFFSETS_COARSE = 48
    OFFSETS_LOCAL = 20
    LOCAL_WINDOW_FRAC = 0.04
    MAX_VERTEX_ENUM = 240
    REFINE_ITERS = 50
    REFINE_EPS = 1e-9

    HARD_NUM_DIRECTIONS = 96
    HARD_OFFSETS_COARSE = 96
    HARD_OFFSETS_LOCAL = 48
    HARD_REFINE_ITERS = 140

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        print(f"I am {self}")

        self.same_size_tol = getattr(c, "SAME_SIZE_TOLERANCE", 0.5)
        self.num_tol = getattr(c, "NUM_TOL", 2e-3)

        self.band_star = max(1e-6, 0.5 * max(0.0, self.same_size_tol - self.num_tol))
        self.per_piece_band = max(1e-6, 0.9 * self.band_star)
        self.area_tol = max(1e-6, self.per_piece_band * 0.8)

        self.min_seg_len = 1e-5
        self.time_budget_sec = 55.0

        try:
            outer = self.cake.exterior_shape
            inner = outer.buffer(-1.0)
            self.inner_cake = self._clean(inner) if not inner.is_empty else Polygon()
            self.total_area = outer.area if outer is not None else 0.0
            crust_area_total = max(0.0, self.total_area - self.inner_cake.area)
            self.P_star = (
                (crust_area_total / self.total_area) if self.total_area > 0 else 0.0
            )
        except Exception:
            self.inner_cake = Polygon()
            self.total_area = self.cake.get_area()
            self.P_star = 0.0

    def _time_exceeded(self, start: float) -> bool:
        return (time.perf_counter() - start) >= self.time_budget_sec

    def get_cuts(self) -> List[Tuple[Point, Point]]:
        if not self._validate_constraints():
            raise PlayerException("Cake does not meet constraints")

        start_time = time.perf_counter()
        temp_cake = self.cake.copy()
        moves: List[Tuple[Point, Point]] = []

        total_area = temp_cake.get_area()
        if self.children <= 0:
            raise PlayerException("Invalid number of children")
        A_star = total_area / self.children

        S = 0.0
        S_prop = 0.0

        for i in range(self.children - 1):
            if self._time_exceeded(start_time):
                return moves

            piece = max(temp_cake.get_pieces(), key=lambda p: p.area)
            piece = self._clean(piece)

            r = self.children - i
            desired = A_star - (S / r)
            desired = max(
                A_star - self.band_star, min(A_star + self.band_star, desired)
            )
            desired_prop = self.P_star - (S_prop / r)

            remaining = r
            base_eps = max(1e-6, 1e-4 * piece.area)
            tight_eps = max(2e-6, 8e-4 * piece.area)
            eps = tight_eps if remaining <= 3 else base_eps
            desired = max(eps, min(desired, piece.area - eps))

            if remaining <= 3:
                tol = max(1e-4, self.area_tol * 0.6)
                band_local = max(1e-6, 0.8 * self.band_star)
            elif remaining == 4:
                tol = max(2e-4, self.area_tol * 0.8)
                band_local = max(1e-6, 0.9 * self.band_star)
            else:
                tol = self.area_tol
                band_local = self.per_piece_band

            cand = self._best_segment_for_target_global(
                temp_cake,
                piece,
                desired,
                tol,
                desired_prop,
                A_star,
                band_local,
                start_time,
                remaining,
            )
            if cand is None:
                cand = self._best_segment_for_target_global(
                    temp_cake,
                    piece,
                    desired,
                    tol,
                    desired_prop,
                    A_star,
                    band_local,
                    start_time,
                    remaining,
                    hard_mode=True,
                )

            if cand is None:
                cand = self._best_nearby_to_astar(temp_cake, piece, A_star, start_time)

            if cand is None:
                return moves

            a, b, produced_A, produced_prop = cand

            if remaining == 2:
                pol = self._polish_last_cut(temp_cake, a, b, piece, A_star, start_time)
                if pol is not None:
                    a, b, produced_A = pol

            moves.append((a, b))
            temp_cake.cut(a, b)

            S += produced_A - A_star
            if produced_prop is not None:
                S_prop += produced_prop - self.P_star

        return moves

    def _best_segment_for_target_global(
        self,
        temp_cake_obj: Cake,
        polygon: Polygon,
        target_area: float,
        area_tol: float,
        desired_prop: float,
        A_star: float,
        band: float,
        start_time: float,
        remaining: int,
        hard_mode: bool = False,
    ) -> Optional[Tuple[Point, Point, float, Optional[float]]]:
        poly = self._clean(polygon)
        if poly.is_empty or poly.area <= 0:
            return None

        if hard_mode:
            ndirs = self.HARD_NUM_DIRECTIONS
            coarse = self.HARD_OFFSETS_COARSE
            local = self.HARD_OFFSETS_LOCAL
            refine_iters = self.HARD_REFINE_ITERS
        else:
            ndirs = self._direction_count(poly, remaining)
            coarse = self._coarse_count(remaining)
            local = self._local_count(remaining)
            refine_iters = self.REFINE_ITERS

        best_in_band: Optional[Tuple[Point, Point, float, float]] = None
        best_prop_delta = float("inf")

        for seg in self._vertex_pair_chords(poly):
            if self._time_exceeded(start_time):
                break
            cand = self._evaluate_candidate_global(
                temp_cake_obj, poly, seg, target_area, A_star, band
            )
            if cand is None:
                continue
            a, b, A, err_desired, prop = cand

            if (abs(A - target_area) <= band) and (abs(A - A_star) <= self.band_star):
                pv = 0.0 if prop is None else prop
                delta = abs(pv - desired_prop)
                if (best_in_band is None) or (delta < best_prop_delta):
                    best_in_band = (a, b, A, pv)

        if best_in_band is not None:
            return (best_in_band[0], best_in_band[1], best_in_band[2], best_in_band[3])

        cand2 = self._search_direction_adaptive_global(
            temp_cake_obj,
            poly,
            target_area,
            area_tol,
            desired_prop,
            A_star,
            band,
            start_time,
            remaining,
            refine_iters=refine_iters,
            local_count=local,
            coarse_count=coarse,
            ndirs=ndirs,
        )
        if cand2 is not None:
            return cand2

        return None

    def _search_direction_adaptive_global(
        self,
        cake: Cake,
        poly: Polygon,
        target_area: float,
        area_tol: float,
        desired_prop: float,
        A_star: float,
        band: float,
        start_time: float,
        remaining: int,
        refine_iters: int,
        local_count: int,
        coarse_count: int,
        ndirs: int,
    ) -> Optional[Tuple[Point, Point, float, Optional[float]]]:
        dirs: List[Tuple[float, float]] = []
        for k in range(ndirs):
            th = (math.pi * k) / ndirs
            dirs.append((math.cos(th), math.sin(th)))

        best_band_tuple = None
        best_prop_delta = float("inf")

        for ux, uy in dirs:
            if self._time_exceeded(start_time):
                break
            ux, uy = self._unit((ux, uy))
            nx, ny = -uy, ux

            minx, miny, maxx, maxy = poly.bounds
            cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
            diameter = max(1e-6, math.hypot(maxx - minx, maxy - miny))
            L = diameter * 8.0

            verts = list(poly.exterior.coords[:-1])
            if not verts:
                continue
            dots = [(vx - cx) * nx + (vy - cy) * ny for (vx, vy) in verts]
            s_min = min(dots) - diameter * 0.5
            s_max = max(dots) + diameter * 0.5
            span = s_max - s_min
            if span <= 0:
                continue

            prev_s = None
            prev_signed = None
            best_s = None

            for j in range(coarse_count):
                if self._time_exceeded(start_time):
                    break
                t = j / (coarse_count - 1) if coarse_count > 1 else 0.5
                s = s_min * (1.0 - t) + s_max * t
                x0, y0 = cx + s * nx, cy + s * ny
                a_inf = Point(x0 - ux * L, y0 - uy * L)
                b_inf = Point(x0 + ux * L, y0 + uy * L)
                cand = self._best_chord_on_line_global(
                    cake, poly, LS([a_inf, b_inf]), target_area, A_star, band
                )
                if cand is None:
                    continue
                a, b, A, err_desired, prop = cand

                if (abs(A - target_area) <= band) and (
                    abs(A - A_star) <= self.band_star
                ):
                    pv = 0.0 if prop is None else prop
                    delta = abs(pv - desired_prop)
                    if (best_band_tuple is None) or (delta < best_prop_delta):
                        best_band_tuple = (a, b, A, pv)
                        best_prop_delta = delta
                    best_s = s

                signed = A - target_area
                if prev_s is not None and prev_signed is not None:
                    if (
                        signed == 0
                        or prev_signed == 0
                        or (signed > 0) != (prev_signed > 0)
                    ):
                        ref = self._refine_between_offsets_global(
                            cake,
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
                            A_star,
                            band,
                            start_time,
                            iters=refine_iters,
                        )
                        if (
                            ref is not None
                            and abs(ref[2] - target_area) <= band
                            and abs(ref[2] - A_star) <= self.band_star
                        ):
                            return ref
                prev_s = s
                prev_signed = signed

            if best_band_tuple is None and best_s is not None:
                half_window = max(span * self.LOCAL_WINDOW_FRAC, span * 1e-3)
                s_lo = best_s - half_window
                s_hi = best_s + half_window
                for j in range(local_count):
                    if self._time_exceeded(start_time):
                        break
                    t = j / (local_count - 1) if local_count > 1 else 0.5
                    s = s_lo * (1.0 - t) + s_hi * t
                    x0, y0 = cx + s * nx, cy + s * ny
                    a_inf = Point(x0 - ux * L, y0 - uy * L)
                    b_inf = Point(x0 + ux * L, y0 + uy * L)
                    cand = self._best_chord_on_line_global(
                        cake, poly, LS([a_inf, b_inf]), target_area, A_star, band
                    )
                    if cand is None:
                        continue
                    a, b, A, err_desired, prop = cand
                    if (abs(A - target_area) <= band) and (
                        abs(A - A_star) <= self.band_star
                    ):
                        pv = 0.0 if prop is None else prop
                        delta = abs(pv - desired_prop)
                        if (best_band_tuple is None) or (delta < best_prop_delta):
                            best_band_tuple = (a, b, A, pv)
                            best_prop_delta = delta

        return best_band_tuple

    def _refine_between_offsets_global(
        self,
        cake: Cake,
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
        A_star: float,
        band: float,
        start_time: float,
        iters: int,
    ) -> Optional[Tuple[Point, Point, float, Optional[float]]]:
        ux, uy = u
        nx, ny = nvec

        def eval_at(s: float):
            x0, y0 = cx + s * nx, cy + s * ny
            a_inf = Point(x0 - ux * L, y0 - uy * L)
            b_inf = Point(x0 + ux * L, y0 + uy * L)
            return self._best_chord_on_line_global(
                cake, poly, LS([a_inf, b_inf]), target_area, A_star, band
            )

        lo, hi = s_lo, s_hi
        best = None
        best_err = float("inf")

        for _ in range(max(20, iters)):
            if self._time_exceeded(start_time):
                return best
            if abs(hi - lo) <= self.REFINE_EPS:
                break
            mid = 0.5 * (lo + hi)
            cand = eval_at(mid)
            if cand is None:
                hi = mid
                continue
            a, b, A, err_desired, prop = cand
            if (abs(A - target_area) <= band) and (abs(A - A_star) <= self.band_star):
                pv = 0.0 if prop is None else prop
                return (a, b, A, pv)
            if err_desired < best_err:
                best_err = err_desired
                best = (a, b, A, None)
            if (A - target_area) > 0:
                hi = mid
            else:
                lo = mid

        return best

    def _best_chord_on_line_global(
        self,
        cake_obj: Cake,
        poly: Polygon,
        inf_line: LS,
        target_area: float,
        A_star: float,
        band: float,
    ) -> Optional[Tuple[Point, Point, float, float, Optional[float]]]:
        try:
            inter = poly.intersection(inf_line)
        except Exception:
            return None

        chords: List[LS] = []
        if isinstance(inter, LS):
            chords = [inter]
        elif hasattr(inter, "geoms"):
            chords = [g for g in inter.geoms if isinstance(g, LS)]

        best = None
        best_err = float("inf")
        for chord in chords:
            cand = self._evaluate_candidate_global(
                cake_obj, poly, chord, target_area, A_star, band
            )
            if cand is None:
                continue
            a, b, A, err_desired, prop = cand
            if err_desired < best_err:
                best_err = err_desired
                best = (a, b, A, err_desired, prop)
                if (abs(A - target_area) <= band) and (
                    abs(A - A_star) <= self.band_star
                ):
                    break
        return best

    def _evaluate_candidate_global(
        self,
        cake_obj: Cake,
        poly: Polygon,
        seg: LS,
        target_area: float,
        A_star: float,
        band: float,
    ) -> Optional[Tuple[Point, Point, float, float, Optional[float]]]:
        if not isinstance(seg, LS) or seg.length < self.min_seg_len:
            return None
        (xa, ya), (xb, yb) = list(seg.coords)[0], list(seg.coords)[-1]
        a, b = Point(xa, ya), Point(xb, yb)

        if not self._valid_on_current_cake(cake_obj, a, b):
            snapped = self._snap_to_cake_perimeter(cake_obj, seg)
            if snapped is None:
                return None
            a, b = snapped
            if not self._valid_on_current_cake(cake_obj, a, b):
                return None
            seg = LS([a, b])

        if self._touches_other_pieces(cake_obj, poly, seg):
            return None

        try:
            sim = cake_obj.copy()
            sim.cut(a, b)
            cut_line = LS([a, b])
            candidates = []
            for p in sim.get_pieces():
                try:
                    inter = p.intersection(cut_line)
                    if not inter.is_empty and getattr(inter, "length", 0.0) > 1e-9:
                        candidates.append(p)
                except Exception:
                    continue
            if not candidates:
                return None
            produced = min(candidates, key=lambda p: abs(p.area - target_area))
            A = produced.area
            err_desired = abs(A - target_area)
            prop = (
                self._prop_of(self._clean(produced))
                if self.inner_cake is not None
                else None
            )
            if prop is not None and not math.isfinite(prop):
                prop = None
            return (a, b, A, err_desired, prop)
        except Exception:
            return None

    def _best_nearby_to_astar(
        self, cake: Cake, poly: Polygon, A_star: float, start_time: float
    ) -> Optional[Tuple[Point, Point, float, Optional[float]]]:
        ndirs = 32
        coarse = 48
        local = 24

        dirs: List[Tuple[float, float]] = []
        for k in range(ndirs):
            th = (math.pi * k) / ndirs
            dirs.append((math.cos(th), math.sin(th)))

        best = None
        best_err = float("inf")

        minx, miny, maxx, maxy = poly.bounds
        cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
        diameter = max(1e-6, math.hypot(maxx - minx, maxy - miny))
        L = diameter * 8.0

        for ux, uy in dirs:
            if self._time_exceeded(start_time):
                break
            ux, uy = self._unit((ux, uy))
            nx, ny = -uy, ux

            verts = list(poly.exterior.coords[:-1])
            if not verts:
                continue
            dots = [(vx - cx) * nx + (vy - cy) * ny for (vx, vy) in verts]
            s_min = min(dots) - diameter * 0.5
            s_max = max(dots) + diameter * 0.5

            for j in range(coarse):
                if self._time_exceeded(start_time):
                    break
                t = j / (coarse - 1) if coarse > 1 else 0.5
                s = s_min * (1.0 - t) + s_max * t
                x0, y0 = cx + s * nx, cy + s * ny
                a_inf = Point(x0 - ux * L, y0 - uy * L)
                b_inf = Point(x0 + ux * L, y0 + uy * L)
                cand = self._best_chord_on_line_global(
                    cake, poly, LS([a_inf, b_inf]), A_star, A_star, 1e12
                )
                if cand is None:
                    continue
                a, b, A, _, prop = cand
                err = abs(A - A_star)
                if err < best_err:
                    best = (a, b, A, prop if prop is not None else 0.0)
                    best_err = err

            if best is not None:
                x0, y0 = cx, cy
                a0, b0 = best[0], best[1]
                dirx, diry = b0.x - a0.x, b0.y - a0.y
                nrm = math.hypot(dirx, diry)
                if nrm > 0:
                    ux2, uy2 = dirx / nrm, diry / nrm
                    nx2, ny2 = -uy2, ux2
                    half_window = max(
                        (s_max - s_min) * self.LOCAL_WINDOW_FRAC, (s_max - s_min) * 1e-3
                    )
                    for j in range(local):
                        if self._time_exceeded(start_time):
                            break
                        tt = (j / (local - 1) - 0.5) * 2.0
                        shift = tt * half_window
                        a_inf = Point(
                            a0.x + nx2 * shift - ux2 * L, a0.y + ny2 * shift - uy2 * L
                        )
                        b_inf = Point(
                            b0.x + nx2 * shift + ux2 * L, b0.y + ny2 * shift + uy2 * L
                        )
                        cand2 = self._best_chord_on_line_global(
                            cake, poly, LS([a_inf, b_inf]), A_star, A_star, 1e12
                        )
                        if cand2 is None:
                            continue
                        a, b, A, _, prop = cand2
                        err = abs(A - A_star)
                        if err < best_err:
                            best = (a, b, A, prop if prop is not None else 0.0)
                            best_err = err

        return best

    def _polish_last_cut(
        self,
        cake: Cake,
        a: Point,
        b: Point,
        poly: Polygon,
        A_star: float,
        start_time: float,
    ) -> Optional[Tuple[Point, Point, float]]:
        try:
            dirx, diry = b.x - a.x, b.y - a.y
            nrm = math.hypot(dirx, diry)
            if nrm == 0:
                return None
            nx, ny = -diry / nrm, dirx / nrm

            minx, miny, maxx, maxy = poly.bounds
            span = max(1e-6, math.hypot(maxx - minx, maxy - miny))
            lo, hi = -0.15 * span, 0.15 * span

            def eval_t(t: float):
                if self._time_exceeded(start_time):
                    return None
                a2 = Point(a.x + nx * t, a.y + ny * t)
                b2 = Point(b.x + nx * t, b.y + ny * t)
                snapped = self._snap_to_cake_perimeter(cake, LS([a2, b2]))
                if snapped is None:
                    return None
                a3, b3 = snapped
                if not self._valid_on_current_cake(cake, a3, b3):
                    return None
                try:
                    sim = cake.copy()
                    sim.cut(a3, b3)
                    cut_line = LS([a3, b3])
                    candidates = []
                    for p in sim.get_pieces():
                        inter = p.intersection(cut_line)
                        if not inter.is_empty and getattr(inter, "length", 0.0) > 1e-9:
                            candidates.append(p)
                    if not candidates:
                        return None
                    produced = min(candidates, key=lambda p_: abs(p_.area - A_star))
                    return (a3, b3, produced.area)
                except Exception:
                    return None

            f_lo = eval_t(lo)
            f_hi = eval_t(hi)
            if f_lo is None and f_hi is None:
                return None

            if f_lo is not None and abs(f_lo[2] - A_star) <= self.band_star:
                return f_lo
            if f_hi is not None and abs(f_hi[2] - A_star) <= self.band_star:
                return f_hi

            left, right = lo, hi
            best = None
            best_err = float("inf")
            for _ in range(40):
                if self._time_exceeded(start_time):
                    break
                mid = 0.5 * (left + right)
                res = eval_t(mid)
                if res is None:
                    right = mid
                    continue
                a3, b3, A = res
                err = abs(A - A_star)
                if err < best_err:
                    best = (a3, b3, A)
                    best_err = err
                if A > A_star:
                    right = mid
                else:
                    left = mid
                if err <= self.band_star * 0.9:
                    break
            return best
        except Exception:
            return None

    def _direction_count(self, poly: Polygon, remaining: int) -> int:
        nverts = len(list(poly.exterior.coords)) - 1 if poly and poly.exterior else 0
        if nverts > 120:
            nd = max(32, self.NUM_DIRECTIONS // 2)
        elif poly.area < 0.05 * max(self.total_area, 1e-9):
            nd = min(64, self.NUM_DIRECTIONS + 16)
        else:
            nd = self.NUM_DIRECTIONS
        if remaining <= 3:
            nd = max(nd, self.NUM_DIRECTIONS + 16)
        return nd

    def _coarse_count(self, remaining: int) -> int:
        if remaining > 6:
            return max(40, self.OFFSETS_COARSE - 8)
        if remaining <= 3:
            return min(64, self.OFFSETS_COARSE + 16)
        return self.OFFSETS_COARSE

    def _local_count(self, remaining: int) -> int:
        if remaining <= 3:
            return min(48, self.OFFSETS_LOCAL + 12)
        return self.OFFSETS_LOCAL

    def _direction_set(self, poly: Polygon) -> List[Tuple[float, float]]:
        nverts = len(list(poly.exterior.coords)) - 1 if poly and poly.exterior else 0
        if nverts > 120:
            nd = max(24, self.NUM_DIRECTIONS // 2)
        elif poly.area < 0.05 * max(self.total_area, 1e-9):
            nd = min(64, self.NUM_DIRECTIONS + 16)
        else:
            nd = self.NUM_DIRECTIONS
        dirs: List[Tuple[float, float]] = []
        for k in range(nd):
            th = (math.pi * k) / nd
            dirs.append((math.cos(th), math.sin(th)))
        return dirs

    def _vertex_pair_chords(self, polygon: Polygon) -> List[LS]:
        out: List[LS] = []
        try:
            coords = list(polygon.exterior.coords[:-1])
        except Exception:
            return out
        n = len(coords)
        if n < 2 or n > min(self.MAX_VERTEX_ENUM, 160):
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
                    try:
                        if seg.difference(poly).is_empty:
                            out.append(seg)
                    except Exception:
                        continue
        return out

    def _snap_to_cake_perimeter(
        self, cake_obj: Cake, seg: LS
    ) -> Optional[Tuple[Point, Point]]:
        try:
            outer = cake_obj.exterior_shape
            if outer is None or outer.is_empty:
                return None
            (xa, ya), (xb, yb) = list(seg.coords)[0], list(seg.coords)[-1]
            dx, dy = xb - xa, yb - ya
            nrm = math.hypot(dx, dy)
            if nrm == 0:
                return None
            ux, uy = dx / nrm, dy / nrm
            cx, cy = (xa + xb) * 0.5, (ya + yb) * 0.5

            minx, miny, maxx, maxy = outer.bounds
            diameter = max(1e-6, math.hypot(maxx - minx, maxy - miny)) * 4.0
            a_inf = Point(cx - ux * diameter, cy - uy * diameter)
            b_inf = Point(cx + ux * diameter, cy + uy * diameter)
            inf_line = LS([a_inf, b_inf])

            inside = outer.intersection(inf_line)
            if inside.is_empty:
                return None

            cand_lines: List[LS] = []
            if isinstance(inside, LS):
                cand_lines = [inside]
            elif hasattr(inside, "geoms"):
                cand_lines = [g for g in inside.geoms if isinstance(g, LS)]
            if not cand_lines:
                return None

            best_line = None
            best_overlap = -1.0
            best_length = float("inf")
            seg_buf = seg.buffer(max(1e-9, min(1e-6, diameter * 1e-9)))
            for L in cand_lines:
                try:
                    ov = L.intersection(seg_buf)
                    overlap_len = ov.length if hasattr(ov, "length") else 0.0
                except Exception:
                    overlap_len = 0.0
                L_len = L.length
                if (overlap_len > best_overlap) or (
                    overlap_len == best_overlap and L_len < best_length
                ):
                    best_overlap = overlap_len
                    best_length = L_len
                    best_line = L

            if best_line is None:
                return None

            coords = list(best_line.coords)
            if len(coords) < 2:
                return None
            p0 = Point(coords[0])
            p1 = Point(coords[-1])
            if p0.distance(p1) < self.min_seg_len:
                return None
            return (p0, p1)
        except Exception:
            return None

    def _touches_other_pieces(self, cake_obj: Cake, poly: Polygon, seg: LS) -> bool:
        try:
            (xa, ya), (xb, yb) = list(seg.coords)[0], list(seg.coords)[-1]
            a = Point(xa, ya)
            b = Point(xb, yb)
            outer = cake_obj.exterior_shape

            def is_endpoint(pt: Point) -> bool:
                return (pt.distance(a) <= 1e-10) or (pt.distance(b) <= 1e-10)

            for p in cake_obj.get_pieces():
                if p.equals(poly) or (
                    abs(p.area - poly.area) < 1e-9
                    and p.symmetric_difference(poly).area < 1e-9
                ):
                    continue

                inter = p.intersection(seg)
                if inter.is_empty:
                    continue

                if hasattr(inter, "length") and inter.length > 1e-12:
                    return True

                if inter.geom_type == "Point":
                    pt = inter
                    if not is_endpoint(pt):
                        return True
                    try:
                        if (
                            outer is None
                            or outer.is_empty
                            or not outer.boundary.covers(pt)
                        ):
                            return True
                    except Exception:
                        return True

                if inter.geom_type in ("MultiPoint", "GeometryCollection"):
                    for g in getattr(inter, "geoms", []):
                        if g.geom_type == "Point":
                            if (not is_endpoint(g)) or (
                                outer is None
                                or outer.is_empty
                                or not outer.boundary.covers(g)
                            ):
                                return True
                        elif hasattr(g, "length") and g.length > 1e-12:
                            return True
            return False
        except Exception:
            return True

    def _prop_of(self, p: Polygon) -> float:
        try:
            if p.area <= 0:
                return 0.0
            inner_overlap = p.intersection(self.inner_cake)
            inner_area = inner_overlap.area if not inner_overlap.is_empty else 0.0
            crust_area = max(0.0, p.area - inner_area)
            v = crust_area / p.area
            if not math.isfinite(v):
                return 0.0
            return v
        except Exception:
            return 0.0

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
