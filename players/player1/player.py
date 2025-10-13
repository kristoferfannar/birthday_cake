from __future__ import annotations
from typing import List, Tuple, Optional
import math

from shapely.geometry import (
    Point as ShyPoint,
    Polygon,
    MultiPolygon,
    GeometryCollection,
    LineString,
    MultiLineString,
)
from shapely.ops import split as sh_split, nearest_points

from players.player import Player
from src.cake import Cake

# ==================== Config ====================
AREA_TOL = 0.5           # cm^2 tolerance for exterior area match
CRUST_W  = 1.0           # (not used now for scoring; left in case we revert)
ANGLE_DEG_STEP = 6       # angle sweep granularity (smaller = slower, finer)
BSEARCH_ITERS = 60
SNAP_EPS = 1e-7
MIN_SMALL_AREA_FLOOR = 1.0  # avoid engine "piece too small" errors

# ==================== Helpers ====================
def _as_points(coords) -> List[Tuple[float, float]]:
    return [(float(x), float(y)) for x, y in coords]

def _heal_polygon(P: Polygon) -> Polygon:
    Q = P.buffer(0)
    if isinstance(Q, Polygon):
        return Q
    if isinstance(Q, MultiPolygon) and Q.geoms:
        return max((g for g in Q.geoms if isinstance(g, Polygon)), key=lambda g: g.area)
    return P

def _long_line(theta: float, c: float, pad: float = 1e5) -> LineString:
    nx, ny = math.cos(theta), math.sin(theta)  # unit normal
    vx, vy = -ny, nx
    mx, my = c * nx, c * ny
    return LineString([(mx - pad * vx, my - pad * vy), (mx + pad * vx, my + pad * vy)])

def _bounds_proj(P: Polygon, theta: float) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = P.bounds
    c, s = math.cos(theta), math.sin(theta)
    vals = [c * xmin + s * ymin, c * xmin + s * ymax, c * xmax + s * ymin, c * xmax * 1.0 + s * ymax]
    # NOTE: tiny change below: correct last term (typo guard)
    vals[-1] = c * xmax + s * ymax
    return min(vals), max(vals)

def _longest_chord(P: Polygon, theta: float, c: float) -> Optional[Tuple[Tuple[float,float], Tuple[float,float]]]:
    inter = P.intersection(_long_line(theta, c))
    best = None
    bestL = -1.0

    def add(ls: LineString):
        nonlocal best, bestL
        if not isinstance(ls, LineString) or len(ls.coords) < 2:
            return
        L = ls.length
        if L > bestL:
            bestL = L
            a, b = ls.coords[0], ls.coords[-1]
            best = ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))

    if isinstance(inter, LineString):
        add(inter)
    elif isinstance(inter, MultiLineString):
        for g in inter.geoms:
            add(g)
    elif isinstance(inter, GeometryCollection):
        for g in inter.geoms:
            if isinstance(g, LineString):
                add(g)
            elif isinstance(g, MultiLineString):
                for h in g.geoms:
                    add(h)
    return best

def _split_by_chord(P: Polygon, chord: Tuple[Tuple[float,float], Tuple[float,float]]) -> List[Polygon]:
    (ax, ay), (bx, by) = chord
    def _do(splitter) -> List[Polygon]:
        try:
            res = sh_split(P, splitter)
        except Exception:
            return []
        return [g for g in getattr(res, "geoms", []) if isinstance(g, Polygon) and g.area > 1e-10]
    seg = LineString([(ax, ay), (bx, by)])
    parts = _do(seg)
    if len(parts) < 2:
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L > 0:
            eps = SNAP_EPS
            seg2 = LineString([(ax - eps*dx, ay - eps*dy), (bx + eps*dx, by + eps*dy)])
            parts = _do(seg2)
    return parts

def _split_by_line(P: Polygon, theta: float, c: float) -> Tuple[float, float, List[Polygon], Optional[Tuple[Tuple[float,float], Tuple[float,float]]]]:
    chord = _longest_chord(P, theta, c)
    if chord is None:
        return 0.0, P.area, [P], None
    parts = _split_by_chord(P, chord)
    if len(parts) < 2:
        parts = [g for g in getattr(sh_split(P, _long_line(theta, c)), "geoms", []) if isinstance(g, Polygon) and g.area > 1e-10]
    if len(parts) < 2:
        return 0.0, P.area, [P], chord
    parts.sort(key=lambda g: g.area)
    return parts[0].area, parts[-1].area, parts, chord

def _snap_to_boundary(pt: ShyPoint, P: Polygon) -> ShyPoint:
    _, on = nearest_points(pt, P.boundary)
    return on

def _bisection_offset_for_alpha(P: Polygon, alpha: float, theta: float) -> Optional[float]:
    """Find c s.t. exterior area fraction on one side equals alpha."""
    if P.area <= 1e-12:
        return None
    lo, hi = _bounds_proj(P, theta)
    target = alpha * P.area
    for _ in range(BSEARCH_ITERS):
        mid = 0.5*(lo+hi)
        n = (math.cos(theta), math.sin(theta))
        _, _, parts, _ = _split_by_line(P, theta, mid)
        left = 0.0
        for g in parts:
            ctr = g.representative_point()
            if n[0]*ctr.x + n[1]*ctr.y - mid <= 0:
                left += g.area
        if abs(left - target) <= AREA_TOL:
            return mid
        if left < target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

# -------------- NEW: θ scoring using engine's ratios --------------
def _solve_alpha_line(
    P: Polygon,
    alpha: float,
    angle_candidates: List[float],
    cake_ratio_source: Cake,  # use engine's get_piece_ratio to score
) -> Optional[Tuple[float, float, Tuple[Tuple[float,float], Tuple[float,float]]]]:
    """
    For each θ:
      - solve c so exterior area on one side is α·area(P)
      - split P; compute r1,r2 via cake_ratio_source.get_piece_ratio(piece)
      - pick θ minimizing |r1 - r2|
    Return (theta, c, chord).
    """
    P = _heal_polygon(P)
    best = None
    for deg in angle_candidates:
        theta = (deg % 180.0) * math.pi / 180.0  # modulo π
        c = _bisection_offset_for_alpha(P, alpha, theta)
        if c is None:
            continue
        aS, aL, parts, chord = _split_by_line(P, theta, c)
        if chord is None or len(parts) < 2:
            continue
        parts_sorted = sorted(parts, key=lambda g: g.area, reverse=True)[:2]
        big, small = parts_sorted[0], parts_sorted[1]
        try:
            r_big = cake_ratio_source.get_piece_ratio(big)
            r_small = cake_ratio_source.get_piece_ratio(small)
        except Exception:
            # fall back to equal score; still allow area correctness to drive recursion
            r_big, r_small = 0.0, 0.0
        score = abs(r_big - r_small)
        if (best is None) or (score < best[0]) or (math.isclose(score, best[0]) and abs(aS - alpha*P.area) < abs(best[3] - alpha*P.area)):
            best = (score, theta, c, aS, chord)
    if best is None:
        return None
    _, theta, c, _, chord = best
    return theta, c, chord

def _angle_grid(step_deg: int = ANGLE_DEG_STEP) -> List[float]:
    return [float(d) for d in range(0, 180, step_deg)]  # [0,180)

# -------------------- Sliver-avoid & validation --------------------
def _small_area(P: Polygon, theta: float, c: float) -> float:
    aS, _, _, _ = _split_by_line(P, theta, c)
    return aS

def _nudge_c_to_avoid_sliver(P: Polygon, theta: float, c: float, min_small_area: float) -> float:
    (xmin, ymin, xmax, ymax) = P.bounds
    step = 0.01 * ((xmax - xmin) + (ymax - ymin))
    best_c = c
    best_a = _small_area(P, theta, c)
    if best_a >= min_small_area:
        return c
    for k in range(1, 21):
        for sign in (+1, -1):
            c2 = c + sign * k * step
            a2 = _small_area(P, theta, c2)
            if a2 > best_a:
                best_a, best_c = a2, c2
                if best_a >= min_small_area:
                    return best_c
    return best_c

def _apply_validated_cut(
    cake_sim: Cake,
    P: Polygon,
    theta: float,
    c: float,
    chord_hint: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None,
    min_small_area: float = MIN_SMALL_AREA_FLOOR,
) -> Optional[Tuple[Polygon, Polygon, Tuple[ShyPoint, ShyPoint]]]:
    c_adj = _nudge_c_to_avoid_sliver(P, theta, c, min_small_area)
    chord = chord_hint or _longest_chord(P, theta, c_adj)
    if chord is None:
        return None
    (ax, ay), (bx, by) = chord
    pa = _snap_to_boundary(ShyPoint(ax, ay), P)
    pb = _snap_to_boundary(ShyPoint(bx, by), P)

    ok, _ = cake_sim.cut_is_valid(pa, pb)
    if not ok:
        (xmin, ymin, xmax, ymax) = P.bounds
        step = 0.01 * ((xmax - xmin) + (ymax - ymin))
        for mul in (1, -1, 2, -2):
            c_try = c_adj + mul * step
            chord2 = _longest_chord(P, theta, c_try)
            if chord2 is None:
                continue
            (ax2, ay2), (bx2, by2) = chord2
            pa2 = _snap_to_boundary(ShyPoint(ax2, ay2), P)
            pb2 = _snap_to_boundary(ShyPoint(bx2, by2), P)
            ok2, _ = cake_sim.cut_is_valid(pa2, pb2)
            if ok2:
                pa, pb = pa2, pb2
                c_adj = c_try
                chord = chord2
                break
        else:
            return None

    try:
        cake_sim.cut(pa, pb)
    except Exception:
        return None

    parts = _split_by_chord(P, ((pa.x, pa.y), (pb.x, pb.y)))
    if len(parts) < 2:
        parts = [g for g in getattr(sh_split(P, LineString([(pa.x, pa.y), (pb.x, pb.y)])), "geoms", [])
                 if isinstance(g, Polygon) and g.area > 1e-10]
    if len(parts) < 2:
        return None

    parts.sort(key=lambda g: g.area, reverse=True)
    return parts[0], parts[1], (pa, pb)

# ==================== Planner ====================
def _plan_piece(
    P: Polygon,
    m: int,
    total_area_unit: float,
    cuts_out: List[Tuple[ShyPoint, ShyPoint]],
    cake_sim: Cake,
) -> None:
    P = _heal_polygon(P)
    if m <= 1 or P.area <= 1e-9:
        return

    angles = _angle_grid()

    if m % 4 == 0:
        sol = _solve_alpha_line(P, 0.5, angles, cake_sim)
        if sol is None:
            return
        theta, c, chord = sol
        applied = _apply_validated_cut(cake_sim, P, theta, c, chord_hint=chord)
        if applied is None:
            return
        L, S, (pa, pb) = applied
        cuts_out.append((pa, pb))
        _plan_piece(L, m // 2, total_area_unit, cuts_out, cake_sim)
        _plan_piece(S, m // 2, total_area_unit, cuts_out, cake_sim)
        return

    if m % 4 == 2:
        k = (m - 2) // 4
        alpha = (2.0 * k) / (4.0 * k + 2.0) if m >= 2 else 0.5
        sol = _solve_alpha_line(P, alpha, angles, cake_sim)
        if sol is not None:
            theta, c, chord = sol
            applied = _apply_validated_cut(cake_sim, P, theta, c, chord_hint=chord)
            if applied is not None:
                L, S, (pa, pb) = applied
                cuts_out.append((pa, pb))
                m1 = int(round(alpha * m)); m1 = max(1, min(m-1, m1))
                m2 = m - m1
                if L.area >= S.area:
                    _plan_piece(L, m2, total_area_unit, cuts_out, cake_sim)
                    _plan_piece(S, m1, total_area_unit, cuts_out, cake_sim)
                else:
                    _plan_piece(L, m1, total_area_unit, cuts_out, cake_sim)
                    _plan_piece(S, m2, total_area_unit, cuts_out, cake_sim)
                return

        # Fallback: bisect, then carve 1/n from each half with parallel family
        sol2 = _solve_alpha_line(P, 0.5, angles, cake_sim)
        if sol2 is None:
            return
        theta0, c0, chord0 = sol2
        applied0 = _apply_validated_cut(cake_sim, P, theta0, c0, chord_hint=chord0)
        if applied0 is None:
            return
        H1, H2, (p0a, p0b) = applied0
        cuts_out.append((p0a, p0b))

        halves = [H1, H2]
        remainders: List[Polygon] = []

        for H in halves:
            if H.is_empty:
                continue
            target_alpha_half = (total_area_unit) / max(H.area, 1e-12)
            target_alpha_half = max(0.0, min(1.0, target_alpha_half))
            local_angles = [((theta0*180/math.pi + d) % 180.0) for d in (-12,-6,0,6,12)]
            solH = _solve_alpha_line(H, target_alpha_half, local_angles, cake_sim)
            if solH is None:
                remainders.append(H); continue
            th, cH, chordH = solH
            appliedH = _apply_validated_cut(cake_sim, H, th, cH, chord_hint=chordH, min_small_area=MIN_SMALL_AREA_FLOOR)
            if appliedH is None:
                remainders.append(H); continue
            Lh, Sh, (pha, phb) = appliedH
            cuts_out.append((pha, phb))
            small, large = (Sh, Lh) if Sh.area <= Lh.area else (Lh, Sh)
            remainders.append(large)

        rem = m - 2
        if rem <= 0 or not remainders:
            return
        totalA = sum(g.area for g in remainders)
        counts = [int(round(rem * (g.area / totalA))) for g in remainders]
        diff = rem - sum(counts)
        order = sorted(range(len(remainders)), key=lambda i: remainders[i].area, reverse=True)
        idx = 0
        while diff != 0 and order:
            j = order[idx % len(order)]
            if diff > 0: counts[j] += 1; diff -= 1
            else:
                if counts[j] > 0: counts[j] -= 1; diff += 1
            idx += 1
        for g, cnt in zip(remainders, counts):
            if cnt > 0:
                _plan_piece(g, cnt, total_area_unit, cuts_out, cake_sim)
        return

    # default: odd m → bisect
    sol = _solve_alpha_line(P, 0.5, angles, cake_sim)
    if sol is None:
        return
    th, cc, chord = sol
    applied = _apply_validated_cut(cake_sim, P, th, cc, chord_hint=chord)
    if applied is None:
        return
    L, S, (pa, pb) = applied
    cuts_out.append((pa, pb))
    m1 = m // 2; m2 = m - m1
    _plan_piece(L, m1, total_area_unit, cuts_out, cake_sim)
    _plan_piece(S, m2, total_area_unit, cuts_out, cake_sim)

# ==================== Player ====================
class Player1(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None = None) -> None:
        super().__init__(children, cake, cake_path)

    def _extract_polygon(self) -> Polygon:
        ext = getattr(self.cake, "exterior_shape", None)
        if isinstance(ext, Polygon):
            return _heal_polygon(ext)
        if hasattr(self.cake, "get_boundary_points") and callable(self.cake.get_boundary_points):
            pts = _as_points(self.cake.get_boundary_points())
            if len(pts) >= 3:
                if pts[0] == pts[-1]:
                    pts = pts[:-1]
                return _heal_polygon(Polygon(pts))
        raise ValueError("Player1: cannot extract exterior polygon")

    def get_cuts(self) -> List[Tuple[ShyPoint, ShyPoint]]:
        P0 = self._extract_polygon()
        if not isinstance(P0, Polygon) or P0.area <= 1e-9:
            return []
        n = int(self.children)
        if n <= 1:
            return []

        cuts: List[Tuple[ShyPoint, ShyPoint]] = []
        unit = P0.area / n
        cake_sim = self.cake.copy()   # use this to score ratios & validate

        _plan_piece(P0, n, unit, cuts, cake_sim)

        if len(cuts) > n - 1:
            cuts = cuts[: n - 1]
        return cuts
