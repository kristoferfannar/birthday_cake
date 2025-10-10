from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import shapely
from shapely.geometry import (
    LinearRing as ShapelyLinearRing,
    MultiPolygon as _ShMulti,
    Point,
    Polygon as ShapelyPolygon,
    Polygon as _ShPoly,
    box as _box,
)
from shapely.geometry.base import BaseGeometry

try:
    # Shapely 2.x
    from shapely.validation import make_valid as _make_valid
except Exception:  # pragma: no cover - older shapely
    _make_valid = None

from players.player import Player
from src.cake import Cake

# ----------------- Types & eps -----------------
Point2D = Tuple[float, float]
Poly = List[Point2D]
EPS = 1e-12


# ----------------- Polygon access -----------------
def _as_points(obj: Iterable) -> Poly:
    pts: Poly = []
    for p in obj:
        x, y = float(p[0]), float(p[1])
        pts.append((x, y))
    return pts


def _extract_polygon(cake: Cake) -> Poly:
    """Extract the cake outline in cm as list[(x,y)]."""
    # 1) Preferred: Shapely exterior on cake.exterior_shape
    ext = getattr(cake, "exterior_shape", None)
    if isinstance(ext, (ShapelyPolygon, ShapelyLinearRing, BaseGeometry)):
        coords = (
            list(ext.exterior.coords)
            if hasattr(ext, "exterior") and getattr(ext, "exterior")
            else list(ext.coords)
        )
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]  # drop duplicate closing vertex
        pts = _as_points(coords)
        if len(pts) >= 3:
            return pts

    # 2) Fallback: boundary points API (already in cm)
    if hasattr(cake, "get_boundary_points") and callable(cake.get_boundary_points):
        pts = _as_points(cake.get_boundary_points())
        if len(pts) >= 3:
            return pts

    # 3) Rare: interior shape
    intr = getattr(cake, "interior_shape", None)
    if isinstance(intr, (ShapelyPolygon, ShapelyLinearRing, BaseGeometry)):
        coords = (
            list(intr.exterior.coords)
            if hasattr(intr, "exterior") and getattr(intr, "exterior")
            else list(intr.coords)
        )
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        pts = _as_points(coords)
        if len(pts) >= 3:
            return pts

    raise ValueError("Player1: Could not find polygon points on Cake.")


# ----------------- Geometry core -----------------
def _area(poly: Poly) -> float:
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return 0.5 * a  # signed


def _ensure_ccw(poly: Poly) -> Poly:
    return poly if _area(poly) > 0 else list(reversed(poly))


def _bbox(pts: Poly) -> Tuple[float, float, float, float]:
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return min(xs), min(ys), max(xs), max(ys)


# --- clip polygon to half-plane x <= t (Sutherland–Hodgman) ---
def _clip_x_le(poly: Poly, t: float) -> Poly:
    def inside(p: Point2D) -> bool:
        return p[0] <= t + 1e-12

    def intersect(a: Point2D, b: Point2D) -> Point2D:
        x1, y1 = a
        x2, y2 = b
        dx = x2 - x1
        if abs(dx) < EPS:
            return (t, y1)  # vertical segment at x ~= t
        s = (t - x1) / dx
        return (t, y1 + s * (y2 - y1))

    out: Poly = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prv = poly[(i - 1) % n]
        ci, pi = inside(cur), inside(prv)
        if ci:
            if not pi:
                out.append(intersect(prv, cur))
            out.append(cur)
        elif pi:
            out.append(intersect(prv, cur))
    return out


def _area_left_of_x(poly_ccw: Poly, x: float) -> float:
    return abs(_area(_clip_x_le(poly_ccw, x)))


# --- vertical intersections & chords ---
def _vertical_intersections(poly: Poly, x: float) -> List[float]:
    ys: List[float] = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x1 <= x <= x2) or (x2 <= x <= x1):
            dx = x2 - x1
            if abs(dx) < EPS:
                ys.extend([y1, y2])  # collect both ends of a vertical edge
            else:
                s = (x - x1) / dx
                ys.append(y1 + s * (y2 - y1))
    ys.sort()
    out: List[float] = []
    for y in ys:
        if not out or abs(y - out[-1]) > 1e-9:  # dedupe vertex touches
            out.append(y)
    return out


def _vertical_chords(poly: Poly, x: float) -> List[Tuple[float, float]]:
    ys = _vertical_intersections(poly, x)
    chords: List[Tuple[float, float]] = []
    for i in range(0, len(ys) - 1, 2):
        y0, y1 = ys[i], ys[i + 1]
        if y1 - y0 > 1e-9:
            chords.append((y0, y1))
    return chords


def _find_cut_x(poly_ccw: Poly, target_area: float, lo: float, hi: float) -> float:
    for _ in range(60):  # ~double precision
        mid = 0.5 * (lo + hi)
        a = _area_left_of_x(poly_ccw, mid)
        if a < target_area:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- generic segment validator using engine hooks (works for any segment) ---
def _is_good_segment(cake: Cake, a: Point2D, b: Point2D) -> bool:
    pa, pb = Point(a[0], a[1]), Point(b[0], b[1])
    if hasattr(cake, "cut_is_valid"):
        try:
            ok, _ = cake.cut_is_valid(pa, pb)
            if not ok:
                return False
        except Exception:
            return False
    if hasattr(cake, "does_line_cut_piece_well"):
        if not cake.does_line_cut_piece_well(pa, pb):
            return False
    return True


def _is_good_cut(cake: Cake, x: float, y0: float, y1: float) -> bool:
    """Back-compat helper for vertical segments."""
    return _is_good_segment(cake, (x, y0), (x, y1))


# ----------------- NEW: horizontal clip & chords -----------------
def _clip_y_le(poly: Poly, t: float) -> Poly:
    def inside(p: Point2D) -> bool:
        return p[1] <= t + 1e-12

    def intersect(a: Point2D, b: Point2D) -> Point2D:
        x1, y1 = a
        x2, y2 = b
        dy = y2 - y1
        if abs(dy) < EPS:
            return (x1, t)
        s = (t - y1) / dy
        return (x1 + s * (x2 - x1), t)

    out: Poly = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prv = poly[(i - 1) % n]
        ci, pi = inside(cur), inside(prv)
        if ci:
            if not pi:
                out.append(intersect(prv, cur))
            out.append(cur)
        elif pi:
            out.append(intersect(prv, cur))
    return out


def _area_below_y(poly_ccw: Poly, y: float) -> float:
    return abs(_area(_clip_y_le(poly_ccw, y)))


def _horizontal_intersections(poly: Poly, y: float) -> List[float]:
    xs: List[float] = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            dy = y2 - y1
            if abs(dy) < EPS:
                xs.extend([x1, x2])
            else:
                s = (y - y1) / dy
                xs.append(x1 + s * (x2 - x1))
    xs.sort()
    out: List[float] = []
    for x in xs:
        if not out or abs(x - out[-1]) > 1e-9:
            out.append(x)
    return out


def _horizontal_chords(poly: Poly, y: float) -> List[Tuple[float, float]]:
    xs = _horizontal_intersections(poly, y)
    chords: List[Tuple[float, float]] = []
    for i in range(0, len(xs) - 1, 2):
        x0, x1 = xs[i], xs[i + 1]
        if x1 - x0 > 1e-9:
            chords.append((x0, x1))
    return chords


def _find_cut_y(poly_ccw: Poly, target_area: float, lo: float, hi: float) -> float:
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a = _area_below_y(poly_ccw, mid)
        if a < target_area:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------- NEW: per-piece area targeting helpers -----------------
def _area_between(poly_ccw: Poly, xa: float, xb: float) -> float:
    """Area of the polygon part with xa < x <= xb (assuming xa < xb)."""
    if xb <= xa + 1e-12:
        return 0.0
    return _area_left_of_x(poly_ccw, xb) - _area_left_of_x(poly_ccw, xa)


def _find_next_cut_x(
    poly_ccw: Poly, target_piece_area: float, xa: float, xb: float
) -> float:
    """Find x in [xa, xb] such that area_between(xa, x) ~= target_piece_area."""
    lo, hi = xa, xb
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a = _area_between(poly_ccw, xa, mid)
        if a < target_piece_area:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------- NEW: shapely helpers for crust & pieces -----------------
def _to_shpoly(poly: Poly) -> _ShPoly:
    return _ShPoly(poly)


def _explode_to_polys(g) -> List[_ShPoly]:
    if isinstance(g, _ShPoly):
        return [g]
    if isinstance(g, _ShMulti):
        return [p for p in g.geoms if isinstance(p, _ShPoly)]
    return []


def _clean_geom(g):
    # robustify invalid rings/holes
    try:
        if _make_valid:
            g = _make_valid(g)
        else:
            g = g.buffer(0)
    except Exception:
        g = g.buffer(0)
    # snap-round to a fine grid to avoid free-hole glitches
    try:
        g = shapely.set_precision(g, 1e-9)
    except Exception:
        pass
    return g


def _crust_interior_polys(poly: Poly, crust_width: float = 1.0):
    """Return (P, P_interior, crust_area, interior_area)."""
    P = _to_shpoly(poly)
    if not P.is_valid or P.area <= 0:
        return P, _ShPoly(), 0.0, 0.0
    Pin = P.buffer(-crust_width, join_style=2)  # mitre-like
    if Pin.is_empty:
        Pin = _ShPoly()
    crust_area = max(P.area - Pin.area, 0.0)
    return P, Pin, crust_area, Pin.area


def _piece_crust_ratio(poly_piece: Poly, Pin_global: _ShPoly) -> float:
    """Crust ratio = crust_area / total_area for this piece using global interior P_in."""
    Pk = _to_shpoly(poly_piece)
    if Pk.is_empty or Pk.area <= 0:
        return 0.0
    interior_k = Pk.intersection(Pin_global) if not Pin_global.is_empty else _ShPoly()
    crust_k = max(Pk.area - interior_k.area, 0.0)
    return crust_k / Pk.area if Pk.area > 0 else 0.0


# ---------- ROBUST slab builders: intersection with strip rectangles ----------
def _slab_pieces_vertical(poly_ccw: Poly, xs: List[float]) -> List[Poly]:
    """Return a list of polygon pieces for vertical slabs defined by xs (monotone)."""
    pieces: List[Poly] = []
    P = _clean_geom(_to_shpoly(poly_ccw))
    xmin, ymin, xmax, ymax = _bbox(poly_ccw)
    EPSG = 1e-9
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        if xb <= xa + EPSG:
            continue
        rect = _box(xa - EPSG, ymin - EPSG, xb + EPSG, ymax + EPSG)
        slab = _clean_geom(P.intersection(rect))
        for p in _explode_to_polys(slab):
            if p.area > 1e-10 and p.exterior:
                pieces.append(_as_points(list(p.exterior.coords)[:-1]))
    return pieces


def _slab_pieces_horizontal(poly_ccw: Poly, ys: List[float]) -> List[Poly]:
    pieces: List[Poly] = []
    P = _clean_geom(_to_shpoly(poly_ccw))
    xmin, ymin, xmax, ymax = _bbox(poly_ccw)
    EPSG = 1e-9
    for i in range(len(ys) - 1):
        ya, yb = ys[i], ys[i + 1]
        if yb <= ya + EPSG:
            continue
        rect = _box(xmin - EPSG, ya - EPSG, xmax + EPSG, yb + EPSG)
        slab = _clean_geom(P.intersection(rect))
        for p in _explode_to_polys(slab):
            if p.area > 1e-10 and p.exterior:
                pieces.append(_as_points(list(p.exterior.coords)[:-1]))
    return pieces


def _score_set(
    pieces: List[Poly], Pin_global: _ShPoly, target_piece_area: float
) -> Tuple[float, float]:
    """
    Return (max_area_err, crust_ratio_range).
    max_area_err: max |area_i - target|
    crust_ratio_range: max(r_i) - min(r_i)
    """
    areas: List[float] = []
    ratios: List[float] = []
    for Pk in pieces:
        A = abs(_area(Pk))
        areas.append(A)
        ratios.append(_piece_crust_ratio(Pk, Pin_global))
    max_err = max(abs(A - target_piece_area) for A in areas) if areas else float("inf")
    rng = (max(ratios) - min(ratios)) if len(ratios) >= 2 else 0.0
    return max_err, rng


def _evaluate_plan(
    poly_ccw: Poly, cuts: List[Tuple[Point, Point]], target_piece_area: float
) -> Tuple[float, float, int]:
    """
    Score a plan of parallel cuts.
    Returns (max_area_err, crust_ratio_range, piece_count).
    Detects orientation by reading segments.
    """
    if not cuts:
        return float("inf"), float("inf"), 0

    # orientation detection (assume all parallel)
    is_vertical = all(abs(c[0].x - c[1].x) < 1e-9 for c in cuts)
    is_horizontal = all(abs(c[0].y - c[1].y) < 1e-9 for c in cuts)

    xmin, ymin, xmax, ymax = _bbox(poly_ccw)
    _, Pin, _, _ = _crust_interior_polys(poly_ccw, crust_width=1.0)

    if is_vertical:
        xs = [xmin] + [c[0].x for c in cuts] + [xmax]
        xs = sorted(xs)
        pieces = _slab_pieces_vertical(poly_ccw, xs)
    elif is_horizontal:
        ys = [ymin] + [c[0].y for c in cuts] + [ymax]
        ys = sorted(ys)
        pieces = _slab_pieces_horizontal(poly_ccw, ys)
    else:
        # mixed orientation → not supported in this scorer
        return float("inf"), float("inf"), 0

    max_err, crust_rng = _score_set(pieces, Pin, target_piece_area)
    return max_err, crust_rng, len(pieces)


# ----------------- NEW: human-style fallback -----------------
def _human_style_fallback(
    poly_ccw: Poly, n: int, cake: Cake
) -> List[Tuple[Point, Point]]:
    """
    Strategy:
      1) Try ONE big cut, vertical or horizontal, that splits kids into k and n-k (k=floor(n/2)).
      2) Finish each side with straight parallel cuts (same orientation).
    """
    xmin, ymin, xmax, ymax = _bbox(poly_ccw)
    total = abs(_area(poly_ccw))
    target_piece = total / n
    k_left = n // 2
    k_right = n - k_left

    best_plan: Optional[List[Tuple[Point, Point]]] = None
    best_score: Optional[Tuple[float, float]] = None  # (max_err, crust_rng)

    # ---- Vertical option ----
    if xmax - xmin > 0:
        target_left = total * (k_left / n)
        x0 = _find_cut_x(poly_ccw, target_left, xmin, xmax)
        chords = _vertical_chords(poly_ccw, x0)
        if chords:
            yl, yh = max(chords, key=lambda ab: ab[1] - ab[0])
            if _is_good_segment(cake, (x0, yl), (x0, yh)):
                cuts_v: List[Tuple[Point, Point]] = [(Point(x0, yl), Point(x0, yh))]

                # Build left-side cuts
                xL = xmin
                for _ in range(max(0, k_left - 1)):
                    xL = _find_next_cut_x(poly_ccw, target_piece, xL, x0)
                    ch = _vertical_chords(poly_ccw, xL)
                    if ch:
                        yl2, yh2 = max(ch, key=lambda ab: ab[1] - ab[0])
                        if _is_good_segment(cake, (xL, yl2), (xL, yh2)):
                            cuts_v.append((Point(xL, yl2), Point(xL, yh2)))

                # Build right-side cuts
                xR = x0
                for _ in range(max(0, k_right - 1)):
                    xR = _find_next_cut_x(poly_ccw, target_piece, xR, xmax)
                    ch = _vertical_chords(poly_ccw, xR)
                    if ch:
                        yl2, yh2 = max(ch, key=lambda ab: ab[1] - ab[0])
                        if _is_good_segment(cake, (xR, yl2), (xR, yh2)):
                            cuts_v.append((Point(xR, yl2), Point(xR, yh2)))

                # Score
                sc = _evaluate_plan(poly_ccw, cuts_v, target_piece)
                if sc[2] == n:  # piece count ok
                    best_plan, best_score = cuts_v, (sc[0], sc[1])

    # ---- Horizontal option ----
    if ymax - ymin > 0:
        target_below = total * (k_left / n)
        y0 = _find_cut_y(poly_ccw, target_below, ymin, ymax)
        chords_h = _horizontal_chords(poly_ccw, y0)
        if chords_h:
            xl, xr = max(chords_h, key=lambda ab: ab[1] - ab[0])
            if _is_good_segment(cake, (xl, y0), (xr, y0)):
                cuts_h: List[Tuple[Point, Point]] = [(Point(xl, y0), Point(xr, y0))]

                # Bottom group
                yL = ymin
                for _ in range(max(0, k_left - 1)):
                    yL = _find_cut_y(poly_ccw, target_piece, yL, y0)
                    ch = _horizontal_chords(poly_ccw, yL)
                    if ch:
                        xl2, xr2 = max(ch, key=lambda ab: ab[1] - ab[0])
                        if _is_good_segment(cake, (xl2, yL), (xr2, yL)):
                            cuts_h.append((Point(xl2, yL), Point(xr2, yL)))

                # Top group
                yU = y0
                for _ in range(max(0, k_right - 1)):
                    yU = _find_cut_y(poly_ccw, target_piece, yU, ymax)
                    ch = _horizontal_chords(poly_ccw, yU)
                    if ch:
                        xl2, xr2 = max(ch, key=lambda ab: ab[1] - ab[0])
                        if _is_good_segment(cake, (xl2, yU), (xr2, yU)):
                            cuts_h.append((Point(xl2, yU), Point(xr2, yU)))

                sc = _evaluate_plan(poly_ccw, cuts_h, target_piece)
                if sc[2] == n:
                    if best_score is None or (sc[0], sc[1]) < best_score:
                        best_plan, best_score = cuts_h, (sc[0], sc[1])

    return best_plan or []


# ----------------- Player -----------------
class Player1(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def get_cuts(self) -> List[Tuple[Point, Point]]:
        """
        Always compute BOTH:
          (A) Primary: vertical straight cuts with per-piece area targeting (equal areas).
          (B) Fallback: one big vertical OR horizontal cut to split kids in half, then
                        finish each side with parallel cuts.
        Score both on (max_area_err, crust_ratio_range) and return the better plan.
        Ties prefer the primary plan.
        """
        n = int(self.children)
        if n <= 1:
            return []

        # polygon in cm
        poly = _extract_polygon(self.cake)
        poly = _ensure_ccw(poly)

        total = abs(_area(poly))
        if total < 1e-10:
            return []

        xmin, ymin, xmax, ymax = _bbox(poly)
        W = xmax - xmin
        if W <= 0:
            return []

        AREA_TOL = 0.5  # cm^2 per spec
        target_piece = total / n

        # ---------- PRIMARY PLAN: vertical per-piece targeting ----------
        def try_x(xx: float) -> Optional[Tuple[float, float]]:
            chords = _vertical_chords(poly, xx)
            if not chords:
                return None

            # prefer longer chords; drop tiny sliver chords
            _, y_min, _, y_max = _bbox(poly)
            H = max(1e-9, y_max - y_min)
            MIN_SPAN = 0.005 * H  # 0.5% of height
            chords = [(yl, yh) for (yl, yh) in chords if (yh - yl) >= MIN_SPAN]
            if not chords:
                return None
            chords.sort(key=lambda ab: ab[1] - ab[0], reverse=True)

            # engine validation if available
            for yl, yh in chords:
                if _is_good_segment(self.cake, (xx, yl), (xx, yh)):
                    return (yl, yh)

            # shapely boundary-tolerant check
            ext = getattr(self.cake, "exterior_shape", None)
            if isinstance(ext, BaseGeometry):
                for yl, yh in chords:
                    mid = 0.5 * (yl + yh)
                    if ext.covers(Point(xx, mid)):  # covers accepts boundary
                        return (yl, yh)

            # last resort pick longest
            return chords[0] if chords else None

        cuts_primary: List[Tuple[Point, Point]] = []
        nudges = [
            0.0,
            0.001 * W,
            -0.001 * W,
            0.005 * W,
            -0.005 * W,
            0.01 * W,
            -0.01 * W,
            0.02 * W,
            -0.02 * W,
        ]
        x_left = xmin
        EPSX = 1e-6 * max(1.0, W)  # ensure monotone progress and avoid borders

        for _i in range(1, n):
            x0 = _find_next_cut_x(poly, target_piece, x_left, xmax)
            chosen_x: Optional[float] = None
            chosen: Optional[Tuple[float, float]] = None
            best: Optional[Tuple[float, float, Tuple[float, float]]] = None  # (err, xx, chord)

            for dx in nudges:
                xx = min(max(x0 + dx, x_left + EPSX), xmax - EPSX)
                res = try_x(xx)
                if res and _is_good_segment(self.cake, (xx, res[0]), (xx, res[1])):
                    piece_area = _area_between(poly, x_left, xx)
                    err = abs(piece_area - target_piece)
                    if err <= AREA_TOL:
                        chosen_x, chosen = xx, res
                        break
                    if best is None or err < best[0]:
                        best = (err, xx, res)

            if chosen is None and best is not None:
                _, chosen_x, chosen = best

            if chosen is None or chosen_x is None:
                break

            yl, yh = chosen
            cuts_primary.append((Point(chosen_x, yl), Point(chosen_x, yh)))
            x_left = chosen_x

        # score primary
        max_err_A, crust_rng_A, piece_count_A = _evaluate_plan(
            poly, cuts_primary, target_piece
        )

        # ---------- FALLBACK PLAN ----------
        cuts_fallback = _human_style_fallback(poly, n, self.cake)
        if cuts_fallback:
            max_err_B, crust_rng_B, piece_count_B = _evaluate_plan(
                poly, cuts_fallback, target_piece
            )
        else:
            max_err_B, crust_rng_B, piece_count_B = float("inf"), float("inf"), 0

        # ---------- CHOOSE BEST ----------
        score_A = (max_err_A, crust_rng_A, piece_count_A == n)
        score_B = (max_err_B, crust_rng_B, piece_count_B == n)

        # enforce piece count feasibility first
        if score_A[2] and not score_B[2]:
            return cuts_primary
        if score_B[2] and not score_A[2]:
            return cuts_fallback

        # both feasible or both not → compare (max_err, crust_rng)
        if (score_B[0], score_B[1]) < (score_A[0], score_A[1]):
            return cuts_fallback
        else:
            return cuts_primary
