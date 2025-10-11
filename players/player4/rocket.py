from shapely.geometry import LineString, Point
from shapely.ops import split


def get_rocket_cuts(cake) -> list[tuple[Point, Point]]:
    poly = cake.exterior_shape
    total_area = poly.area
    target_area_per_band = total_area / 3  # 3 horizontal layers

    # compute area below a horizontal line
    def area_below_y(y):
        line = LineString([(-1e6, y), (1e6, y)])
        pieces = split(poly, line)
        if len(pieces.geoms) == 2:
            bottom, top = sorted(pieces.geoms, key=lambda g: g.centroid.y)
            return bottom.area
        elif len(pieces.geoms) == 1:
            return total_area if y >= poly.bounds[3] else 0
        return 0

    # binary search for equal-area y levels
    def find_y_for_target(target):
        lo, hi = poly.bounds[1], poly.bounds[3]
        for _ in range(50):
            mid = (lo + hi) / 2
            a = area_below_y(mid)
            if abs(a - target) < 0.05:
                return mid
            if a < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    y1 = find_y_for_target(target_area_per_band)
    y2 = find_y_for_target(2 * target_area_per_band)

    # snap those lines to polygon boundary
    def chord_within_poly(y):
        line = LineString([(-1e6, y), (1e6, y)])
        seg = poly.intersection(line)
        if seg.geom_type == "MultiLineString":
            seg = max(seg.geoms, key=lambda g: g.length)
        p1, p2 = list(seg.coords)
        return Point(*p1), Point(*p2)

    # vertical center cut
    v_line = LineString([(10, -1e6), (10, 1e6)])
    v_seg = poly.intersection(v_line)
    if v_seg.geom_type == "MultiLineString":
        v_seg = max(v_seg.geoms, key=lambda g: g.length)
    vp1, vp2 = list(v_seg.coords)
    vertical_cut = (Point(*vp1), Point(*vp2))

    # two horizontal equal-area chords
    h1_left, h1_right = chord_within_poly(y1)
    h2_left, h2_right = chord_within_poly(y2)

    #  make each horizontal stop at the vertical cut
    v_cross1 = Point(10, y1)
    v_cross2 = Point(10, y2)

    # total 5 cuts (1 vertical + 2 left halves + 2 right halves)
    cuts = [
        vertical_cut,  # main vertical
        (h1_left, v_cross1),  # lower-left horizontal
        (v_cross1, h1_right),  # lower-right horizontal
        (h2_left, v_cross2),  # upper-left horizontal
        (v_cross2, h2_right),  # upper-right horizontal
    ]
    return cuts
