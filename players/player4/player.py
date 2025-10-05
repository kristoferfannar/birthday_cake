from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split

from players.player import Player
from src.cake import Cake


class Player4(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        piece: Polygon = self.cake.exterior_shape
        n = self.children
        total_area = piece.area

        cuts = []
        minx, miny, maxx, maxy = piece.bounds
        ysafety = (maxy - miny) * 2  # extend cut line enough

        for i in range(1, n):
            target = total_area * i / n
            x_cut = self.find_cut_x(piece, target, minx, maxx)

            # vertical line at x_cut
            line = LineString([(x_cut, miny - ysafety), (x_cut, maxy + ysafety)])
            intersection = piece.boundary.intersection(line)

            if intersection.geom_type == "MultiPoint":
                pts = list(intersection.geoms)
                if len(pts) >= 2:
                    cuts.append((pts[0], pts[1]))
            elif intersection.geom_type == "Point":
                cuts.append((intersection, intersection))
            else:
                # fallback: full vertical line
                cuts.append((Point(x_cut, miny), Point(x_cut, maxy)))

        print(f"[Player4] Returning {len(cuts)} cuts (expected {n-1})")
        return cuts

    def find_cut_x(self, poly: Polygon, target_area: float, minx: float, maxx: float) -> float:
        """Binary search for the x-coordinate where the polygon left of x has area = target_area."""
        lo, hi = minx, maxx
        tol = 1e-2
        max_iter = 50

        miny, maxy = poly.bounds[1], poly.bounds[3]
        ysafety = (maxy - miny) * 2

        def area_left_of_x(x): #find area of left polygon until it reaches target
            line = LineString([(x, miny - ysafety), (x, maxy + ysafety)])
            pieces = split(poly, line)
            if len(pieces.geoms) == 2:
                left, right = sorted(pieces.geoms, key=lambda g: g.centroid.x)
                return left.area
            elif len(pieces.geoms) == 1:
                if x <= minx:
                    return 0.0
                elif x >= maxx:
                    return poly.area
            return None

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            aL = area_left_of_x(mid)
            if aL is None:
                break
            if abs(aL - target_area) <= tol:
                return mid
            if aL < target_area:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2













