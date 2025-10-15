from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, split

from src.cake import Cake
import src.constants as c


def get_mushroom_cuts(children, cake) -> list[tuple[Point, Point]]:
    cuts: list[tuple[Point, Point]] = []
    assert children == 6, "Pre-set mushroom cake can only serve 6 children!"

    ideal_area_per_piece: float = cake.get_area() / children
    min_area_per_piece: float = ideal_area_per_piece - (c.PIECE_SPAN_TOL * 0.5)
    max_area_per_piece: float = ideal_area_per_piece + (c.PIECE_SPAN_TOL * 0.5)

    cake_crust: LineString = cake.exterior_shape.boundary
    cake_centroid = cake.exterior_shape.centroid
    minx, miny, maxx, maxy = _get_bounds(cake)

    # first cut vertically down the middle
    first_v_cut = LineString([(cake_centroid.x, miny), (cake_centroid.x, maxy)])
    v_cut_p1, v_cut_p2 = Point(first_v_cut.coords[0]), Point(first_v_cut.coords[1])
    cuts.append((v_cut_p1, v_cut_p2))

    # Get the two halves
    split_halves = split(cake.exterior_shape, first_v_cut)

    # Cut the halves in pieces with good ratio and area
    cuts_per_halve = children // 2
    center_upper_p = Point(cake_centroid.x, 0)

    for i, piece in enumerate(split_halves.geoms):
        # Get crust of current half
        current_crust = linemerge(cake_crust.intersection(piece.boundary))
        if not center_upper_p.equals(Point(current_crust.coords[0])):
            current_crust = LineString(list(current_crust.coords)[::-1])

        crust_len_per_piece = current_crust.length / cuts_per_halve
        current_piece = piece
        for cut in range(1, cuts_per_halve):
            from_p = current_crust.interpolate(crust_len_per_piece * cut)
            from_p = current_crust.interpolate(current_crust.project(from_p))
            to_p = cake_centroid

            def adjust_to_fit_area(piece, from_p, to_p):
                max_steps = 100
                step = 0.1
                for _ in range(max_steps):
                    test_cut = LineString([to_p, from_p])
                    split_pieces = split(piece, test_cut)
                    top_piece, bottom_piece = sorted(
                        split_pieces.geoms, key=lambda p: p.centroid.y
                    )

                    # Check if area of top piece is within bounds and adjust to_p accordingly
                    if min_area_per_piece <= top_piece.area <= max_area_per_piece:
                        return to_p, top_piece, bottom_piece
                    if top_piece.area > max_area_per_piece:
                        to_p = Point(to_p.x, to_p.y - step)
                    else:
                        to_p = Point(to_p.x, to_p.y + step)

                return to_p, top_piece, bottom_piece

            to_p, _, bottom_piece = adjust_to_fit_area(current_piece, from_p, to_p)
            cuts.append((from_p, to_p))
            if bottom_piece is not None:
                current_piece = bottom_piece

    return cuts


def _get_bounds(current_cake: Polygon) -> tuple[float, float, float, float]:
    cake_boundary_points = (
        current_cake.get_boundary_points()
        if isinstance(current_cake, Cake)
        else [Point(co) for co in current_cake.boundary.coords]
    )
    min_x = min(point.x for point in cake_boundary_points)
    min_y = min(point.y for point in cake_boundary_points)
    max_x = max(point.x for point in cake_boundary_points)
    max_y = max(point.y for point in cake_boundary_points)
    return min_x, min_y, max_x, max_y
