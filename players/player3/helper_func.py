from shapely import Point, Polygon
from shapely.geometry import LineString
from src.cake import Cake

def get_perimeter_points(cake: Cake) -> list[Point]:
    
    largest_piece = max(cake.exterior_pieces, key=lambda p: p.area)
    vertices = list(largest_piece.exterior.coords)

    gen_points = []
    num_points = 50  # number of points to generate per edge

    # for each edge, generate num_points points
    for i in range(len(vertices) - 1):
        vert1 = vertices[i]
        vert2 = vertices[i + 1]

        # generate points along this edge
        for j in range(num_points):
            # how far along the edge we are (0 to 1)
            spacing = j / (num_points - 1) 

            # move x and y by spacing along the edge form
            x = vert1[0] + (vert2[0] - vert1[0]) * spacing
            y = vert1[1] + (vert2[1] - vert1[1]) * spacing

            gen_points.append(Point(x, y))

    # print("Generated perimeter points:")
    # print(all_points)
    return gen_points

def get_areas(cake: Cake, xy1: Point, xy2: Point, target_ratio: float = 0.5, acceptable_error: float = 0.5, original_area=None) -> tuple[bool, float | None]:
    # find cuts that produce pieces with target ratio
    # target_ratio = 1/24 means cut off 1/24 of the piece
    # acceptable_error is in cm^2 so 0.5 means 0.5 cm^2
    valid, _ = cake.cut_is_valid(xy1, xy2)

    if not valid:
        return False, None

    target_piece, _ = cake.get_cuttable_piece(xy1, xy2)

    if target_piece is None:
        return False, None

    split_pieces = cake.cut_piece(target_piece, xy1, xy2)

    areas = [piece.area for piece in split_pieces]

    # check the ratio with whole cake area
    target_area = original_area * target_ratio

    # check if one piece is the target area of 1/children
    if abs(areas[0] - target_area) <= acceptable_error or abs(areas[1] - target_area) <= acceptable_error:
        return True, min(abs(areas[0] - target_area), abs(areas[1] - target_area))

    return False, None

# probably can use binary search to make this faster and optimize our search route instead of n^2 time complexity going through each point
def find_valid_cuts(cake: Cake, perim_points: list[Point] | None = None, target_ratio: float = 0.5, original_area=None) -> list[tuple[Point, Point, Polygon]]:
    valid_cuts = []

    for piece in cake.exterior_pieces:

        # get points on this piece's perimeter
        piece_xy = [p for p in perim_points if cake.point_lies_on_piece_boundary(p, piece)]

        # try all pairs
        for i, xy1 in enumerate(piece_xy):
            for xy2 in piece_xy[i+1:]:
                if cake.cut_is_valid(xy1, xy2):
                    areas_valid, area_diff = get_areas(cake, xy1, xy2, target_ratio, original_area)

                    if areas_valid:
                        valid_cuts.append((xy1, xy2, piece, area_diff))

    # sort by almost to least accurate just in terms of area (future incorporate ratio)
    valid_cuts.sort(key=lambda x: x[3])

    return [(xy1, xy2, piece) for xy1, xy2, piece, _ in valid_cuts]