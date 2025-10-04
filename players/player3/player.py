from shapely import Point

from players.player import Player
from src.cake import Cake


class Player3(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        return []

       #print(get_perimeter_points(self.cake, 15))
        #print(get_valid_cut_points(Point(1, 1), self.cake, 15))
def get_perimeter_points(cake, num_samples: int):
    # get the current largest polygon piece (the cake body)
    largest_piece = max(cake.get_pieces(), key=lambda piece: piece.area)

    boundary = largest_piece.exterior
    points = [
        boundary.interpolate(i / num_samples, normalized=True)
        for i in range(num_samples)
    ]
    return [(p.x, p.y) for p in points]

def get_valid_cut_points(start_point: Point, cake: Cake, num_samples: int):
    valid_points = []

    perimeter = get_perimeter_points(cake, num_samples)

    candidate_points = [Point(xy[0], xy[1]) for xy in perimeter]

    for end_point in candidate_points:  
        if end_point.equals(start_point):
            continue  # Skip if the end point is the same as the start point
        is_valid, _ = cake.cut_is_valid(start_point, end_point)
        if is_valid:
            valid_points.append(end_point)
    
    return valid_points