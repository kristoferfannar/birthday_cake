from shapely.geometry import LineString, Point

def get_hilbert_cuts(self) -> list[tuple[Point,Point]]:
    cut_lines = [
        LineString([(5, 0), (5, 5)]),
        LineString([(10, 0), (10, 5)]),
        LineString([(10, 5), (15, 5)]),
        LineString([(10, 10), (15, 10)]),
        LineString([(5, 10), (5, 15)]),
        LineString([(10, 10), (10, 15)]),
        LineString([(0, 15), (5, 15)]),
        LineString([(0, 20), (5, 20)]),
        LineString([(0, 25), (5, 25)]),
        LineString([(0, 30), (5, 30)]),
        LineString([(5, 30), (5, 35)]),
        LineString([(10, 30), (10, 35)]),
        LineString([(10, 30), (15, 30)]),
        LineString([(10, 25), (15, 25)]),
        LineString([(15, 20), (15, 25)]),
        LineString([(20, 20), (20, 25)]),
        LineString([(20, 25), (25, 25)]),
        LineString([(20, 30), (25, 30)]),
        LineString([(25, 30), (25, 35)]),
        LineString([(30, 30), (30, 35)]),
        LineString([(30, 30), (35, 30)]),
        LineString([(30, 25), (35, 25)]),
        LineString([(30, 20), (35, 20)]),
        LineString([(30, 15), (35, 15)]),
        LineString([(30, 10), (30, 15)]),
        LineString([(25, 10), (25, 15)]),
        LineString([(20, 10), (25, 10)]),
        LineString([(20, 5), (25, 5)]),
        LineString([(25, 0), (25, 5)]),
        LineString([(30, 0), (30, 5)])
    ]
    cuts = []
    for line in cut_lines:
        p1, p2 = list(line.coords)
        cuts.append((Point(*p1), Point(*p2)))
    return cuts