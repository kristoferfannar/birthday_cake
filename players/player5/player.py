from shapely import LineString, Point, intersection
from random import shuffle                         
from players.player import Player, PlayerException  
from src.cake import Cake                           
import math                                         

class Player5(Player):                              # Define a new player strategy subclass of Player
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path) # Initialize parent Player with given parameters
        print(f"I am {self}")                       # Print player identity for debugging

    def find_random_cut(self) -> tuple[Point, Point]:
        """Find a random cut.

        Algorithm:
        1. Find the largest piece of cake
        2. Find the lines outlining that piece
        3. Find two random lines whose centroids make
           a line that will cut the piece into two
        """
        largest_piece = max(self.cake.get_pieces(), key=lambda piece: piece.area)
        # Get all current cake pieces and select the one with the largest area

        vertices = list(largest_piece.exterior.coords[:-1])
        # Extract the vertices (outer boundary coordinates) of that polygon, excluding duplicate last point

        lines = [
            LineString([vertices[i], vertices[i + 1]]) for i in range(len(vertices) - 1)
        ]
        # Construct line segments between each pair of consecutive vertices

        from_p = largest_piece.centroid
        # Choose the centroid (geometric center) of the largest piece as the first cut endpoint

        vertices = list(largest_piece.exterior.coords[:-1])
        # Re-extract vertices (redundant but ensures current list is used)

        farthest_vertex = max(vertices, key=lambda v: from_p.distance(Point(v)))
        # Find the vertex farthest from the centroid

        to_p = Point(farthest_vertex)
        # Use that farthest vertex as the second endpoint for the cut

        is_valid, _ = self.cake.cut_is_valid(from_p, to_p)
        # Check whether the cut defined by these two points is valid (i.e., divides the cake properly)

        if is_valid:
            return from_p, to_p
        # If valid, return the cut

        lines = [LineString([vertices[i], vertices[(i + 1) % len(vertices)]]) for i in range(len(vertices))]
        # Otherwise, build all edges again, ensuring wrap-around to close polygon

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                from_p = lines[i].centroid
                to_p = lines[j].centroid
                # Use centroids of two different edges as potential cut endpoints

                is_valid, _ = self.cake.cut_is_valid(from_p, to_p)
                # Validate this new possible cut

                if is_valid:
                    return from_p, to_p
                # If valid, return immediately

        raise PlayerException("could not find valid cut")
        # If no valid cut was found after checking all combinations, raise an exception

    # Assumptions: Area never decreases, always two intersections.
    # Basically has to be a convex polygon
    # If it does not work, use previous algorithm
    def scan_cut(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []
        # Define moves at the start, but do not cut continuously.
        remaining_children = self.children
        # Only one piece (the whole cake)
        vertices = list(self.cake.get_pieces()[0].exterior.coords[:-1])
        print(vertices)

        # Consts
        lowest_y = min(y for _, y in vertices)
        lowest_x = min(x for x, _ in vertices)
        highest_y = max(y for _, y in vertices)
        highest_x = max(x for x, _ in vertices)
        step_size = (highest_y - lowest_y) / 200

        # Mut
        current_y = lowest_y

        # Verification
        print(lowest_x, lowest_y)
        print(highest_x, highest_y)
        print(step_size)

        lines = [
            LineString([vertices[i], vertices[(i + 1) % len(vertices)]])
            for i in range(len(vertices))
        ]
        tracked_area = 0.0
        while current_y < highest_y:
            current_y += step_size
            from_p = Point(lowest_x, current_y)
            to_p = Point(highest_x, current_y)

            cut = LineString([from_p, to_p])
            intersections = []
            for edge in lines:
                inter = cut.intersection(edge)
                if not inter.is_empty:
                    intersections.append(inter)
            assert(len(intersections) == 2)
            from_p = intersections[0]
            to_p = intersections[1]
            cpy_cake = self.cake.copy()
            try:
                cpy_cake.cut(from_p, to_p)
                smallest_piece = min(cpy_cake.get_pieces(), key=lambda piece: piece.area)
                area = smallest_piece.area
                if area < tracked_area:
                    moves.append((from_p, to_p))
                    self.cake.cut(from_p, to_p)
                    break
                tracked_area = smallest_piece.area
                if smallest_piece.area > 20.14:
                    moves.append((from_p, to_p))
                    #print(from_p, to_p)
                    self.cake.cut(from_p, to_p)
                    remaining_children -= 1
                    tracked_area = 0
                else:
                    continue
            except Exception as e:
                pass
        #print(moves)
        return moves
            
            
    
    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves = self.scan_cut()

        return moves
        # Return list of all cuts made