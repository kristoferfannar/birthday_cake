from shapely import LineString, Point        
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

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []
        # Initialize list to store all cuts

        for _ in range(self.children - 1):
            # Each child (except last) needs a cut to divide cake into correct number of pieces

            from_p, to_p = self.find_random_cut()
            # Find a valid cut using method above

            moves.append((from_p, to_p))
            # Record the cut in the moves list

            self.cake.cut(from_p, to_p)
            # Apply the cut to update the internal cake state

        return moves
        # Return list of all cuts made