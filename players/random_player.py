from shapely import LineString, Point
from random import shuffle

from players.player import Player, PlayerException
from src.cake import Cake


class RandomPlayer(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        print(f"I am {self}")

    def find_random_cut(self) -> tuple[Point, Point]:
        """Find a random cut.

        Algorithm:
        1. Find the largest piece of cake
        2. Find the lines outlining that piece
        3. Find two random lines whos centroids make
           a line that will cut the piece into two
        """
        largest_piece = max(self.cake.get_pieces(), key=lambda piece: piece.area)
        vertices = list(largest_piece.exterior.coords[:-1])
        lines = [
            LineString([vertices[i], vertices[i + 1]]) for i in range(len(vertices) - 1)
        ]

        # introduce some sort of randomness to the player
        idxs = list(range(len(lines)))
        shuffle(idxs)

        for i in idxs:
            jdxs = list(range(i + 1, len(lines)))
            shuffle(jdxs)
            for j in jdxs:
                from_p = lines[i].centroid
                to_p = lines[j].centroid

                is_valid, _ = self.cake.cut_is_valid(from_p, to_p)
                if is_valid:
                    return (from_p, to_p)

        raise PlayerException("could not find random move :(")

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        for _ in range(self.children - 1):
            from_p, to_p = self.find_random_cut()
            moves.append((from_p, to_p))

            # simulate cut on our cake to ensure we have a
            # valid representation of our current environment
            self.cake.cut(from_p, to_p)

        return moves
