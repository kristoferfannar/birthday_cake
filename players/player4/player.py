from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split

from players.player import Player
from src.cake import Cake
import math

import time

N_SWEEP_DIRECTIONS = 180
SWEEP_STEP_DISTANCE = 0.2 # cm
N_BEST_CUT_CANDIDATES_TO_TRY = 50

class Player4(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        
        # When we accept a candidate final piece, if we search strictly for pieces larger than the requirement, we will fail to find a solution since the remaining area is smaller than required. To avoid this, we introduce a small tolerance to the minimum area required for each final piece. Practically, the minimum final piece size that we can accept for a candidate solution is related to the total number of children:
        # Let the target final size (determined by the number of children) be A, and our tolerance be e. Since our total size span tolerance is 0.5, if each candidate final piece other than the last one has area A - e, the last piece will have area A_n = A + (n - 1)e.
        # This means that the maximum size span, when all pieces are as small as possible except the last one, is:
        # size_span = A_n - (A - e) = (A + (n - 1)e) - (A - e) = ne
        # The practical goal here is to keep the size span less than 0.5, so we want ne < 0.5, or e < 0.5/n.
        self.final_piece_target_area = cake.exterior_shape.area / float(children)
        self.final_piece_size_epsilon = 0.5 / float(children)
        self.final_piece_min_area = self.final_piece_target_area - self.final_piece_size_epsilon
        self.final_piece_max_area = self.final_piece_target_area + self.final_piece_size_epsilon
        print(f"Player 4: Final piece target area: {self.final_piece_target_area}, epsilon min area: {self.final_piece_min_area}.")
        
        # Similarly, ratio target is 5%, meaning we can accept pieces with ratio +- 2.5%.
        self.final_piece_target_ratio = cake.get_piece_ratio(cake.exterior_shape)
        self.final_piece_min_ratio = self.final_piece_target_ratio - 0.05
        self.final_piece_max_ratio = self.final_piece_target_ratio + 0.05
        print(f"Player 4: Final piece target ratio: {self.final_piece_target_ratio}, min ratio: {self.final_piece_min_ratio}, max ratio: {self.final_piece_max_ratio}.")

    def get_cuts(self) -> list[tuple[Point, Point]]:
        piece: Polygon = self.cake.exterior_shape
        print(f"Player 4: Starting search for {self.children} children...")
        self.start_time = time.time()
        result = self.DFS(piece, self.children)
        self.end_time = time.time()
        print(f"Player 4: Search complete, took {self.end_time - self.start_time} seconds.")
        return result

    def DFS(self, piece: Polygon, children: int) -> list[tuple[Point, Point]]:
        print(f"DFS solving for {children} children remaining...")
        cut_sequence: list[tuple[Point, Point]] = []
        if children == 1:
            return cut_sequence
        
        possible_cuts: list[tuple[Point, Point]] = self.get_all_possible_cuts(piece)
        top_N_valid_cuts: list[tuple[Point, Point]] = self.filter_cuts(piece, children, possible_cuts)
        if not top_N_valid_cuts:
            return None
        
        cut_to_try: tuple[Point, Point]
        while top_N_valid_cuts:
            # cut_to_try = top_N_valid_cuts[random.randint(0, len(top_N_valid_cuts) - 1)]
            cut_to_try = top_N_valid_cuts[0]
            top_N_valid_cuts.remove(cut_to_try)
            
            from_p = cut_to_try[0]
            to_p = cut_to_try[1]
            line = LineString([from_p, to_p])
            split_pieces = split(piece, line)
            if len(split_pieces.geoms) != 2:
                continue
            subpiece1, subpiece2 = split_pieces.geoms

            # Central DFS idea: assign each subpiece to have (k, children - k) final pieces and try
            for k in range(1, children):
                subpiece1_children = k
                subpiece2_children = children - k
                subpiece1_min_area = self.final_piece_min_area * subpiece1_children
                subpiece2_min_area = self.final_piece_min_area * subpiece2_children

                # Is each subpiece big enough to share with its assigned children?
                if subpiece1.area < subpiece1_min_area or subpiece2.area < subpiece2_min_area:
                    continue
                # Is each subpiece too big to share with its assigned children?
                if subpiece1.area > self.final_piece_max_area * subpiece1_children or subpiece2.area > self.final_piece_max_area * subpiece2_children:
                    continue
                
                # DFS on each subpiece
                cuts1 = []
                if subpiece1_children > 1:
                    cuts1 = self.DFS(subpiece1, subpiece1_children)
                    if cuts1 is None:
                        continue
                cuts2 = []
                if subpiece2_children > 1:
                    cuts2 = self.DFS(subpiece2, subpiece2_children)
                    if cuts2 is None:
                        continue
                
                # Success! Found k where both subpieces can be cut properly
                cut_sequence = [(from_p, to_p)] + cuts1 + cuts2
                return cut_sequence
        return None

    def get_all_possible_cuts(self, piece: Polygon) -> list[tuple[Point, Point]]:
        """
        This method generates a list of all possible valid cuts for a given polygon piece. A possible valid cut is a pair of two points on the piece's perimeter, where the line segment connecting these points lies entirely within the piece.
        """
        candidate_cuts: list[tuple[Point, Point]] = []
        
        # Collect all boundary coordinates once
        coords = list(piece.exterior.coords)
        if piece.interiors:
            for interior in piece.interiors:
                coords.extend(list(interior.coords))
        
        # Sweep at different angles (0° to 179°)
        for angle_deg in range(0, 180, 180 // N_SWEEP_DIRECTIONS):
            angle_rad = math.radians(angle_deg)
            
            # Normal vector (direction we sweep along)
            normal_x = math.cos(angle_rad)
            normal_y = math.sin(angle_rad)
            
            # Line direction (perpendicular to normal)
            line_dx = -math.sin(angle_rad)
            line_dy = math.cos(angle_rad)
            
            # Project all boundary points onto the normal axis to find sweep range
            projections = [x * normal_x + y * normal_y for x, y in coords]
            min_proj = min(projections)
            max_proj = max(projections)
            
            # Project centroid onto normal to get center point
            centroid_proj = piece.centroid.x * normal_x + piece.centroid.y * normal_y
            # Calculate step size
            max_distance = max(abs(max_proj - centroid_proj), abs(min_proj - centroid_proj))
            num_sweeps = max(1, int(max_distance / SWEEP_STEP_DISTANCE) * 2 + 1)  # *2 for both directions, +1 for center
       
            
            # for i in range(N_SWEEPS_PER_DIRECTION):
            for i in range(num_sweeps):
                if i == 0:
                    offset = 0
                elif i % 2 == 1:  # odd indices go positive
                    offset = ((i + 1) // 2) * SWEEP_STEP_DISTANCE
                else:  # even indices go negative
                    offset = -((i // 2)) * SWEEP_STEP_DISTANCE
                
                proj = centroid_proj + offset
                
                # Skip if we've gone beyond the bounds
                if proj < min_proj or proj > max_proj:
                    continue
                
                # Create a line at this projection
                center_x = proj * normal_x
                center_y = proj * normal_y
                
                # Extend the line far enough to cover the polygon
                extension = 1000
                p1 = (center_x - extension * line_dx, center_y - extension * line_dy)
                p2 = (center_x + extension * line_dx, center_y + extension * line_dy)
                
                sweep_line = LineString([p1, p2])
                intersection = sweep_line.intersection(piece.boundary)
                
                if intersection.is_empty:
                    continue
                
                # Extract points from the intersection
                points: list[Point] = []
                if intersection.geom_type == 'Point':
                    points = [intersection]
                elif intersection.geom_type == 'MultiPoint':
                    points = list(intersection.geoms)
                elif intersection.geom_type == 'GeometryCollection':
                    # Extract only Point geometries from the collection
                    points = [geom for geom in intersection.geoms if geom.geom_type == 'Point']
                
                # We need at least 2 points to make a cut
                if len(points) >= 2:
                    # Check all pairs of intersection points
                    for j in range(len(points)):
                        for k in range(j + 1, len(points)):
                            from_p: Point = points[j]
                            to_p: Point = points[k]
                            line = LineString([from_p, to_p])
                            try:
                                split_result = split(piece, line)
                                if len(split_result.geoms) != 2:
                                    continue
                            except Exception as e:
                                continue

                            candidate_cuts.append((from_p, to_p))
        return candidate_cuts

    def filter_cuts(self, piece: Polygon, children: int, cuts: list[tuple[Point, Point]]) -> list[tuple[Point, Point]]:
        """
        This method filters a list of cuts and returns a subset (or none) of them that are valid for the given piece and number of children.
        """
        valid_cuts: list[tuple[Point, Point]] = []
        valid_cuts_scores : dict[tuple[Point, Point], dict] = {}
        
        filter_reject_reasons = {"piece_count": 0, "ratio": 0, "area": 0, "suboptimal": 0}
        for from_p, to_p in cuts:
            line = LineString([from_p, to_p])
            split_pieces = split(piece, line)
            if len(split_pieces.geoms) != 2:
                filter_reject_reasons["piece_count"] += 1
                continue
            
            piece1, piece2 = split_pieces.geoms
            piece1_area = piece1.area
            piece2_area = piece2.area
            piece1_ratio = self.cake.get_piece_ratio(piece1)
            piece2_ratio = self.cake.get_piece_ratio(piece2)
            
            # # Keep only cuts where both subpieces have acceptable ratio
            # if piece1_ratio < self.final_piece_min_ratio or piece2_ratio < self.final_piece_min_ratio:
            #     # print(f"Skipped cut with bad ratio: {piece1_ratio}, {piece2_ratio}.")
            #     filter_reject_reasons["ratio"] += 1
            #     continue
            # if piece1_ratio > self.final_piece_max_ratio or piece2_ratio > self.final_piece_max_ratio:
            #     # print(f"Skipped cut with bad ratio: {piece1_ratio}, {piece2_ratio}.")
            #     filter_reject_reasons["ratio"] += 1
            #     continue
            # Keep only cuts where both subpieces are bigger than one final piece in area
            if piece1_area < self.final_piece_min_area or piece2_area < self.final_piece_min_area:
                # print(f"Skipped cut with insufficient area: {piece1_area}, {piece2_area}.")
                filter_reject_reasons["area"] += 1
                continue

            valid_cuts.append((from_p, to_p))
            valid_cuts_scores[(from_p, to_p)] = {
                "area1": piece1_area,
                "area2": piece2_area,
                "ratio1": piece1_ratio,
                "ratio2": piece2_ratio,
            }
            
        for cut, scores in valid_cuts_scores.items():
            # Extra area score
            smaller_subpiece_area = min(scores["area1"], scores["area2"])
            smaller_subpiece_area_max_children = int(smaller_subpiece_area // self.final_piece_min_area)
            smaller_subpiece_extra_area = smaller_subpiece_area - (smaller_subpiece_area_max_children * self.final_piece_min_area)
            scores["smaller_subpiece_extra_area"] = smaller_subpiece_extra_area
            
            # Ratio error score
            ratio1_error = abs(scores["ratio1"] - self.final_piece_target_ratio)
            ratio2_error = abs(scores["ratio2"] - self.final_piece_target_ratio)
            scores["ratio_error"] = ratio1_error + ratio2_error
            
            valid_cuts_scores[cut] = scores
        
        print(f"Filtered {len(cuts) - len(valid_cuts)} invalid cuts, {len(valid_cuts)} candidate cuts remaining.")
        for reason, count in filter_reject_reasons.items():
            print(f"\t{count} cuts rejected due to: {reason}")
            
        # Sort to search least wasteful cuts first
        # Sort by ratio error score (smaller is better)
        valid_cuts.sort(key=lambda cut: valid_cuts_scores[cut]["ratio_error"])
        top_N_valid_cuts = valid_cuts[:N_BEST_CUT_CANDIDATES_TO_TRY]
        print(f"Keeping top {len(top_N_valid_cuts)} candidate cuts after filtering.")
        
        # top_N_valid_cuts.sort(key=lambda cut: valid_cuts_scores[cut]["smaller_subpiece_extra_area"])
        return top_N_valid_cuts