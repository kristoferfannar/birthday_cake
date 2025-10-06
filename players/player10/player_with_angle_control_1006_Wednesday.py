from shapely.geometry import LineString, Point, Polygon
import math
import random
from statistics import stdev

from players.player import Player
from src.cake import Cake
from shapely.ops import split



class Player10(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None, num_angle_attempts: int = 360) -> None:
        super().__init__(children, cake, cake_path)
        # Binary search tolerance: area within 0.5 cm² of target
        self.target_area_tolerance = 0.005
        # Number of different angles to try (more attempts = better for complex shapes)
        self.num_angle_attempts = num_angle_attempts

    def find_line(self, position: float, piece: Polygon, angle: float):
        """Make a line at a given angle through a position that cuts the piece.
        
        Args:
            position: Position along the sweep direction (0 to 1)
            piece: The polygon piece to cut
            angle: Angle in degrees (0-360) where 0 is right, 90 is up
        """
        
        # Get bounding box of piece
        leftmost, lowest, rightmost, highest = piece.bounds
        width = rightmost - leftmost
        height = highest - lowest
        max_dim = max(width, height) * 2
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Calculate the perpendicular direction for the sweep
        sweep_angle = angle_rad + math.pi / 2
        
        # Start from center of bounding box
        center_x = (leftmost + rightmost) / 2
        center_y = (lowest + highest) / 2
        
        # Calculate sweep offset based on position (0 to 1)
        sweep_offset = (position - 0.5) * max_dim
        offset_x = sweep_offset * math.cos(sweep_angle)
        offset_y = sweep_offset * math.sin(sweep_angle)
        
        # Calculate point on the sweep line
        point_x = center_x + offset_x
        point_y = center_y + offset_y
        
        # Create a line at the given angle through this point
        dx = math.cos(angle_rad) * max_dim
        dy = math.sin(angle_rad) * max_dim
        
        # Create line extending in both directions
        start_point = (point_x - dx, point_y - dy)
        end_point = (point_x + dx, point_y + dy)
        cut_line = LineString([start_point, end_point])
        
        return cut_line
    
    def find_cuts(self, line: LineString, piece: Polygon): 
        """ This function finds the actual points where the cut line goes through cake"""
        intersection = line.intersection(piece.boundary)
        
        # What is the intersections geometry? - want it to be at least two points
        if intersection.is_empty or intersection.geom_type == 'Point':
            return None
        points = []
        if intersection.geom_type == 'MultiPoint':
            points = list(intersection.geoms)
        elif intersection.geom_type == 'LineString':
            coords = list(intersection.coords)
            points = [Point(c) for c in coords]

        # Need at least 2 points for a valid cut
        if len(points) < 2:
            return None

        # return the points where the sweeping line intersects with the cake
        return(points[0], points[1])
    
    def calculate_piece_area(self, piece: Polygon, position: float, angle: float):
        """ Determines the area of the pieces we cut.
        
        Args:
            piece: The polygon piece to cut
            position: Position along sweep direction (0 to 1)
            angle: Angle in degrees for the cutting line
        """
        line = self.find_line(position, piece, angle)
        pieces = split(piece, line)
        
        # we should get two pieces if not, line didn't cut properly
        if len(pieces.geoms) != 2:
            # if we're at the extremes
            if position <= 0.0:
                return 0.0
            elif position >= 1.0:
                return piece.area
            else:
                return 0.0
        
        piece1, piece2 = pieces.geoms
        
        # Calculate which piece is "first" based on sweep direction
        angle_rad = math.radians(angle)
        sweep_angle = angle_rad + math.pi / 2
        sweep_dir_x = math.cos(sweep_angle)
        sweep_dir_y = math.sin(sweep_angle)
        
        # Project centroids onto sweep direction
        centroid1 = piece1.centroid
        centroid2 = piece2.centroid
        
        proj1 = centroid1.x * sweep_dir_x + centroid1.y * sweep_dir_y
        proj2 = centroid2.x * sweep_dir_x + centroid2.y * sweep_dir_y
        
        # Return the area of the "first" piece in sweep direction
        if proj1 < proj2:
            return piece1.area
        else:
            return piece2.area


    def binary_search(self, piece: Polygon, target_area: float, angle: float):
        """Use binary search to find position along sweep that cuts off target_area.
        
        Args:
            piece: The polygon piece to cut
            target_area: The target area for the cut piece
            angle: Angle in degrees for the cutting line
        """
        
        left_pos = 0.0
        right_pos = 1.0
        best_pos = None
        best_error = float('inf')

        # try for best cut for 50 iterations
        for iteration in range(50):  
            # try middle first 
            mid_pos = (left_pos + right_pos) / 2

            # get the area of that prospective position
            cut_area = self.calculate_piece_area(piece, mid_pos, angle)

            if cut_area == 0:
                # Too far left, move right
                left_pos = mid_pos
                continue

            if cut_area >= piece.area:
                # Too far right, move left
                right_pos = mid_pos
                continue

            # how far away from the target value
            error = abs(cut_area - target_area)

            # Track best
            if error < best_error:
                best_error = error
                best_pos = mid_pos

            # Check if it's good enough
            if error < self.target_area_tolerance:
                return mid_pos

            # Adjust search based on distance from target area
            if cut_area < target_area:
                left_pos = mid_pos  # Need more, move right
            else:
                right_pos = mid_pos  # Too much, move left

        return best_pos

    def try_cutting_at_angle(self, angle: float, verbose: bool = False) -> tuple[list[tuple[Point, Point]], float] | None:
        """Try cutting the cake at a specific angle and return cuts with their score.
        
        Args:
            angle: Angle in degrees (0-360) for cutting direction
            verbose: Whether to print debug information
            
        Returns:
            Tuple of (cuts, variance_score) if successful, None if failed
            Lower variance_score is better
        """
        if verbose:
            print(f"\n  Trying angle {angle:.1f} degrees...")
        
        target_area = self.cake.get_area() / self.children
        cuts = []
        cake_copy = self.cake.copy()
        
        # Try to make all n-1 cuts
        for cut_idx in range(self.children - 1):
            current_pieces = cake_copy.get_pieces()
            # Always cut the biggest piece
            cutting_piece = max(current_pieces, key=lambda pc: pc.area)
            
            if verbose:
                print(f"    Cut {cut_idx + 1}: Cutting piece with area {cutting_piece.area:.2f}, target piece {target_area:.2f}")
            
            # Find the best position using binary search
            position = self.binary_search(cutting_piece, target_area, angle)
            
            # If we can't find a position, this angle doesn't work
            if position is None:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Failed to find position")
                return None
            
            # Find the actual cut points
            cut_line = self.find_line(position, cutting_piece, angle)
            cut_points = self.find_cuts(cut_line, cutting_piece)
            
            if cut_points is None:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Failed to find cut points")
                return None
            
            from_p, to_p = cut_points
            
            # Check if the cut is valid
            is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
            if not is_valid:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Invalid - {why}")
                return None
            
            # Try to make the cut
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))
                
                # Check the resulting piece sizes
                if verbose:
                    new_pieces = cake_copy.get_pieces()
                    areas = [p.area for p in new_pieces]
                    print(f"      -> Resulted in areas: {[f'{a:.2f}' for a in sorted(areas)]}")
                    
            except Exception as e:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Exception - {e}")
                return None
        
        # Check if we got the right number of pieces
        if len(cake_copy.get_pieces()) != self.children:
            if verbose:
                print(f"    Failed: Got {len(cake_copy.get_pieces())} pieces, expected {self.children}")
            return None
        
        # Check piece size consistency
        areas = [p.area for p in cake_copy.get_pieces()]
        size_span = max(areas) - min(areas)
        
        # Per project spec: pieces within 0.5 cm² are considered same size
        # But for the sweeping algorithm, we need some tolerance
        # Use a reasonable threshold based on number of children
        max_acceptable_span = max(5.0, target_area * 0.15)  # At least 5 cm² or 15% of target
        
        if size_span > max_acceptable_span:
            if verbose:
                print(f"    Failed: Size span {size_span:.2f} is too large (>{max_acceptable_span:.2f})")
            return None
        
        # Calculate crust ratio variance (our score to minimize)
        ratios = cake_copy.get_piece_ratios()
        
        # Check if all pieces are valid (have reasonable ratios)
        if any(r < 0 or r > 1 for r in ratios):
            if verbose:
                print(f"    Failed: Invalid ratios {ratios}")
            return None
        
        # Calculate variance in ratios (lower is better)
        if len(ratios) > 1:
            variance = stdev(ratios)
        else:
            variance = 0.0
        
        if verbose:
            print(f"    Success! Variance: {variance:.4f}, Size span: {size_span:.2f}")
            print(f"    Areas: {[f'{a:.2f}' for a in sorted(areas)]}")
            print(f"    Ratios: {[f'{r:.3f}' for r in ratios]}")
        
        return (cuts, variance)

    def find_best_cut_at_angle(self, cutting_piece: Polygon, target_area: float, angle: float, cake_copy: Cake, is_last_cut: bool = False) -> tuple[Point, Point, float, float] | None:
        """Try cutting at a specific angle and return the cut points and resulting piece info.
        
        Args:
            cutting_piece: The piece to cut
            target_area: Target area for the cut piece
            angle: Angle to try
            cake_copy: Current state of the cake
            is_last_cut: Whether this is the last cut (more lenient validation)
            
        Returns:
            Tuple of (from_point, to_point, crust_ratio_of_new_piece, area_of_new_piece) or None if invalid
        """
        # Find the best position using binary search
        position = self.binary_search(cutting_piece, target_area, angle)
        
        if position is None:
            return None
        
        # Find the actual cut points
        cut_line = self.find_line(position, cutting_piece, angle)
        cut_points = self.find_cuts(cut_line, cutting_piece)
        
        if cut_points is None:
            return None
        
        from_p, to_p = cut_points
        
        # Check if the cut is valid
        is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
        if not is_valid:
            return None
        
        # Simulate the cut to get the new piece and its crust ratio
        test_cake = cake_copy.copy()
        try:
            # Find which piece gets created
            pieces_before = [p.area for p in test_cake.get_pieces()]
            test_cake.cut(from_p, to_p)
            pieces_after = test_cake.get_pieces()
            
            # Find the new piece (smallest one, as we're cutting off target_area)
            new_piece = min(pieces_after, key=lambda p: p.area)
            new_piece_ratio = test_cake.get_piece_ratio(new_piece)
            new_piece_area = new_piece.area
            
            # Validate the new piece size is reasonable
            # More lenient for last cut, or when cutting small pieces
            if is_last_cut:
                tolerance = 0.5  # Very lenient for last cut
            elif cutting_piece.area < target_area * 2:
                tolerance = 0.4  # Lenient for small pieces
            else:
                tolerance = 0.3  # Standard tolerance
            
            if abs(new_piece_area - target_area) > target_area * tolerance:
                return None
            
            return (from_p, to_p, new_piece_ratio, new_piece_area)
            
        except Exception:
            return None

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - for each cut, try multiple angles and pick the best one"""
        print(f"__________Cutting for {self.children} children_______")
        
        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        print(f"TARGET AREA: {target_area:.2f} cm²")
        print(f"TARGET CRUST RATIO: {target_ratio:.3f}\n")
        
        cuts = []
        cake_copy = self.cake.copy()
        
        # For each cut, try multiple angles
        for cut_idx in range(self.children - 1):
            print(f"=== Cut {cut_idx + 1}/{self.children - 1} ===")
            
            current_pieces = cake_copy.get_pieces()
            cutting_piece = max(current_pieces, key=lambda pc: pc.area)
            
            print(f"Cutting piece area: {cutting_piece.area:.2f}")
            
            # For the last cut, we just need any valid cut that splits the piece
            # (both resulting pieces will be close to target since we've been careful)
            is_last_cut = (cut_idx == self.children - 2)
            
            best_cut = None
            best_score = float('inf')
            best_angle = None
            
            # Try multiple angles for THIS cut
            # Use more attempts for the last cut as it's often harder
            num_attempts = self.num_angle_attempts * 2 if is_last_cut else self.num_angle_attempts
            valid_attempts = 0
            
            for attempt in range(num_attempts):
                angle = random.uniform(0, 180)
                
                result = self.find_best_cut_at_angle(cutting_piece, target_area, angle, cake_copy, is_last_cut)
                
                if result is not None:
                    from_p, to_p, piece_ratio, piece_area = result
                    
                    # Calculate balanced score: PRIORITIZE piece size (primary), then crust ratio (secondary)
                    size_error = abs(piece_area - target_area) / target_area  # Normalized size error [0, 1]
                    ratio_error = abs(piece_ratio - target_ratio)  # Ratio error [0, 1]
                    
                    # Weighted score: SIZE is much more important (weight=3), ratio is secondary (weight=1)
                    score = size_error * 3.0 + ratio_error * 1.0
                    
                    valid_attempts += 1
                    
                    if score < best_score:
                        best_score = score
                        best_cut = (from_p, to_p)
                        best_angle = angle
                        best_ratio = piece_ratio
                        best_size = piece_area
            
            if best_cut is None:
                print(f"  Failed: No valid cut found after {num_attempts} attempts ({valid_attempts} were valid)")
                # Try to continue with a simple vertical cut as fallback
                continue
            
            # Apply the best cut found
            from_p, to_p = best_cut
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))
                
                areas = [p.area for p in cake_copy.get_pieces()]
                size_error = abs(best_size - target_area)
                ratio_error = abs(best_ratio - target_ratio)
                print(f"  Best angle: {best_angle:.1f}° (tried {valid_attempts} valid angles)")
                print(f"  Piece: size={best_size:.2f} (target={target_area:.2f}, err={size_error:.2f}), ratio={best_ratio:.3f} (target={target_ratio:.3f}, err={ratio_error:.3f})")
                print(f"  Current areas: {[f'{a:.2f}' for a in sorted(areas)]}\n")
                
            except Exception as e:
                print(f"  Error applying cut: {e}")
                break
        
        # Final summary
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: {len(cuts)}/{self.children-1} cuts completed")
        
        if len(cake_copy.get_pieces()) == self.children:
            areas = [p.area for p in cake_copy.get_pieces()]
            ratios = cake_copy.get_piece_ratios()
            
            print(f"\nPiece areas: {[f'{a:.2f}' for a in sorted(areas)]}")
            print(f"  Min: {min(areas):.2f}, Max: {max(areas):.2f}, Span: {max(areas) - min(areas):.2f}")
            
            print(f"\nCrust ratios: {[f'{r:.3f}' for r in ratios]}")
            if len(ratios) > 1:
                ratio_variance = stdev(ratios)
                print(f"  Variance: {ratio_variance:.4f}")
                print(f"  Min: {min(ratios):.3f}, Max: {max(ratios):.3f}, Span: {max(ratios) - min(ratios):.3f}")
        
        print(f"{'='*50}\n")
        
        return cuts
