from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split

from players.player import Player
from src.cake import Cake
import math

def get_final_piece_targets(cake: Cake, children: int) -> dict[str, float]:
	result = {}
 
	# When we accept a candidate final piece, if we search strictly for pieces larger than the requirement, we will fail to find a solution since the remaining area is smaller than required. To avoid this, we introduce a small tolerance to the minimum area required for each final piece. Practically, the minimum final piece size that we can accept for a candidate solution is related to the total number of children:
	# Let the target final size (determined by the number of children) be A, and our tolerance be e. Since our total size span tolerance is 0.5, if each candidate final piece other than the last one has area A - e, the last piece will have area A_n = A + (n - 1)e.
	# This means that the maximum size span, when all pieces are as small as possible except the last one, is:
	# size_span = A_n - (A - e) = (A + (n - 1)e) - (A - e) = ne
	# The practical goal here is to keep the size span less than 0.5, so we want ne < 0.5, or e < 0.5/n.
	target_area: float = cake.exterior_shape.area / float(children)
	area_epsilon: float = 0.5 / float(children)
	min_area: float = target_area - area_epsilon
	max_area: float = target_area + area_epsilon

	# Similarly, ratio target is 5%, meaning we can accept pieces with ratio +- 2.5%.
	target_ratio: float = cake.get_piece_ratio(cake.exterior_shape)
	ratio_epsilon: float = 0.05 / 2
	min_ratio: float = target_ratio - ratio_epsilon
	max_ratio: float = target_ratio + ratio_epsilon
	
	result["min_area"] = min_area
	result["max_area"] = max_area
	result["target_ratio"] = target_ratio
	result["min_ratio"] = min_ratio
	result["max_ratio"] = max_ratio
	return result

def piece_is_convex(piece: Polygon) -> bool:
    return piece.area / piece.convex_hull.area > 0.99

def make_cut(cake: Cake, piece: Polygon, from_p: Point, to_p: Point) -> tuple[Polygon, Polygon] | None:
	"""
	Attempt to cut the piece with a cut line from from_p to to_p.
	If successful, return the two resulting pieces as a tuple (piece1, piece2).
	If the cut is invalid (does not split the piece into two), return None.
	"""
	if not cake.cut_is_valid(from_p, to_p):
		return None

	cut_line = LineString([from_p, to_p])
	try:
		result = split(piece, cut_line)
		if len(result.geoms) != 2:
			return None
		piece1, piece2 = result.geoms
		if not piece1.is_valid or not piece2.is_valid:
			return None
		return piece1, piece2
	except Exception as e:
		# If splitting fails for any reason, return None
		return None