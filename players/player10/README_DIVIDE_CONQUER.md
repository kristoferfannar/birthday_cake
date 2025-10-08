# Divide-and-Conquer Cake Cutting Algorithm

## What Does This Do?

This algorithm cuts a cake into equal pieces by repeatedly dividing pieces until each piece is for exactly one child. Think of it like a smart version of "split the biggest piece" that explores different ways to split.

## The Big Idea

Instead of always cutting 1/10th of the cake, this algorithm can try splitting in different ways:
- Cut a 10-child piece into (5, 5) - split in half
- Cut a 10-child piece into (3, 7) - split unevenly
- Cut a 5-child piece into (2, 3) - whatever works best!

**The Goal:** Each piece ends up being the right size with a fair amount of crust.

## How It Works (Simple Version)

### Step 1: Track Each Piece
Each piece "knows" how many children it's for:
```
Start:     [whole cake → 10 children]
After cut: [piece A → 5 children, piece B → 5 children]
Continue:  [piece A1 → 2, piece A2 → 3, piece B → 5]
...
End:       [piece 1, piece 2, piece 3, ... piece 10] ✓
```

### Step 2: Pick a Piece to Cut
Always pick the piece that needs to serve the most children (or the biggest piece if tied).

### Step 3: Find the Best Way to Cut It (Two Phases)

**Phase 1: Explore Different Split Ratios** (first 180 tries)
- Try splitting in different ways: 1/5, 2/5, 3/5, etc.
- For each way, try different angles: 0°, 90°, 180°, 270°, plus random angles
- Remember which split ratio worked best

**Phase 2: Perfect the Best Split** (next 180 tries)
- Take the best split ratio from Phase 1
- Try LOTS of different angles with just that ratio
- Pick the absolute best angle

### Step 4: Make the Cut & Repeat
- Cut the cake
- Add the two new pieces to the queue
- Go back to Step 2 until all pieces are for 1 child

## Example with Pictures

### Cutting a Cake for 10 Children

```
Cut 1: 
┌─────────────────────────┐
│     Whole Cake          │  Split: 5/10 (cut in half)
│     (10 children)       │  Angle: 0° (horizontal)
│─────────────────────────│  Result: Two pieces of 5 children each
│                         │
└─────────────────────────┘

Cut 2:
┌─────────────┐  ┌─────────────┐
│  Piece A    │  │  Piece B    │  Pick Piece A (or B)
│ (5 children)│  │ (5 children)│  Split: 2/5
│─────────────│  │             │  Result: 2 and 3 children
│      │      │  │             │
└──────┴──────┘  └─────────────┘

... continues until 10 pieces ...
```

## Why Use Cardinal Angles?

Cardinal angles are just the simple ones: **0°, 90°, 180°, 270°**

We always try these because:
- They're the most natural cuts (horizontal and vertical)
- They often work best for regular shapes
- We never want to miss an obvious good cut

## Scoring System (How We Pick the Best Cut)

Every potential cut gets a score:
```
Score = (Size Error × 3) + (Crust Error × 1)
```

**Lower score = Better cut**

- **Size Error (×3)**: How far from target size? ← **Most important!**
- **Crust Error (×1)**: How unfair is the crust distribution? ← Secondary

Example:
- Cut A: Size perfect (0), Crust slightly off (0.2) → Score = 0×3 + 0.2×1 = **0.2** ✓ Better
- Cut B: Size slightly off (0.1), Crust perfect (0) → Score = 0.1×3 + 0×1 = **0.3**

## Settings

```python
NUMBER_ATTEMPTS = 360  # How many (ratio, angle) combinations to try per cut
```

**What this means:**
- Phase 1: Try 180 random (split ratio, angle) combinations
- Phase 2: Try 180 more angles with the best ratio
- More attempts = Better cuts but slower
- Fewer attempts = Faster but might miss good cuts

## Usage

```python
from players.player10.player_divide_conquer_1006 import Player10
from src.cake import read_cake

# Load your cake
cake = read_cake('cakes/rectangle.csv', 10, False)

# Create the player (uses 360 attempts by default)
player = Player10(10, cake, 'cakes/rectangle.csv')

# Get the cuts!
cuts = player.get_cuts()
```

Want faster? Use fewer attempts:
```python
player = Player10(10, cake, 'cakes/rectangle.csv', num_angle_attempts=200)
```

Want better quality? Use more attempts:
```python
player = Player10(10, cake, 'cakes/rectangle.csv', num_angle_attempts=500)
```

## Results

### Rectangle (10 pieces)
```
✓ All pieces: 36.10 cm² (span: 0.01 cm²) ← Nearly perfect!
✓ Crust variance: 0.0042 ← Very fair distribution
✓ Time: 2-3 seconds
```

### Star (10 pieces)  
```
✓ All pieces: 18.79-18.80 cm² (span: 0.01 cm²) ← Excellent!
✓ Crust variance: 0.0317 ← Good distribution
✓ Time: 3-4 seconds
```

### Figure Eight (10 pieces)
```
✓ All pieces: 29.90 cm² (span: 0.00 cm²) ← Perfect!
✓ Crust variance: 0.0818 ← Acceptable distribution
✓ Time: 3-4 seconds
```

## Key Advantages

1. **Flexible Splitting**: Can split pieces in many ways, not just 1/n
2. **Smart Exploration**: Tries different ratios first, then perfects the best one
3. **Never Misses Basics**: Always tries simple horizontal/vertical cuts
4. **Size First**: Prioritizes getting piece sizes right
5. **No Recursion**: Fast and predictable performance

## Comparison: This vs Simple Sequential

| What | Divide-and-Conquer | Sequential Cutting |
|------|-------------------|-------------------|
| How to split | Try many ways (1/n, 2/n, 3/n...) | Always cut 1/n |
| Tracks children per piece | Yes ✓ | No |
| Flexibility | High | Limited |
| Speed | Fast (~3 sec) | Fast (~3 sec) |
| Piece sizes | Excellent | Excellent |

## When Things Go Wrong

**Problem: It's too slow!**
- **Fix**: Use fewer attempts: `num_angle_attempts=200`

**Problem: Pieces aren't fair (crust-wise)**
- **Fix**: Use more attempts: `num_angle_attempts=500`

**Problem: Some cuts fail**
- **Check**: Make sure your cake shape is valid (no self-intersections)

## Quick Reference

```python
# Default (good balance)
Player10(10, cake, path)  # 360 attempts

# Fast mode
Player10(10, cake, path, num_angle_attempts=200)

# High quality mode
Player10(10, cake, path, num_angle_attempts=500)
```

## The Math (For Those Who Care)

- **Time per cut**: ~360 attempts × 50 binary search steps = ~18,000 operations
- **Total time**: ~(n-1) cuts × 18,000 operations ≈ 2-4 seconds for 10 children
- **Memory**: Just need to store the queue of pieces and the list of cuts

## Summary

This algorithm is like having a really smart person cut your cake. Instead of just blindly cutting the same way every time, they:
1. Look at what needs to be cut
2. Try different ways to split it (both ratio and angle)
3. Pick the best one
4. Repeat until everyone has their piece

Simple idea, powerful results! 🎂

---

**File**: `players/player10/player_divide_conquer_1006.py`  
**Default Attempts**: 360 per cut  
**Typical Time**: 2-4 seconds for 10 children
