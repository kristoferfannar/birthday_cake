# Player3 Parallel Processing Implementation

## Overview
This document describes the parallel processing capabilities added to Player3's cake cutting strategy to significantly speed up the cut-finding process.

## Files Created

### `parallel_search.py`
A new module that provides multiprocessing capabilities for parallelizing the evaluation of potential cuts across multiple CPU cores.

## Key Features

### 1. **Automatic CPU Detection**
- Automatically detects the number of available CPU cores
- Uses `cpu_count() - 1` workers by default (leaves one core free for system)
- Maximum of 8 workers (diminishing returns beyond that)

### 2. **Smart Parallelization**
- Only uses parallel processing when it makes sense (>=20 points for cuts, >=15 for coarse search)
- Falls back to serial processing for small datasets
- Graceful error handling with automatic fallback to serial mode

### 3. **Two Parallel Functions**

#### `parallel_find_valid_cuts()`
Parallelizes the main cut-finding process:
- Splits perimeter points into chunks
- Each worker evaluates a subset of potential cuts
- Results are merged and sorted by accuracy

#### `parallel_find_coarse_cuts()`
Parallelizes the coarse search phase in the refinement system:
- Used for the initial broad search with relaxed tolerances
- Speeds up the first phase of the two-phase refinement process

### 4. **Configuration Flags**
In `player.py`, new configuration options:
```python
self.use_parallel = True  # Toggle for parallel processing
self.num_workers = None  # None = auto-detect optimal worker count
```

## Usage

### Enable/Disable Parallel Processing
```python
# In player.py __init__
self.use_parallel = True   # Enable parallel processing
self.use_parallel = False  # Disable (use serial processing)
```

### Custom Worker Count
```python
self.num_workers = 4  # Use exactly 4 workers
self.num_workers = None  # Auto-detect (recommended)
```

## Performance Benefits

### Expected Speedup
- **2-6x faster** on multi-core systems (depending on CPU cores)
- Best performance on systems with 4+ cores
- No performance penalty on single-core systems (automatic fallback)

### When Parallel Processing Helps Most
- Large cakes with many perimeter points (70+ points)
- Complex shapes requiring many cut evaluations
- Multiple tolerance levels being tried

### When It Doesn't Help Much
- Very small cakes (<20 perimeter points)
- Simple shapes with few potential cuts
- Single-core systems

## Implementation Details

### Work Distribution
- Points are split into overlapping chunks to ensure all pairs are checked
- Chunk size is adaptive: `len(points) // (workers * 2)`
- More chunks than workers for better load balancing

### Data Sharing
- Uses Python's `multiprocessing.Pool` for process management
- Each worker gets its own copy of the cake data
- Results are collected and merged in the main process

### Error Handling
All parallel functions include:
- Try-except blocks for graceful error handling
- Automatic fallback to serial processing on failure
- Error messages printed for debugging

## Tolerance Adjustments

Along with parallel processing, the tolerances have been tuned for better homogeneity:

### Current Tolerance Values
- **Area tolerance**: `[0.15, 0.25, 0.35, 0.5, 0.75, 1.2]`
  - Starts at 0.15 (70% stricter than original 0.5)
- **Ratio tolerance**: `[0.015, 0.025, 0.035, 0.05, 0.075, 0.12]`
  - Starts at 0.015 (70% stricter than original 0.05)

### Refinement System Tolerances
- **Coarse search**: 2.5x base tolerance
- **Refinement fallback**: 1.8x base tolerance  
- **Last resort fallback**: 3.5x base tolerance

## Testing

### Verified Working On
- ✅ Star shape (10 children): `size span: 1.61cm²`
- ✅ Minecraft sword (10 children): Successfully completes
- ✅ Complex shapes with varying geometries

### How to Test
```bash
# Test with star shape
uv run main.py --import-cake cakes/players/player3/star.csv --player 3

# Test with minecraft sword  
uv run main.py --import-cake cakes/players/player3/minecraft_sword.csv --player 3

# Test with GUI
uv run main.py --import-cake cakes/players/player3/star.csv --player 3 --gui
```

## Future Improvements

Potential optimizations:
1. **Shared memory**: Use shared memory arrays for cake data to reduce copying overhead
2. **Process pool persistence**: Reuse worker pool across multiple cuts
3. **GPU acceleration**: For very large datasets, consider GPU-based geometry calculations
4. **Adaptive parallelization**: Dynamically adjust worker count based on workload

## Compatibility

- **Python Version**: Requires Python 3.8+ (for multiprocessing improvements)
- **Dependencies**: Only uses standard library (`multiprocessing`)
- **Platform**: Works on Linux, macOS, and Windows
- **Thread Safety**: Uses process-based parallelism (avoids GIL issues)

## Notes

- Parallel processing adds a small overhead for process creation and communication
- For very fast operations on small datasets, serial might be faster
- The automatic threshold detection ensures parallel is only used when beneficial
- All parallel functions maintain the same output format as their serial counterparts

