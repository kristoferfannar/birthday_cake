#!/usr/bin/env python3
"""
Test script to find the optimal num_of_processes for Player10 concurrency.
Tests different process counts and measures execution time.
"""

import time
import subprocess
from pathlib import Path


def run_test(num_processes: int, test_runs: int = 3) -> float:
    """Run the game with specified num_processes and return average execution time."""

    # Create a temporary modified Player10 class with the specified num_processes
    player_file = Path("players/player10/player_concurrency_1013.py")
    original_content = player_file.read_text()

    # Replace the default num_of_processes value in the constructor
    modified_content = original_content.replace(
        "num_of_processes: int = 4,", f"num_of_processes: int = {num_processes},"
    )

    # Write the modified version
    player_file.write_text(modified_content)

    times = []

    for run in range(test_runs):
        print(f"  Testing {num_processes} processes, run {run + 1}/{test_runs}...")

        start_time = time.time()

        # Run the game command (without GUI for accurate timing)
        cmd = [
            "uv",
            "run",
            "main.py",
            "--import-cake",
            "cakes/players/player10/star.csv",
            "--player",
            "10",
            "--children",
            "10",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            end_time = time.time()
            execution_time = end_time - start_time

            if result.returncode == 0:
                times.append(execution_time)
                print(f"    Run {run + 1}: {execution_time:.2f}s")
            else:
                print(f"    Run {run + 1} failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"    Run {run + 1} timed out")
            return None
        except Exception as e:
            print(f"    Run {run + 1} error: {e}")
            return None

    # Restore original file
    player_file.write_text(original_content)

    if times:
        avg_time = sum(times) / len(times)
        print(
            f"  {num_processes} processes: {avg_time:.2f}s average ({len(times)} runs)"
        )
        return avg_time
    else:
        return None


def main():
    """Test different num_of_processes values and find the optimal one."""

    print("Testing optimal num_of_processes for Player10 concurrency...")
    print("=" * 60)

    # Test different process counts (focusing on optimal range)
    process_counts = [1, 2, 4, 6, 8, 10, 12]

    results = {}

    for num_processes in process_counts:
        avg_time = run_test(
            num_processes, test_runs=2
        )  # Reduced to 2 runs for faster testing
        if avg_time is not None:
            results[num_processes] = avg_time

    if not results:
        print("No successful test runs completed!")
        return

    # Find the best (fastest) configuration
    best_processes = min(results.keys(), key=lambda k: results[k])
    best_time = results[best_processes]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print("=" * 60)

    for processes, avg_time in sorted(results.items()):
        speedup = results[1] / avg_time if processes > 1 else 1.0
        print(f"{processes:2d} processes: {avg_time:6.2f}s (speedup: {speedup:.2f}x)")

    print("\n" + "=" * 60)
    print(f"OPTIMAL: {best_processes} processes ({best_time:.2f}s)")
    print(f"That's {results[1] / best_time:.2f}x faster than single-threaded")
    print("=" * 60)


if __name__ == "__main__":
    main()
