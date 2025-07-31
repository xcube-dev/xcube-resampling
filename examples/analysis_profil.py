import os
import pstats
from pathlib import Path

# Load the saved profiling data
DIR = Path(__file__).parent.resolve()
stats = pstats.Stats(os.path.join(DIR, "debug.prof"))

# Sort by cumulative time and display the top 100 entries
# stats.strip_dirs().sort_stats("cumtime").print_stats(100)
stats.strip_dirs().sort_stats("cumtime").print_stats(100)
