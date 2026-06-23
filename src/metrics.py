import numpy as np
import pandas as pd


def compute_stats(times_ms: list) -> dict:
    """Compute latency statistics from a list of millisecond timings."""
    arr = np.array(times_ms, dtype=float)
    return {
        'count':   int(len(arr)),
        'mean_ms': float(np.mean(arr)),
        'std_ms':  float(np.std(arr)),
        'min_ms':  float(np.min(arr)),
        'p50_ms':  float(np.percentile(arr, 50)),
        'p90_ms':  float(np.percentile(arr, 90)),
        'p99_ms':  float(np.percentile(arr, 99)),
        'max_ms':  float(np.max(arr)),
    }


def stats_to_dataframe(stats_dict: dict) -> pd.DataFrame:
    return pd.DataFrame([stats_dict])


def fps_from_mean_ms(mean_ms: float) -> float:
    return 1000.0 / mean_ms if mean_ms > 0 else 0.0
