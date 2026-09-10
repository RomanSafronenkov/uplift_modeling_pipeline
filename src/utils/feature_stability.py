import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10, bucket_type='quantiles', epsilon=1e-10):
    """
    Calculate PSI for each common feature in two DataFrames

    Params:
    -----
        expected: pd.DataFrame
            Base DataFrame
        actual: pd.DataFrame
            New (current) DataFrame to compare
        bins: int, default=10
            Number of bins for continuous features
        bucket_type: str, default='quantiles'
            The way to create buckets:
            - 'quantiles' - expected quantile borders (recommended)
            - 'equal' - equal intervals
        epsilon: float, default=1e-10
            Add to avoid log(0)

    Returns:
    -----
        pd.DataFrame
            DataFrame with features and their PSI
    """

    if isinstance(expected, pd.Series):
        expected = expected.to_frame()
    if isinstance(actual, pd.Series):
        actual = actual.to_frame()

    common_cols = set(expected.columns) & set(actual.columns)
    if not common_cols:
        raise ValueError('There are no common columns in expected and actual')

    results = []

    for col in common_cols:
        exp_vals = expected[col].dropna()
        act_vals = actual[col].dropna()

        if exp_vals.empty or act_vals.empty:
            results.append({'feature': col, 'psi': np.nan})
            continue

        if bucket_type == 'quantiles':
            edges = np.percentile(exp_vals, np.linspace(0, 100, bins + 1))
            edges = np.unique(edges)

            if len(edges) < 2:
                bucket_type = 'equal'

        if bucket_type == 'equal':
            min_val = min(exp_vals.min(), act_vals.min())
            max_val = max(exp_vals.max(), act_vals.max())
            edges = np.linspace(min_val, max_val, bins + 1)
            edges = np.unique(edges)

            if len(edges) < 2:
                edges = np.array([min_val, min_val + 1.0])

        exp_counts, _ = np.histogram(exp_vals, bins=edges)
        act_counts, _ = np.histogram(act_vals, bins=edges)

        exp_props = exp_counts / exp_counts.sum()
        act_props = act_counts / act_counts.sum()

        exp_props = np.where(exp_props == 0, epsilon, exp_props)
        act_props = np.where(act_props == 0, epsilon, act_props)

        psi = np.sum((act_props - exp_props) * np.log(act_props / exp_props))
        results.append({'feature': col, 'psi': psi})

    return pd.DataFrame(results)
