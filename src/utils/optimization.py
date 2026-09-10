def metric_stability(metric_train: float, metric_valid: float, max_difference: float) -> float:
    if (metric_train - metric_valid) / metric_train > max_difference:
        return 0
    else:
        return metric_valid
