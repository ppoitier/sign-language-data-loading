import numpy as np


def compute_class_weights(
    occurrences: dict,
    strategy: str = "effective_number",
    alpha: float = 0.5,
    beta: float = 0.999,
    log_offset: float = 1.02,
    median_clip: float | None = None,
    normalize: bool = True,
) -> dict:
    """Compute class weights from a dict of label -> count.

    Strategies:
        - "uniform": w_c = 1 (trivial baseline)
        - "power": w_c = 1 / n_c ** alpha
            Generalizes inverse frequency (alpha=1) and inverse sqrt
            (alpha=0.5). Mikolov et al. (2013) use alpha=0.75 for
            word2vec negative sampling. Values in [0.5, 0.75] are a
            good starting range for long-tail data.
        - "inverse": shorthand for power with alpha=1.
        - "inverse_sqrt": shorthand for power with alpha=0.5.
        - "median": w_c = median(n) / n_c (Eigen & Fergus, 2015).
        - "log": w_c = 1 / log(log_offset + n_c) (Paszke et al., 2016, ENet).
        - "effective_number": w_c = (1 - beta) / (1 - beta ** n_c)
          (Cui et al., 2019). Often the best default for long-tail
          classification. beta=0.999 is the standard value.

    When normalize is True, weights are rescaled so their mean is 1.

    Args:
        occurrences: Mapping from label to its occurrence count.
        strategy: One of the strategies listed above.
        alpha: Exponent used by the "power" strategy. Must be in `[0, 1]`.
        beta: Decay rate used by the "effective_number" strategy. Must be
            in `[0, 1)`.
        log_offset: Offset added before taking the log in the "log"
            strategy. Must be `> 1`.
        median_clip: If set, clip each weight to
            `median_clip * median(weights)`.
        normalize: If `True`, rescale weights so their mean is 1.

    Returns:
        A dict mapping each label to its weight.

    Raises:
        ValueError: If any count is non-positive, if `strategy` is unknown,
            or if `alpha`/`beta`/`log_offset` is out of its valid range.
    """
    labels = list(occurrences.keys())
    counts = np.array([occurrences[l] for l in labels], dtype=np.float64)

    if np.any(counts <= 0):
        raise ValueError("All class counts must be strictly positive.")

    if strategy == "uniform":
        weights = np.ones_like(counts)
    elif strategy == "power":
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        weights = 1.0 / np.power(counts, alpha)
    elif strategy == "inverse":
        weights = 1.0 / counts
    elif strategy == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    elif strategy == "median":
        weights = np.median(counts) / counts
    elif strategy == "log":
        if log_offset <= 1.0:
            raise ValueError("log_offset must be > 1.")
        weights = 1.0 / np.log(log_offset + counts)
    elif strategy == "effective_number":
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy!r}")

    if median_clip is not None:
        weights = np.minimum(weights, median_clip * np.median(weights))

    if normalize:
        weights = weights * (len(weights) / weights.sum())

    return dict(zip(labels, weights))
