import random

import numpy as np
import pandas as pd


def compute_window_indices(
    sequence_length: int, window_size: int, stride: int
) -> np.ndarray:
    """Compute the start and end indices for each window.

    Includes a possibly shorter last window if `sequence_length` is not an
    exact multiple of `stride`.

    Args:
        sequence_length: Total length of the sequence.
        window_size: Size of each window.
        stride: Stride between windows.

    Returns:
        A `(n_windows, 2)` array of `(start, end)` indices for each window.
    """
    start_indices = np.arange(0, sequence_length, stride)
    end_indices = np.minimum(start_indices + window_size, sequence_length)
    valid_windows = start_indices != end_indices
    return np.column_stack((start_indices[valid_windows], end_indices[valid_windows]))


def _get_annotations_in_window(
    annotations: pd.DataFrame, window_start: int, window_end: int
) -> pd.DataFrame:
    """Filter, clip, and shift annotations to fit within a specific window.

    Args:
        annotations: Annotation DataFrame with `start_frame` / `end_frame`
            columns (and optionally `start_ms` / `end_ms`).
        window_start: Start frame of the window (inclusive).
        window_end: End frame of the window (exclusive).

    Returns:
        The subset of `annotations` overlapping the window, with
        `start_frame` / `end_frame` (and `start_ms` / `end_ms`, if present)
        clipped to the window bounds and shifted to be window-relative.

    Raises:
        ValueError: If `annotations` lacks `start_frame` / `end_frame`
            columns.
    """
    if not {"start_frame", "end_frame"}.issubset(annotations.columns):
        raise ValueError(
            "Annotations must have 'start_frame' and 'end_frame' columns for windowing."
        )

    mask = (annotations["start_frame"] < window_end) & (
        annotations["end_frame"] >= window_start
    )

    df_window = annotations.loc[mask].copy()
    if df_window.empty:
        return df_window

    # Infer framerate from original values before clipping, if ms columns exist
    has_ms = {"start_ms", "end_ms"}.issubset(df_window.columns)
    fps = None
    if has_ms:
        dur_frames = df_window["end_frame"] - df_window["start_frame"]
        dur_ms = df_window["end_ms"] - df_window["start_ms"]
        valid = dur_ms > 0
        if valid.any():
            fps = (dur_frames[valid] / dur_ms[valid] * 1000).median()

    df_window["start_frame"] = df_window["start_frame"].clip(
        lower=window_start, upper=window_end
    )
    df_window["end_frame"] = df_window["end_frame"].clip(
        lower=window_start, upper=window_end
    )

    df_window["start_frame"] -= window_start
    df_window["end_frame"] -= window_start

    if has_ms and fps is not None:
        df_window["start_ms"] = (
            (df_window["start_frame"] / fps * 1000).round().astype(int)
        )
        df_window["end_ms"] = (df_window["end_frame"] / fps * 1000).round().astype(int)

    return df_window


def _get_window(
    data,
    start: int,
    end: int,
    add_window_metadata: bool = False,
    ignored_keys: set[str] | None = None,
    custom_window_functions: dict | None = None,
) -> dict:
    """Recursively slice a sample (or nested dict) to a `[start, end)` window.

    Args:
        data: The value to slice — an array/list is sliced directly, a dict
            is recursed into, anything else is returned unchanged.
        start: Start index of the window (inclusive).
        end: End index of the window (exclusive).
        add_window_metadata: If `True` (and `data` is a dict), add `start`,
            `end` and `n_frames` keys to the returned dict.
        ignored_keys: Dict keys to skip (kept out of the returned dict).
        custom_window_functions: Mapping from dict key to a
            `(value, start, end) -> value` callable, used instead of the
            default recursive slicing for that key (e.g. for annotations).

    Returns:
        The windowed value, with the same shape as `data`.
    """
    if isinstance(data, np.ndarray) or isinstance(data, list):
        return data[start:end]
    elif isinstance(data, dict):
        new_data = dict()
        for k, v in data.items():
            if ignored_keys and k in ignored_keys:
                continue
            elif custom_window_functions and k in custom_window_functions:
                new_data[k] = custom_window_functions[k](v, start, end)
            else:
                new_data[k] = _get_window(
                    v,
                    start,
                    end,
                    ignored_keys=ignored_keys,
                    custom_window_functions=custom_window_functions,
                )
        if add_window_metadata:
            new_data["start"] = start
            new_data["end"] = end
            new_data["n_frames"] = end - start
        return new_data
    else:
        return data


def _get_all_windows_from_sample(
    sample: dict, window_size: int, window_stride: int
) -> list[dict]:
    """Split a single sample into overlapping temporal windows.

    Args:
        sample: The sample dict. Must contain `n_frames` and, if present,
            an `annotations` dict of DataFrames keyed by annotation id.
        window_size: Window length, in the same unit as `sample["n_frames"]`.
        window_stride: Stride between consecutive windows.

    Returns:
        A list of windowed sample dicts, each with added `start`, `end` and
        `n_frames` keys.
    """
    window_indices = compute_window_indices(
        sample["n_frames"], window_size, window_stride
    )
    custom_fns = {}
    if "annotations" in sample:
        custom_fns["annotations"] = lambda x, start, end: {
            k: _get_annotations_in_window(v, start, end) for k, v in x.items()
        }
    return [
        _get_window(
            sample,
            start,
            end,
            add_window_metadata=True,
            custom_window_functions=custom_fns,
        )
        for start, end in window_indices
    ]


def convert_samples_to_windows(
    samples: list[dict], window_size: int, window_stride: int
):
    """Split every sample in a list into overlapping temporal windows.

    Args:
        samples: The samples to split, as produced by the WebDataset
            mapping functions in `sldl.dataset`.
        window_size: Window length, in the same unit as `sample["n_frames"]`.
        window_stride: Stride between consecutive windows.

    Returns:
        The flattened list of windowed samples across all input samples.
    """
    return sum(
        [
            _get_all_windows_from_sample(sample, window_size, window_stride)
            for sample in samples
        ],
        start=[],
    )


def filter_empty_windows(samples: list[dict], max_empty_windows: int):
    """Randomly drop windows that contain no annotations, down to a limit.

    A window is considered empty if every annotation track in its
    `annotations` dict is empty. If there are more empty windows than
    `max_empty_windows`, a random subset of `max_empty_windows` of them is
    kept and the rest are dropped; non-empty windows are always kept.

    Args:
        samples: Windowed samples, each with an `annotations` dict of
            DataFrames keyed by annotation id.
        max_empty_windows: Maximum number of empty windows to keep.

    Returns:
        The filtered list of samples.
    """
    empty_window_indices = [
        index
        for index, sample in enumerate(samples)
        if all(annots.shape[0] < 1 for annots in sample["annotations"].values())
    ]
    if len(empty_window_indices) <= max_empty_windows:
        return samples
    kept_empty_windows = random.sample(empty_window_indices, max_empty_windows)
    removed_windows_indices = set(empty_window_indices) - set(kept_empty_windows)
    return [
        sample for i, sample in enumerate(samples) if i not in removed_windows_indices
    ]
