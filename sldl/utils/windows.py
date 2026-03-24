import random

import numpy as np
import pandas as pd


def compute_window_indices(
    sequence_length: int, window_size: int, stride: int
) -> np.ndarray:
    """
    Compute the start and end indices for each window, including a possibly shorter last window.

    Args:
        sequence_length: Total length of the sequence
        window_size: Size of each window
        stride: Stride between windows

    Returns:
        np.ndarray: Array of shape (n_windows, 2) containing start and end indices for each window
    """
    start_indices = np.arange(0, sequence_length, stride)
    end_indices = np.minimum(start_indices + window_size, sequence_length)
    valid_windows = start_indices != end_indices
    return np.column_stack((start_indices[valid_windows], end_indices[valid_windows]))


def _get_annotations_in_window(
    annotations: pd.DataFrame, window_start: int, window_end: int
) -> pd.DataFrame:
    """
    Filters, clips, and shifts annotations to fit within a specific window.
    Assumes `annotations` has 'start_frame' and 'end_frame' columns.
    """
    mask = (annotations["start_frame"] < window_end) & (
        annotations["end_frame"] >= window_start
    )

    df_window = annotations.loc[mask].copy()
    if df_window.empty:
        return df_window

    df_window["start_frame"] = df_window["start_frame"].clip(
        lower=window_start, upper=window_end
    )
    df_window["end_frame"] = df_window["end_frame"].clip(
        lower=window_start, upper=window_end
    )

    # Shift the frames so they are relative to the start of the window
    df_window["start_frame"] -= window_start
    df_window["end_frame"] -= window_start

    # TODO: update start_ms and end_ms. We need the framerate here...
    return df_window


def _get_window(
    data,
    start: int,
    end: int,
    add_window_metadata: bool = False,
    ignored_keys: set[str] | None = None,
    custom_window_functions: dict | None = None,
) -> dict:
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
    return sum(
        [
            _get_all_windows_from_sample(sample, window_size, window_stride)
            for sample in samples
        ],
        start=[],
    )


def filter_empty_windows(samples: list[dict], max_empty_windows: int):
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
