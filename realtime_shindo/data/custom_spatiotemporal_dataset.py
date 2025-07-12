from typing import List

import numpy as np
import pandas as pd
import torch
from tsl.data import SpatioTemporalDataset


class CustomSpatioTemporalDataset(SpatioTemporalDataset):
    """
    A SpatioTemporalDataset specifically designed for discontinuous, event-based
    time series data.

    This class overrides the default sliding window behavior to ensure that
    samples do not cross large time gaps between events. It does so by
    pre-computing a set of valid start indices and setting them in the parent
    class.

    Args:
        target (pd.DataFrame): The main time series data.
        time_gap (pd.Timedelta): The minimum time difference between two
            consecutive timestamps to be considered a gap between events.
        **kwargs: Additional arguments passed to the parent
            `SpatioTemporalDataset`.
    """

    def __init__(self, target: pd.DataFrame, time_gap: pd.Timedelta = pd.Timedelta("1s"), **kwargs):
        super(CustomSpatioTemporalDataset, self).__init__(target=target, **kwargs)

        valid_indices = self._compute_valid_indices(time_gap)
        self.set_indices(torch.tensor(valid_indices, dtype=torch.long))

    def _compute_valid_indices(self, time_gap: pd.Timedelta) -> List[int]:
        """
        Identifies continuous data blocks (events) and computes all valid
        starting indices for sliding window sampling within these blocks.
        """
        time_index = self.index

        time_diffs = np.diff(time_index)

        gap_indices = np.where(time_diffs > time_gap)[0] + 1

        event_boundaries = [0] + gap_indices.tolist() + [len(time_index)]

        valid_starting_indices = []
        sample_len = self.sample_span

        for start, end in zip(event_boundaries[:-1], event_boundaries[1:]):
            if end - start >= sample_len:
                last_possible_start = end - sample_len
                indices_in_block = list(range(start, last_possible_start + 1, self.stride))
                valid_starting_indices.extend(indices_in_block)

        return valid_starting_indices
