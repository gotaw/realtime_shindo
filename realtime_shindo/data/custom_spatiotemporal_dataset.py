from typing import List

import numpy as np
import pandas as pd
import torch
from tsl.data import SpatioTemporalDataset


class CustomSpatioTemporalDataset(SpatioTemporalDataset):
    def __init__(
        self, target: pd.DataFrame, time_gap: pd.Timedelta = pd.Timedelta("1s"), **kwargs
    ):
        super(CustomSpatioTemporalDataset, self).__init__(target=target, **kwargs)

        valid_indices = self._compute_valid_indices(time_gap)
        self.set_indices(torch.tensor(valid_indices, dtype=torch.long))

    def _compute_valid_indices(self, time_gap: pd.Timedelta) -> List[int]:
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
