import os

import numpy as np
import pandas as pd
from tsl.datasets.prototypes import DatetimeDataset
from tsl.ops.similarities import gaussian_kernel

from realtime_shindo.config import PROCESSED_DATA_DIR


class RealtimeShindo(DatetimeDataset):
    similarity_options = {"distance"}

    IMPUTATION_VALUE = 0.0
    SCALING_FACTOR = 10.0

    def __init__(self, root=None, net: str = "knet", freq=None):
        self.root = root if root is not None else PROCESSED_DATA_DIR
        self.net = net
        df, dist, mask = self.load()
        super().__init__(
            target=df,
            mask=mask,
            freq=freq,
            similarity_score="distance",
            temporal_aggregation="max",
            name=f"RealtimeShindo-{net}",
        )
        self.add_covariate("dist", dist, pattern="n n")

    @property
    def root_dir(self):
        return self.root

    @property
    def required_file_names(self):
        return [
            f"realtime_shindo_{self.net}.h5",
            f"distances_{self.net}.csv",
            f"station_ids_{self.net}.txt",
            f"station_locations_{self.net}.csv",
        ]

    def _get_required_path(self, filename: str) -> str:
        return os.path.join(self.root_dir, filename)

    def build(self) -> None:
        raw_dist_path = self._get_required_path(f"distances_{self.net}.csv")
        distances = pd.read_csv(raw_dist_path)

        ids_path = self._get_required_path(f"station_ids_{self.net}.txt")
        with open(ids_path) as f:
            ids = f.read().strip().split(",")

        num_sensors = len(ids)
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf

        sensor_to_ind = {sensor_id: i for i, sensor_id in enumerate(ids)}

        for row in distances.values:
            from_id, to_id, distance_val = row
            if from_id in sensor_to_ind and to_id in sensor_to_ind:
                dist[sensor_to_ind[from_id], sensor_to_ind[to_id]] = distance_val
                dist[sensor_to_ind[to_id], sensor_to_ind[from_id]] = distance_val

        path = self._get_required_path(f"dist_{self.net}.npy")
        np.save(path, dist)

    def load_raw(self):
        dist_path = self._get_required_path(f"dist_{self.net}.npy")
        if not os.path.exists(dist_path):
            self.build()

        data_path = self._get_required_path(f"realtime_shindo_{self.net}.h5")
        df = pd.read_hdf(data_path)

        dist = np.load(dist_path)
        return df, dist

    def load(self):
        df_raw, dist = self.load_raw()

        is_valid = df_raw > 0

        meaningful_ts_mask = is_valid.any(axis=1)
        df_filtered = df_raw[meaningful_ts_mask]
        is_valid_filtered = is_valid[meaningful_ts_mask]

        mask = is_valid_filtered.to_numpy(dtype="uint8")

        df_processed = df_filtered.where(is_valid_filtered, self.IMPUTATION_VALUE)

        df_final = (df_processed / self.SCALING_FACTOR).astype(np.float32)

        return df_final, dist, mask

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            dist_matrix = self.covariates["dist"]
            finite_dist = dist_matrix[np.isfinite(dist_matrix)]
            sigma = finite_dist.std()
            return gaussian_kernel(dist_matrix, sigma)
        raise ValueError(f"Unknown similarity method: {method}")
