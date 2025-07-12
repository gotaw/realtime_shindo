"""
Dataset creation script for the realtime_shindo project.

This script provides Typer CLI commands to:
1. `external`: Download raw waveform data from the Kyoshin Net website.
2. `interim`: Process raw tar.gz files into per-event obspy.Stream pickle files.
3. `processed`: Calculate Ir values from interim pickle files,
               save them in a structured HDF5 file, and generate station feature files.
4. `main`: Run the full data processing pipeline: external -> interim -> processed.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import gc
from itertools import combinations_with_replacement
from math import atan2, cos, radians, sin, sqrt
import os
from pathlib import Path
from subprocess import run
import tarfile
import tempfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from loguru import logger
import numpy as np
from obspy import Stream
from obspy import read as obspy_read
import pandas as pd
from scipy.signal import lfilter
from sortedcontainers import SortedList
from tqdm import tqdm
import typer

from realtime_shindo.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    KYOSHIN_PASSWORD,
    KYOSHIN_USER,
    PROCESSED_DATA_DIR,
)

HDF_KEY = "data"
app = typer.Typer(help="A CLI for downloading and processing Kyoshin Net (K-NET, KiK-net) data.")


@dataclass(frozen=True)
class Config:
    """Central configuration for data processing."""

    net_configs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "knet": {
                "comp_dir": "3comp",
                "exts": [".EW", ".NS", ".UD"],
                "channels": ("NS", "EW", "UD"),
            },
            "kik": {
                "comp_dir": "6comp",
                "exts": [".EW2", ".NS2", ".UD2"],
                "channels": ("NS2", "EW2", "UD2"),
            },
        }
    )
    freq_params: Dict[str, float] = field(
        default_factory=lambda: {
            "f0": 0.45,
            "f1": 7.0,
            "f2": 0.5,
            "f3": 12.0,
            "f4": 20.0,
            "f5": 30.0,
        }
    )
    damping_params: Dict[str, float] = field(
        default_factory=lambda: {"h2a": 1.0, "h2b": 0.75, "h3": 0.9, "h4": 0.6, "h5": 0.6}
    )
    gain: float = 1.262
    dt: float = 0.01
    sampling_rate: int = 100
    window_seconds: int = 1
    k_seconds: float = 0.3

    @property
    def window_samples(self) -> int:
        return self.window_seconds * self.sampling_rate

    @property
    def k_samples(self) -> int:
        return int(self.k_seconds * self.sampling_rate)


@dataclass(frozen=True)
class Paths:
    """A data class to manage paths for a specific network."""

    net: str

    @property
    def site_file(self) -> Path:
        base_path = EXTERNAL_DATA_DIR / "www.kyoshin.bosai.go.jp/kyoshin/download"
        return base_path / self.net / "sitedb" / f"sitepub_{self.net}_en.csv"

    @property
    def interim_dir(self) -> Path:
        return INTERIM_DATA_DIR / self.net

    @property
    def output_hdf_path(self) -> Path:
        return PROCESSED_DATA_DIR / f"realtime_shindo_{self.net}.h5"


class IrCalculator:
    """Calculates Ir from pre-processed waveform data (in cm/s^2)."""

    def __init__(self, config: Config):
        self.config = config
        self.filter_coeffs = self._calculate_filter_coefficients()

    def _calculate_filter_coefficients(self) -> List[Dict[str, np.ndarray]]:
        """Calculates IIR filter coefficients based on Kunugi et al. (2013)."""
        c, PI, dt2 = self.config, np.pi, self.config.dt**2

        def omega(f):
            return 2 * PI * f

        def create_biquad(b, a):
            return {"b": np.array(b) / a[0], "a": np.array(a) / a[0]}

        def create_filter_5_coeffs(w, h):
            """Helper to create coefficients for the 5th-order filters."""
            a = [
                12 / dt2 + 12 * h * w / c.dt + w**2,
                10 * w**2 - 24 / dt2,
                12 / dt2 - 12 * h * w / c.dt + w**2,
            ]
            b = [w**2, 10 * w**2, w**2]
            return create_biquad(b, a)

        coeffs = []
        w0, w1_ = omega(c.freq_params["f0"]), omega(c.freq_params["f1"])
        a = [
            8 / dt2 + (4 * w0 + 2 * w1_) / c.dt + w0 * w1_,
            2 * w0 * w1_ - 16 / dt2,
            8 / dt2 - (4 * w0 + 2 * w1_) / c.dt + w0 * w1_,
        ]
        b = [4 / dt2 + 2 * w1_ / c.dt, -8 / dt2, 4 / dt2 - 2 * w1_ / c.dt]
        coeffs.append(create_biquad(b, a))

        w1 = omega(c.freq_params["f1"])
        a = [
            16 / dt2 + 17 * w1 / c.dt + w1**2,
            2 * w1**2 - 32 / dt2,
            16 / dt2 - 17 * w1 / c.dt + w1**2,
        ]
        b = [
            4 / dt2 + 8.5 * w1 / c.dt + w1**2,
            2 * w1**2 - 8 / dt2,
            4 / dt2 - 8.5 * w1 / c.dt + w1**2,
        ]
        coeffs.append(create_biquad(b, a))

        w2, h2a, h2b = omega(c.freq_params["f2"]), c.damping_params["h2a"], c.damping_params["h2b"]
        a = [
            12 / dt2 + 12 * h2b * w2 / c.dt + w2**2,
            10 * w2**2 - 24 / dt2,
            12 / dt2 - 12 * h2b * w2 / c.dt + w2**2,
        ]
        b = [
            12 / dt2 + 12 * h2a * w2 / c.dt + w2**2,
            10 * w2**2 - 24 / dt2,
            12 / dt2 - 12 * h2a * w2 / c.dt + w2**2,
        ]
        coeffs.append(create_biquad(b, a))

        for f_key, h_key in [("f3", "h3"), ("f4", "h4"), ("f5", "h5")]:
            coeffs.append(
                create_filter_5_coeffs(omega(c.freq_params[f_key]), c.damping_params[h_key])
            )
        return coeffs

    def _apply_filters(self, trace_data: np.ndarray) -> np.ndarray:
        """Applies the pre-computed series of IIR filters to trace data."""
        data = trace_data
        for coeff in self.filter_coeffs:
            data = lfilter(coeff["b"], coeff["a"], data)
        return data * self.config.gain

    def _calculate_ir_timeseries(self, synth_waveform: np.ndarray) -> np.ndarray:
        """Calculates the time series of Ir values from a synthetic waveform."""
        intensity_series = np.full(len(synth_waveform), -128, dtype=np.int8)
        sorted_list = SortedList()
        win_samples, k_samples = self.config.window_samples, self.config.k_samples

        for i, val in enumerate(synth_waveform):
            sorted_list.add(val)
            if i >= win_samples:
                sorted_list.remove(synth_waveform[i - win_samples])

            if len(sorted_list) >= k_samples and (kth_val := sorted_list[-k_samples]) > 0:
                intensity_series[i] = np.round((2 * np.log10(kth_val) + 0.94) * 10.0).astype(
                    np.int8
                )
        return intensity_series

    def calculate_for_station(self, st_station: Stream, net: str) -> Optional[pd.Series]:
        """Processes a single station's data to calculate an Ir time series."""
        if len(st_station) < 3:
            return None
        try:
            channels = self.config.net_configs[net]["channels"]
            tr_n, tr_e, tr_z = (st_station.select(channel=ch)[0] for ch in channels)
            data_n, data_e, data_z = (self._apply_filters(tr.data) for tr in (tr_n, tr_e, tr_z))
            synth = np.sqrt(data_n**2 + data_e**2 + data_z**2)
            ir_values = self._calculate_ir_timeseries(synth)
            timestamps = pd.to_datetime(tr_n.times("timestamp"), unit="s")
            return pd.Series(ir_values, index=timestamps, name=tr_n.stats.station)
        except IndexError:
            logger.debug(f"Missing channels for station {st_station[0].stats.station}. Skipping.")
            return None
        except Exception as e:
            logger.warning(f"Failed to process station {st_station[0].stats.station}: {e}")
            return None


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance (km) between two points."""
    R = 6371.0
    phi1, phi2, d_phi, d_lambda = map(radians, [lat1, lat2, lat2 - lat1, lon2 - lon1])
    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_station_locations(site_file_path: Path) -> Optional[pd.DataFrame]:
    """Loads station location data from the site information CSV file."""
    if not site_file_path.exists():
        logger.error(f"Station site file not found: {site_file_path}")
        return None
    try:
        df = pd.read_csv(
            site_file_path,
            header=None,
            usecols=[0, 2, 3, 4],
            names=["station_id", "latitude", "longitude", "altitude"],
            dtype={"station_id": str},
        ).dropna()
        df["station_id"] = df["station_id"].str.strip()
        return df
    except Exception as e:
        logger.error(f"Failed to load station locations from {site_file_path}: {e}")
        return None


def generate_station_features(locations_df: pd.DataFrame, net: str) -> None:
    """Generates and saves station location, ID, and distance files."""
    logger.info(f"Generating feature files for {net}...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    locations_df[["station_id", "latitude", "longitude"]].to_csv(
        PROCESSED_DATA_DIR / f"station_locations_{net}.csv", index=True, index_label="index"
    )
    with open(PROCESSED_DATA_DIR / f"station_ids_{net}.txt", "w") as f:
        f.write(",".join(locations_df["station_id"]))

    logger.info(f"Calculating pairwise distances for {net}...")
    pairs = combinations_with_replacement(locations_df.itertuples(), 2)
    total_pairs = len(locations_df) * (len(locations_df) + 1) // 2

    distance_records = (
        {
            "from_station": s1.station_id,
            "to_station": s2.station_id,
            "distance": _haversine_distance(s1.latitude, s1.longitude, s2.latitude, s2.longitude),
        }
        for s1, s2 in pairs
    )

    df_distances = pd.DataFrame(
        tqdm(distance_records, total=total_pairs, desc=f"Distances for {net}")
    )
    df_distances.to_csv(PROCESSED_DATA_DIR / f"distances_{net}.csv", index=False)
    logger.success(f"Successfully generated all feature files for {net}.")


def save_to_hdf(df: pd.DataFrame, output_path: Path):
    """Saves a DataFrame to a compressed HDF5 file in 'fixed' format."""
    logger.info(f"Saving {len(df)} records to HDF5 at {output_path}...")
    df.to_hdf(output_path, key=HDF_KEY, format="fixed", complevel=9, complib="zlib")
    logger.success("Data saved successfully.")


def _run_in_parallel(
    func: Callable, args_list: List[Tuple], max_workers: int, desc: str
) -> Iterator[Any]:
    """A generic wrapper for running functions in parallel and tracking progress."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                yield future.result()
            except Exception as e:
                logger.error(f"A worker process failed: {e}")


def _worker_create_interim_pkl(
    event_dir: Path, out_pkl_path: Path, exts: List[str]
) -> Optional[str]:
    """Worker for 'interim': Extracts and merges waveforms for a single event."""
    if out_pkl_path.exists():
        return str(out_pkl_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for tar_path in event_dir.glob("*.tar.gz"):
            try:
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=tmp_path)
            except tarfile.TarError as e:
                logger.warning(f"Could not extract {tar_path}: {e}")
                continue

        files_to_read = (f for ext in exts for f in tmp_path.rglob(f"*{ext}"))

        streams = []
        for f in files_to_read:
            try:
                streams.append(obspy_read(str(f), format="KNET"))
            except Exception as e:
                logger.warning(f"Could not read waveform file {f}: {e}")

    if not streams:
        return None

    merged_stream = Stream(traces=[tr for st in streams for tr in st])
    merged_stream.write(str(out_pkl_path), format="PICKLE")
    return str(out_pkl_path)


def _preprocess_stream(st: Stream, config: Config) -> Stream:
    """Applies standard preprocessing steps to a Stream object."""
    st.detrend("linear")
    st.taper(max_percentage=None, max_length=3.0)
    for tr in st:
        tr.data = tr.data * tr.stats.calib * 100
    st.resample(config.sampling_rate)
    return st


def _worker_calculate_ir(
    pkl_path: Path,
    calculator: IrCalculator,
    config: Config,
    net: str,
    all_station_codes: List[str],
) -> Optional[pd.DataFrame]:
    """Worker for 'processed': Reads, processes, calculates Ir, and formats data for one event."""
    try:
        st = obspy_read(str(pkl_path), format="PICKLE")
    except Exception as e:
        logger.warning(f"Failed to read pickle {pkl_path}: {e}")
        return None

    if not st:
        return None

    st_preprocessed = _preprocess_stream(st, config)

    station_codes = list(set(tr.stats.station for tr in st_preprocessed))
    all_series = []
    for code in station_codes:
        st_station = st_preprocessed.select(station=code)
        if (series := calculator.calculate_for_station(st_station, net)) is not None:
            all_series.append(series)

    if not all_series:
        return None

    df = pd.concat(all_series, axis=1)
    df_resampled = df.asfreq("s")
    df_final = df_resampled.reindex(columns=all_station_codes)

    return df_final.fillna(-128).astype(np.int8)


def _run_processed_pipeline_for_net(net: str, config: Config, max_workers: int, force: bool):
    """Orchestrates the 'processed' data pipeline for a single network."""
    paths = Paths(net)
    if paths.output_hdf_path.exists() and not force:
        logger.info(
            f"Output file {paths.output_hdf_path} exists. Skipping. Use --force to overwrite."
        )
        return

    locations_df = load_station_locations(paths.site_file)
    if locations_df is None:
        return
    all_station_codes = locations_df["station_id"].tolist()

    pkl_files = sorted(list(paths.interim_dir.rglob("*.pkl")))
    if not pkl_files:
        logger.warning(f"No .pkl files found in {paths.interim_dir}. Skipping {net}.")
        return

    if paths.output_hdf_path.exists() and force:
        logger.warning(f"Force enabled. Deleting existing file: {paths.output_hdf_path}")
        paths.output_hdf_path.unlink()

    ir_calculator = IrCalculator(config)
    args_list = [(pkl, ir_calculator, config, net, all_station_codes) for pkl in pkl_files]

    results = _run_in_parallel(
        _worker_calculate_ir, args_list, max_workers, f"Processing {net} events"
    )
    all_event_dfs = [df for df in results if df is not None and not df.empty]

    if all_event_dfs:
        logger.info(f"Combining {len(all_event_dfs)} processed event DataFrames for {net}...")
        combined_df = pd.concat(all_event_dfs)

        logger.info(
            "Resolving duplicate timestamps by taking the maximum value for each station..."
        )
        final_df = combined_df.groupby(combined_df.index).max()

        original_rows = len(combined_df)
        final_rows = len(final_df)
        logger.info(
            f"Combined {original_rows:,} rows, resolved to {final_rows:,} unique timestamp rows."
        )

        save_to_hdf(final_df, paths.output_hdf_path)
        generate_station_features(locations_df, net)

        del final_df
        del combined_df
    else:
        logger.warning(f"No data was successfully processed for {net}. HDF5 file not created.")

    del all_event_dfs
    gc.collect()


@app.command()
def external() -> None:
    """Download raw waveform data from kyoshin.bosai.go.jp."""
    logger.info(f"Downloading data from kyoshin.bosai.go.jp as user: {KYOSHIN_USER}")
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        f"--user={KYOSHIN_USER}",
        f"--password={KYOSHIN_PASSWORD}",
        "-m",
        "-np",
        "https://www.kyoshin.bosai.go.jp/kyoshin/download/",
    ]
    run(cmd, cwd=EXTERNAL_DATA_DIR, check=True)
    logger.success("Download complete.")


@app.command()
def interim(
    max_workers: Optional[int] = typer.Option(
        None, help="Number of worker processes. Defaults to CPU count."
    ),
):
    """Process raw tar files into per-event obspy.Stream pickle files."""
    max_workers = max_workers or os.cpu_count() or 1
    config = Config()

    for net, net_conf in config.net_configs.items():
        src_root = (
            EXTERNAL_DATA_DIR
            / f"www.kyoshin.bosai.go.jp/kyoshin/download/{net}/{net_conf['comp_dir']}"
        )
        out_root = INTERIM_DATA_DIR / net

        if not src_root.exists():
            logger.warning(f"Source directory not found for {net}: {src_root}")
            continue

        event_dirs = sorted([p for p in src_root.glob("????/??/*") if p.is_dir()])
        if not event_dirs:
            logger.warning(f"No events found to process for net: {net}")
            continue

        args_list = []
        for event_dir in event_dirs:
            relative_path = event_dir.relative_to(src_root)
            out_pkl_path = (out_root / relative_path).with_suffix(".pkl")
            out_pkl_path.parent.mkdir(parents=True, exist_ok=True)
            args_list.append((event_dir, out_pkl_path, net_conf["exts"]))

        processed_files = list(
            _run_in_parallel(
                _worker_create_interim_pkl,
                args_list,
                max_workers,
                f"Creating interim pickles for {net}",
            )
        )
        logger.info(
            f"Finished processing for {net}. {sum(1 for f in processed_files if f)} files created/updated."
        )

    logger.success("Interim data creation complete.")


@app.command()
def processed(
    max_workers: Optional[int] = typer.Option(
        None, help="Number of worker processes. Defaults to CPU count."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite of existing processed files."
    ),
):
    """Calculate Ir values from interim pkl files and create final datasets."""
    max_workers = max_workers or os.cpu_count() or 1
    config = Config()

    for net in config.net_configs.keys():
        logger.info(f"--- Starting 'processed' for network: {net} ---")
        _run_processed_pipeline_for_net(net, config, max_workers, force)
        logger.info(f"--- Finished 'processed' for network: {net} ---")


@app.command()
def main(
    max_workers: Optional[int] = typer.Option(None, help="Number of worker processes."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite of existing processed files."
    ),
):
    """Run the full data processing pipeline: external -> interim -> processed."""
    logger.info("--- Step 1/3: Starting `external` data download ---")
    try:
        external()
        logger.info("--- Step 1/3: `external` data download finished ---")
    except Exception as e:
        logger.error(f"Step 1/3 `external` failed: {e}")
        raise typer.Exit(code=1)

    logger.info("--- Step 2/3: Starting `interim` data creation ---")
    interim(max_workers=max_workers)
    logger.info("--- Step 2/3: `interim` data creation finished ---")

    logger.info("--- Step 3/3: Starting `processed` data creation ---")
    processed(max_workers=max_workers, force=force)
    logger.info("--- Step 3/3: `processed` data creation finished ---")

    logger.success("Full data pipeline completed successfully!")


if __name__ == "__main__":
    app()
