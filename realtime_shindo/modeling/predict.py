from pathlib import Path

from loguru import logger
import numpy as np
import pytorch_lightning as pl
import torch
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.utils.casting import torch_to_numpy
import typer

from realtime_shindo.data import CustomSpatioTemporalDataset
from realtime_shindo.datasets import RealtimeShindo
from realtime_shindo.engines import CustomPredictor
from realtime_shindo.nn import custom_models  # noqa: F401

# PyTorch 2.0+ performance optimization
torch.set_float32_matmul_precision("medium")

app = typer.Typer()


@app.command()
def main(
    version_dir: Path = typer.Option(
        ...,
        "--version",
        "-v",
        help="Path to the Lightning logs version directory (e.g., lightning_logs/version_0).",
        exists=True,
    ),
    net: str = typer.Option("knet", help="Network to use: 'knet' or 'kik'."),
    data_split: str = typer.Option(
        "val", help="Data split to use for prediction: 'val' or 'test'."
    ),
    batch_size: int = typer.Option(64, help="Batch size for prediction."),
    gpus: int = typer.Option(
        -1 if torch.cuda.is_available() else 0,
        help="Number of GPUs to use (-1 for all available, 0 for CPU).",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output file."),
):
    """
    Perform inference with a trained model and save the predictions.
    """
    checkpoint_files = list(version_dir.glob("*.ckpt"))
    if not checkpoint_files:
        logger.error(f"No checkpoint file found in {version_dir}")
        raise typer.Exit(code=1)
    if len(checkpoint_files) > 1:
        logger.warning(
            f"Multiple checkpoint files found in {version_dir}. Using the first one: {checkpoint_files[0].name}"
        )
    checkpoint_path = checkpoint_files[0]

    logger.info(f"Starting inference for model: {checkpoint_path}")

    output_path = version_dir / "predictions.npz"

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu" if gpus != 0 else "cpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else "auto",
    )

    if trainer.is_global_zero:
        if output_path.exists() and not force:
            logger.error(f"Output file {output_path} already exists. Use --force to overwrite.")
            raise typer.Exit(code=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load hyperparameters from checkpoint to configure the dataset
    logger.info("Loading hyperparameters from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    hparams = checkpoint["hyper_parameters"]

    window = hparams["model_kwargs"]["window"]
    horizon = hparams["model_kwargs"]["horizon"]
    stride = hparams["model_kwargs"]["stride"]

    logger.info(f"Recreating dataset with window={window}, horizon={horizon}, stride={stride}")

    # 2. Dataset and DataModule setup
    dataset = RealtimeShindo(net=net)
    connectivity = dataset.get_connectivity(
        threshold=0.1, include_self=False, normalize_axis=1, layout="edge_index"
    )
    torch_dataset = CustomSpatioTemporalDataset(
        target=dataset.dataframe(),
        connectivity=connectivity,
        mask=dataset.mask,
        horizon=horizon,
        window=window,
        stride=stride,
    )

    scalers = {"target": StandardScaler(axis=(0, 1))}
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2)
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
    )
    dm.setup()
    logger.info("Dataset and DataModule are ready.")

    # 3. Load predictor directly from checkpoint
    logger.info("Loading predictor from checkpoint...")
    predictor = CustomPredictor.load_from_checkpoint(checkpoint_path)
    predictor.freeze()

    # 4. Run prediction
    logger.info(f"Running prediction on '{data_split}' split...")
    dataloader = dm.test_dataloader() if data_split == "test" else dm.val_dataloader()
    outputs = trainer.predict(predictor, dataloaders=dataloader)

    # 5. Collate and save results on the main process
    if trainer.is_global_zero:
        logger.info("Collating and processing predictions...")
        predictions = predictor.collate_prediction_outputs(outputs)
        predictions_numpy = torch_to_numpy(predictions)

        y_hat_raw = predictions_numpy["y_hat"]
        y_true_raw = predictions_numpy["y"]
        mask = predictions_numpy.get("mask")

        y_hat_quantized = np.round(y_hat_raw * 10).astype(np.int8)
        y_true_quantized = np.round(y_true_raw * 10).astype(np.int8)

        logger.info(f"Saving predictions to {output_path}")
        np.savez_compressed(
            output_path,
            y_hat=y_hat_quantized,
            y_true=y_true_quantized,
            mask=mask,
        )
        logger.success("Inference complete.")

    if trainer.world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    app()
