from pathlib import Path

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
import typer

from realtime_shindo.config import MODELS_DIR
from realtime_shindo.data import CustomSpatioTemporalDataset
from realtime_shindo.datasets import RealtimeShindo
from realtime_shindo.engines import CustomPredictor
from realtime_shindo.metrics.torch import (
    MaskedWeightedMAELoss,
    compute_lds_weights,
)
from realtime_shindo.nn import custom_models

# PyTorch 2.0+ performance optimization
torch.set_float32_matmul_precision("medium")

app = typer.Typer()


@app.command()
def main(
    net: str = typer.Option("knet", help="Network to use: 'knet' or 'kik'."),
    log_dir: Path = typer.Option(MODELS_DIR, help="Directory for logs and model checkpoints."),
    epochs: int = typer.Option(100, help="Number of training epochs."),
    batch_size: int = typer.Option(128, help="Batch size for training."),
    learning_rate: float = typer.Option(0.001, help="Initial learning rate."),
    gpus: int = typer.Option(
        -1 if torch.cuda.is_available() else 0,
        help="Number of GPUs to use (-1 for all available, 0 for CPU).",
    ),
    # Dataset parameters
    window: int = typer.Option(12, help="Lookback window size."),
    horizon: int = typer.Option(12, help="Prediction horizon."),
    stride: int = typer.Option(1, help="Stride for sliding window sampling."),
    val_len: float = typer.Option(0.1, help="Fraction of data for validation."),
    test_len: float = typer.Option(0.2, help="Fraction of data for testing."),
    # Model hyperparameters
    hidden_size: int = typer.Option(32, help="Hidden size of the model."),
    rnn_layers: int = typer.Option(1, help="Number of RNN layers."),
    gnn_kernel: int = typer.Option(2, help="Kernel size for the GNN."),
):
    """
    Train a spatio-temporal forecasting model on the Realtime Shindo dataset.
    """

    # 1. Dataset and DataModule setup
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
    splitter = TemporalSplitter(val_len=val_len, test_len=test_len)

    num_workers = 4
    if gpus == 0:
        num_workers = 0

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=num_workers,
        pin_memory=True,
    )
    dm.setup()

    # 2. Model instantiation parameters
    model_kwargs = {
        "input_size": dm.n_channels,
        "n_nodes": dm.n_nodes,
        "horizon": horizon,
        "window": window,
        "stride": stride,
        "hidden_size": hidden_size,
        "rnn_layers": rnn_layers,
        "gnn_kernel": gnn_kernel,
    }

    # 3. Loss, metrics, and predictor setup
    y_train = dm.torch_dataset.target
    mask_train = dm.torch_dataset.mask

    if "t" in dm.torch_dataset.patterns.get("target", ""):
        y_train = y_train[dm.train_slice]
    if mask_train is not None and "t" in dm.torch_dataset.patterns.get("mask", ""):
        mask_train = mask_train[dm.train_slice]

    # Pre-compute weights using Label Distribution Smoothing (LDS)
    lds_weights = compute_lds_weights(
        y=y_train,
        mask=mask_train,
        n_bins=70,
        sigma=2.0,
        kernel_size=9,
        scaling_factor=dataset.SCALING_FACTOR,
    )

    loss_fn = MaskedWeightedMAELoss(
        weights=lds_weights, n_bins=70, scaling_factor=dataset.SCALING_FACTOR
    )

    metrics = {
        "mae": MaskedMAE(),
        "mse": MaskedMSE(),
        "mape": MaskedMAPE(),
        f"mae_at_{horizon // 4}": MaskedMAE(at=horizon // 4 - 1),
        f"mae_at_{horizon // 2}": MaskedMAE(at=horizon // 2 - 1),
        f"mae_at_{horizon}": MaskedMAE(at=horizon - 1),
    }

    scheduler_kwargs = {"T_max": epochs, "eta_min": 1e-6}

    predictor = CustomPredictor(
        model_class=custom_models.TimeThenSpaceModel,
        model_kwargs=model_kwargs,
        loss_fn=loss_fn,
        metrics=metrics,
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": learning_rate},
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=scheduler_kwargs,
    )

    # 4. Trainer setup
    tensorboard_logger = TensorBoardLogger(log_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=tensorboard_logger.log_dir,
        save_top_k=1,
        monitor="val_mae",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback],
        precision="bf16-mixed",
        accelerator="gpu" if gpus != 0 else "cpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else "auto",
    )

    # 5. Run training
    trainer.fit(predictor, datamodule=dm)
    logger.success(
        f"Training complete. Best model saved in: {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    app()
