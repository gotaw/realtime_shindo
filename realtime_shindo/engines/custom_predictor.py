from tsl.engines import Predictor


class CustomPredictor(Predictor):
    def log_metrics(self, metrics, **kwargs):
        kwargs.setdefault("sync_dist", True)
        super().log_metrics(metrics, **kwargs)

    def log_loss(self, name: str, loss, **kwargs):
        kwargs.setdefault("sync_dist", True)
        super().log_loss(name, loss, **kwargs)
