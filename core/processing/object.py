from typing import Optional, Dict, Any, List


class HistoryDataObject:
    """Class to store history data for training"""
    train_loss: List[float] = []
    val_loss: List[float] = []
    val_metrics: Dict[str, float] = {}

    def __getitem__(self, key: str) -> Any:
        """Get item by key"""
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key"""
        setattr(self, key, value)

    def _pretty_print(self) -> str:
        """Pretty print the history data"""
        _str = "History Data:\n"
        _str += f"Train Loss: {self.train_loss}\n"
        _str += f"Validation Loss: {self.val_loss}\n"
        if self.val_metrics:
            _str += "Validation Metrics:\n"
            for key, value in self.val_metrics.items():
                _str += f"\t - {key}: {value}\n"
        return _str

    def __repr__(self):
        return self._pretty_print()

    def __str__(self):
        """String representation of the history data"""
        return self._pretty_print()


class TrainingStatus:
    """Possible training statuses"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def is_valid(cls, status: str) -> bool:
        """Check if the status is valid"""
        return status in vars(cls).values()

    @classmethod
    def get_all_statuses(cls) -> list:
        """Get all valid training statuses"""
        return [value for value in vars(cls).values() if isinstance(value, str) and not value.startswith('_')]


class TrainingResult:
    """Class to store training results"""

    def __init__(self, history: HistoryDataObject, metrics: Dict[str, Any],
                 best_metric: Optional[float] = None, metric_type: Optional[str] = None,
                 status: str = TrainingStatus.NOT_STARTED, message: Optional[str] = None):
        """
        Initialize the training result with history and metrics.
        :param history: Dictionary containing training history (e.g., loss, accuracy)
        :param metrics: Dictionary containing final metrics (e.g., validation loss, accuracy)
        :param best_metric: Best validation metric value achieved during training
        :param metric_type: Type of the best metric (e.g., 'accuracy', 'loss')
        """
        self.history = history
        self.metrics = metrics
        self.best_metric = best_metric
        self.metric_type = metric_type
        self.status = status
        self.message = message
        if not TrainingStatus.is_valid(status):
            raise ValueError(
                f"Invalid training status: {status}. Valid statuses are: {', '.join(TrainingStatus.__dict__.values())}")

    def _pretty_print(self) -> str:
        """Pretty print the training result"""
        _str = "Training Result:\n"
        _str += f"Status: {self.status}\n"
        _str += f"Best Validation Metric: {self.best_metric} ({self.metric_type})\n"
        _str += "Metrics:\n"

        for key, value in self.metrics.items():
            _str += f"\t - {key}: {value}\n"

        _str += f"Message: {self.message}\n"
        return _str

    def __str__(self):
        """String representation of the training result"""
        return self._pretty_print()

    def __repr__(self):
        return (f"TrainingResult(history={self.history}, metrics={self.metrics}, best_metric={self.best_metric},"
                f" metric_type={self.metric_type}, status={self.status}, message={self.message})")


class MetricType:
    """Class for metric types"""
    ACCURACY = "accuracy"
    LOSS = "loss"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"
    MAE = "mae"
    MSE = "mse"
    PSNR = "psnr"
    SSIM = "ssim"

    @classmethod
    def get_limit(cls, metric: str) -> float:
        """Get the limit for the metric type"""
        if metric in [cls.ACCURACY, cls.PSNR, cls.SSIM]:
            return float('-inf')
        if metric in [cls.LOSS, cls.MAE, cls.MSE]:
            return float('inf')
        if metric in [cls.F1_SCORE, cls.PRECISION, cls.RECALL, cls.AUC]:
            return 1.0

        raise ValueError(f"Unknown metric type: {metric}")

    @classmethod
    def is_valid(cls, metric: str) -> bool:
        """Check if the metric type is valid"""
        return metric in vars(cls).values()
