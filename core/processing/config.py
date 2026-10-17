import os
from typing import Dict, Any, Optional
from torch import device


class DistributedConfig:
    """Configuration for DistributedDataParallel training"""

    def __init__(self,
                 backend: str = "nccl",
                 init_method: str = "env://",
                 world_size: int = 1,
                 rank: int = 0,
                 local_rank: int = 0,
                 find_unused_parameters: bool = False,
                 gradient_as_bucket_view: bool = True,
                 static_graph: bool = False):
        """
        Initialize DistributedDataParallel configuration.
        
        Args:
            backend: Backend to use for distributed training ('nccl', 'gloo', 'mpi')
            init_method: Method to initialize the process group
            world_size: Total number of processes participating in the job
            rank: Rank of the current process
            local_rank: Local rank of the current process on the current node
            find_unused_parameters: Whether to find unused parameters
            gradient_as_bucket_view: Whether to use gradient as bucket view
            static_graph: Whether the graph is static (for optimization)
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph

    def auto_detect_from_env(self):
        """Auto-detect distributed configuration from environment variables"""
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return self

    def is_distributed(self) -> bool:
        """Check if distributed training is enabled"""
        return self.world_size is not None and self.world_size > 1

    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0 if self.rank is not None else True

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {key: getattr(self, key) for key in vars(self) if not key.startswith('_')}

    def __str__(self):
        """String representation of the configuration"""
        return f"DistributedConfig(backend={self.backend}, world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank})"


class TrainingConfig:
    """
    Handles training configuration parameters.
    """

    def __init__(self, epochs: int = 50, learning_rate: float = 1e-3, weight_decay: float = 1e-5,
                 patience: int = 10, save_dir: str = './checkpoints',
                 export_model: bool = True,
                 export_onnx: bool = False,
                 export_torchscript: bool = False,
                 better_init: bool = False,
                 optimizer_type: str = 'adamw',
                 use_distributed: bool = False,
                 distributed_config: Optional[DistributedConfig] = None,
                 main_device: Optional[device] = None,
                 max_grad_norm: float = 0.0,
                 stop_on_error: bool = False,
                 stop_on_batch_error: bool = False,
                 auto_resume: bool = True,
                 checkpoint_frequency: int = 5,
                 use_amp: bool = False):
        """
        Initialize training configuration parameters.

        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        :param weight_decay: Weight decay for the optimizer
        :param patience: Early stopping patience
        :param save_dir: Directory to save model checkpoints
        :param export_model: Whether to export the model after training
        :param export_onnx: Whether to export the model to ONNX format
        :param export_torchscript: Whether to export the model to TorchScript format
        :param better_init: Whether to use better initialization for model parameters
        :param optimizer_type: Type of optimizer to use (e.g., 'adamw', 'sgd')
        :param use_distributed: Whether to use DistributedDataParallel for multi-GPU training
        :param distributed_config: Configuration for distributed training
        :param main_device: Main device for training
        :param max_grad_norm: Maximum gradient norm for clipping
        :param stop_on_error: Whether to stop training on error in general
        :param stop_on_batch_error: Whether to stop training on batch error only
        :param auto_resume: Whether to automatically resume from the last checkpoint
        :param checkpoint_frequency: Deprecated. No longer controls checkpointing:
            the latest checkpoint (last_model.pt) is now saved every epoch and the
            best checkpoint (best_model.pt) whenever the monitored metric improves.
        :param use_amp: Whether to use Automatic Mixed Precision (bfloat16) during training
        """
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._patience = patience
        self._save_dir = save_dir
        self._export_model = export_model
        self._export_onnx = export_onnx
        self._export_torchscript = export_torchscript
        self._better_init = better_init
        self._optimizer_type = optimizer_type
        self._use_distributed = use_distributed
        self._distributed_config = distributed_config or DistributedConfig()
        self._main_device = main_device
        self._max_grad_norm = max_grad_norm
        self._stop_on_error = stop_on_error
        self._stop_on_batch_error = stop_on_batch_error
        self._auto_resume = auto_resume
        self._checkpoint_frequency = checkpoint_frequency
        self._use_amp = use_amp

    @property
    def epochs(self) -> int:
        """Number of training epochs"""
        return self._epochs

    @epochs.setter
    def epochs(self, value: int):
        """Set number of training epochs"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Epochs must be a positive integer")
        self._epochs = value

    @property
    def learning_rate(self) -> float:
        """Learning rate for the optimizer"""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        """Set learning rate for the optimizer"""
        if not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("Learning rate must be a positive number")
        self._learning_rate = float(value)

    @property
    def weight_decay(self) -> float:
        """Weight decay for the optimizer"""
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, value: float):
        """Set weight decay for the optimizer"""
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("Weight decay must be a non-negative number")
        self._weight_decay = float(value)

    @property
    def patience(self) -> int:
        """Early stopping patience"""
        return self._patience

    @patience.setter
    def patience(self, value: int):
        """Set early stopping patience"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Patience must be a positive integer")
        self._patience = value

    @property
    def save_dir(self) -> str:
        """Directory to save model checkpoints"""
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value: str):
        """Set directory to save model checkpoints"""
        self._save_dir = value

    @property
    def better_init(self) -> bool:
        """Whether to use better initialization for model parameters"""
        return self._better_init

    @better_init.setter
    def better_init(self, value: bool):
        """Set whether to use better initialization for model parameters"""
        self._better_init = value

    @property
    def optimizer_type(self) -> str:
        """Type of optimizer to use"""
        return self._optimizer_type

    @optimizer_type.setter
    def optimizer_type(self, value: str):
        """Set type of optimizer to use"""
        self._optimizer_type = value.lower()

    @property
    def use_distributed(self) -> bool:
        """Whether to use DistributedDataParallel for multi-GPU training"""
        return self._use_distributed

    @use_distributed.setter
    def use_distributed(self, value: bool):
        """Set whether to use DistributedDataParallel for multi-GPU training"""
        self._use_distributed = value

    @property
    def distributed_config(self) -> DistributedConfig:
        """Configuration for distributed training"""
        return self._distributed_config

    @distributed_config.setter
    def distributed_config(self, value: DistributedConfig):
        """Set configuration for distributed training"""
        self._distributed_config = value

    @property
    def max_grad_norm(self) -> float:
        """Maximum gradient norm for clipping"""
        return self._max_grad_norm

    @max_grad_norm.setter
    def max_grad_norm(self, value: float):
        """Set maximum gradient norm for clipping"""
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("Max gradient norm must be a non-negative number")
        self._max_grad_norm = float(value)

    @property
    def stop_on_error(self) -> bool:
        """Whether to stop training on error"""
        return self._stop_on_error

    @stop_on_error.setter
    def stop_on_error(self, value: bool):
        """Set whether to stop training on error"""
        self._stop_on_error = value

    @property
    def stop_on_batch_error(self) -> bool:
        """Whether to stop training on batch error"""
        return self._stop_on_batch_error

    @stop_on_batch_error.setter
    def stop_on_batch_error(self, value: bool):
        """Set whether to stop training on batch error"""
        self._stop_on_batch_error = value

    @property
    def export_model(self) -> bool:
        """Whether to export the model after training"""
        return self._export_model

    @export_model.setter
    def export_model(self, value: bool):
        """Set whether to export the model after training"""
        self._export_model = value

    @property
    def export_onnx(self) -> bool:
        """Whether to export the model to ONNX format"""
        return self._export_onnx

    @export_onnx.setter
    def export_onnx(self, value: bool):
        """Set whether to export the model to ONNX format"""
        self._export_onnx = value

    @property
    def export_torchscript(self) -> bool:
        """Whether to export the model to TorchScript format"""
        return self._export_torchscript

    @export_torchscript.setter
    def export_torchscript(self, value: bool):
        """Set whether to export the model to TorchScript format"""
        self._export_torchscript = value

    @property
    def main_device(self) -> Optional[device]:
        """Main device for training"""
        return self._main_device

    @main_device.setter
    def main_device(self, value: device):
        """Set main device for training"""
        self._main_device = value

    @property
    def auto_resume(self) -> bool:
        """Whether to automatically resume from the last checkpoint"""
        return self._auto_resume

    @auto_resume.setter
    def auto_resume(self, value: bool):
        """Set whether to automatically resume from the last checkpoint"""
        self._auto_resume = value

    @property
    def checkpoint_frequency(self) -> int:
        """Deprecated. No longer used: the latest checkpoint is saved every epoch
        and the best checkpoint whenever the monitored metric improves."""
        return self._checkpoint_frequency

    @checkpoint_frequency.setter
    def checkpoint_frequency(self, value: int):
        """Set how often to save checkpoints (every N epochs)"""
        if not isinstance(value, int) or value < 1:
            raise ValueError("Checkpoint frequency must be a positive integer")
        self._checkpoint_frequency = value

    @property
    def use_amp(self) -> bool:
        """Whether to use Automatic Mixed Precision (bfloat16) during training"""
        return self._use_amp

    @use_amp.setter
    def use_amp(self, value: bool):
        """Set whether to use Automatic Mixed Precision during training"""
        self._use_amp = bool(value)

    def _pretty_print(self):
        """Pretty print the training configuration"""
        _str = "Training Configuration:\n"
        for key, value in vars(self).items():
            _str += f"{key.replace('_', ' ').capitalize()}: {value}\n"
        return _str

    def __str__(self):
        """String representation of the training configuration"""
        return self._pretty_print()

    def set_config(self, **kwargs):
        """Set training configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter value"""
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {key.replace('_', ''): getattr(self, key) for key in vars(self) if key.startswith('_')
                and not (key.endswith('config') or key.endswith('device'))}

    def from_dict(self, config: Dict[str, Any]):
        """Load configuration from dictionary"""
        self.set_config(**config)
        return self
