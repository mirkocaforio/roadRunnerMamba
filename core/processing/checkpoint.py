"""
Modular checkpoint management system with separated Saver and Loader
"""

import glob
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .interfaces import TrainerInterface
from .observer import GenericObserver

# ================================ CORE TYPES ================================


class CheckpointType(Enum):
    """Types of checkpoints"""

    BEST = "best"
    PERIODIC = "periodic"
    LATEST = "latest"
    EMERGENCY = "emergency"
    FINAL = "final"
    CUSTOM = "custom"


class LoadType(Enum):
    """Types for loading checkpoints"""

    LATEST_PERIODIC = "latest_periodic"
    BEST_MODEL = "best_model"
    SPECIFIC_EPOCH = "specific_epoch"
    LATEST_ANY = "latest_any"
    CUSTOM_PATTERN = "custom_pattern"


@dataclass
class CheckpointConfig:
    """Unified configuration for checkpoint operations

    Attributes:
        save_dir: Directory to save or load checkpoints
        epoch: Current epoch for periodic checkpoints
        checkpoint_type: Type of checkpoint to save
        filename: Optional custom filename for the checkpoint
        val_metrics: Validation metrics to save with the checkpoint
        message: Optional message to include in the checkpoint
        custom_data: Additional custom data to save with the checkpoint
        load_strategy: Strategy to use for loading checkpoints
        pattern: Custom pattern for loading checkpoints
        strict_loading: Whether to strictly match model architecture when loading
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        load_training_state: Whether to load training state (epoch, history, etc.)

        # Upload configuration
        enable_upload: Whether to enable uploading to external services
        upload_callbacks: List of upload callbacks to execute after saving

        timestamp: Timestamp of the checkpoint operation
    """

    save_dir: str
    epoch: int = 0
    checkpoint_type: CheckpointType = CheckpointType.BEST
    filename: Optional[str] = None
    val_metrics: Dict[str, float] = field(default_factory=dict)
    message: Optional[str] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

    # Load-specific options
    load_strategy: Optional[LoadType] = None
    pattern: Optional[str] = None
    strict_loading: bool = True
    load_optimizer: bool = True
    load_scheduler: bool = True
    load_training_state: bool = True

    # Upload configuration
    enable_upload: bool = False
    upload_callbacks: List["UploadCallback"] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)


class CheckpointData:
    """Immutable checkpoint data container"""

    def __init__(self, **kwargs):
        # Core data
        self.epoch = kwargs.pop("epoch", 0)
        self.model_state_dict = kwargs.pop("model_state_dict", {})
        self.optimizer_state_dict = kwargs.pop("optimizer_state_dict", None)
        self.scheduler_state_dict = kwargs.pop("scheduler_state_dict", None)

        # Training state
        self.val_metrics = kwargs.pop("val_metrics", {})
        self.history = kwargs.pop(
            "history", {"train_loss": [], "val_loss": [], "val_metrics": {}}
        )
        self.best_val_metric = kwargs.pop("best_val_metric", None)
        self.early_stopping_counter = kwargs.pop("early_stopping_counter", 0)

        # Metadata
        self.config = kwargs.pop("config", {})
        self.is_distributed = kwargs.pop("is_distributed", False)
        self.timestamp = kwargs.pop("timestamp", time.time())

        # Additional data
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Deserialize from dictionary"""
        return cls(**data)


# ================================ SAVE STRATEGIES ================================


class GenericCallback(ABC):
    """Base class for generic callbacks"""

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Execute the callback action"""


class CleanupStrategy(GenericCallback):
    """Base strategy for cleaning up old checkpoints"""

    @abstractmethod
    def cleanup(self, save_dir: str, keep_last: int = 3) -> None:
        """Cleanup old checkpoints in the specified directory"""

    def execute(self, *args, **kwargs) -> None:
        """Execute cleanup action"""
        self.cleanup(*args, **kwargs)


class UploadCallback(GenericCallback):
    """Base class for uploading checkpoints to external services"""

    @abstractmethod
    def upload(
        self, filepath: str, config: CheckpointConfig, checkpoint_data: CheckpointData
    ) -> bool:
        """
        Upload checkpoint to external service

        Args:
            filepath: Local path to the checkpoint file
            config: Checkpoint configuration
            checkpoint_data: Checkpoint data container

        Returns:
            bool: True if upload was successful, False otherwise
        """

    def execute(
        self, filepath: str, config: CheckpointConfig, checkpoint_data: CheckpointData
    ) -> None:
        """Execute upload action"""
        self.upload(filepath, config, checkpoint_data)


class PeriodicCleanupStrategy(CleanupStrategy):
    """Cleanup strategy for periodic checkpoints"""

    def __init__(self, logger: Optional[GenericObserver] = None):
        self.logger = logger

    def _log(self, message: str, color: str = "yellow"):
        """Log message using the provided logger"""
        if self.logger:
            self.logger.log_message(message, color)

    def cleanup(self, save_dir: str, keep_last: int):
        """Clean up old periodic checkpoints"""
        try:
            pattern = os.path.join(save_dir, "checkpoint_epoch_*.pt")
            files = glob.glob(pattern)

            epoch_files = []
            for file in files:
                try:
                    filename = os.path.basename(file)
                    epoch = int(
                        filename.replace("checkpoint_epoch_", "").replace(".pt", "")
                    )
                    epoch_files.append((epoch, file))
                except ValueError:
                    continue

            if len(epoch_files) > keep_last:
                epoch_files.sort(reverse=True)
                for _, file_path in epoch_files[keep_last:]:
                    os.remove(file_path)

        except Exception as e:
            self._log(f"Error cleaning up checkpoints: {e}", "yellow")

    def execute(self, save_dir: str, keep_last: int = 3):
        """Execute the cleanup action"""
        self.cleanup(save_dir, keep_last)
        self._log(
            f"Cleaned up old periodic checkpoints, keeping last {keep_last} epochs",
            "green",
        )


class WandbUploadCallback(UploadCallback):
    """Wandb implementation for uploading checkpoints"""

    def __init__(
        self,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        artifact_type: str = "model",
        logger: Optional[GenericObserver] = None,
    ):
        """
        Initialize Wandb upload callback

        Args:
            project_name: Wandb project name (if None, uses current run's project)
            run_name: Run name for the artifact (if None, auto-generated)
            tags: List of tags to apply to the artifact
            notes: Notes/description for the artifact
            artifact_type: Type of artifact (default: "model")
            logger: Logger instance
        """
        self.project_name = project_name
        self.run_name = run_name
        self.tags = tags or []
        self.notes = notes
        self.artifact_type = artifact_type
        self.logger = logger
        self._wandb = None

    def _ensure_wandb(self):
        """Lazy import and initialization of wandb"""
        if self._wandb is None:
            try:
                import wandb

                self._wandb = wandb
                return True
            except ImportError:
                self._log("Wandb not installed. Install with: pip install wandb", "red")
                return False
        return True

    def _log(self, message: str, color: str = "blue"):
        """Log message using the provided logger"""
        if self.logger:
            self.logger.log_message(f"[Wandb] {message}", color)

    def upload(
        self, filepath: str, config: CheckpointConfig, checkpoint_data: CheckpointData
    ) -> bool:
        """
        Upload checkpoint to Wandb as an artifact

        Args:
            filepath: Local path to the checkpoint file
            config: Checkpoint configuration
            checkpoint_data: Checkpoint data container

        Returns:
            bool: True if upload was successful
        """
        if not self._ensure_wandb():
            return False

        try:
            # Check if wandb is initialized
            if not self._wandb.run:
                self._log("No active Wandb run found. Skipping upload.", "yellow")
                return False

            # Create artifact name based on checkpoint type
            artifact_name = self._generate_artifact_name(config, checkpoint_data)

            # Create artifact
            artifact = self._wandb.Artifact(
                name=artifact_name,
                type=self.artifact_type,
                description=self._generate_description(config, checkpoint_data),
                metadata=self._generate_metadata(config, checkpoint_data),
            )

            # Add file to artifact
            filename = os.path.basename(filepath)
            artifact.add_file(filepath, name=filename)

            # Log artifact
            self._wandb.log_artifact(artifact)

            self._log(f"Successfully uploaded {artifact_name} to Wandb", "green")
            return True

        except Exception as e:
            import traceback

            self._log(f"Failed to upload checkpoint to Wandb: {e}", "red")
            self._log(traceback.format_exc(), "red")
            return False

    def _generate_artifact_name(
        self, config: CheckpointConfig, checkpoint_data: CheckpointData
    ) -> str:
        """Generate artifact name based on configuration"""
        base_name = self.run_name or f"model_{config.checkpoint_type.value}"

        if config.checkpoint_type == CheckpointType.PERIODIC:
            return f"{base_name}_epoch_{config.epoch}"
        elif config.checkpoint_type == CheckpointType.LATEST:
            return f"{base_name}_latest"
        elif config.checkpoint_type == CheckpointType.BEST:
            # Include best metric if available
            if (
                hasattr(checkpoint_data, "best_val_metric")
                and checkpoint_data.best_val_metric
            ):
                metric_str = f"_{checkpoint_data.best_val_metric:.4f}".replace(".", "_")
                return f"{base_name}_best{metric_str}"
            return f"{base_name}_best"
        elif config.checkpoint_type == CheckpointType.FINAL:
            return f"{base_name}_final_epoch_{config.epoch}"
        elif config.checkpoint_type == CheckpointType.EMERGENCY:
            return f"{base_name}_emergency_epoch_{config.epoch}"
        else:
            return f"{base_name}_custom_epoch_{config.epoch}"

    def _generate_description(
        self, config: CheckpointConfig, checkpoint_data: CheckpointData
    ) -> str:
        """Generate artifact description"""
        desc_parts = [
            f"Checkpoint from epoch {config.epoch}",
            f"Type: {config.checkpoint_type.value}",
        ]

        if config.val_metrics:
            metrics_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in config.val_metrics.items()]
            )
            desc_parts.append(f"Metrics: {metrics_str}")

        if config.message:
            desc_parts.append(f"Note: {config.message}")

        if self.notes:
            desc_parts.append(self.notes)

        return " | ".join(desc_parts)

    def _generate_metadata(
        self, config: CheckpointConfig, checkpoint_data: CheckpointData
    ) -> Dict[str, Any]:
        """Generate artifact metadata"""
        metadata = {
            "epoch": config.epoch,
            "checkpoint_type": config.checkpoint_type.value,
            "timestamp": config.timestamp,
        }

        # Add validation metrics - ensure all keys are strings and values are proper types
        if config.val_metrics:
            val_metrics_clean = {}
            for key, value in config.val_metrics.items():
                clean_key = str(key)  # Ensure key is string
                if isinstance(value, (int, float)):
                    val_metrics_clean[clean_key] = float(value)
                elif isinstance(value, bool):
                    val_metrics_clean[clean_key] = value
                else:
                    val_metrics_clean[clean_key] = str(value)
            metadata["val_metrics"] = val_metrics_clean

        # Add best metric if available
        if (
            hasattr(checkpoint_data, "best_val_metric")
            and checkpoint_data.best_val_metric is not None
        ):
            metadata["best_val_metric"] = float(checkpoint_data.best_val_metric)

        # Add training state info
        if hasattr(checkpoint_data, "history") and checkpoint_data.history:
            history = checkpoint_data.history
            if hasattr(history, "train_loss") and history.train_loss:
                last_train_loss = history.train_loss[-1] if history.train_loss else None
                if last_train_loss is not None:
                    metadata["current_train_loss"] = float(last_train_loss)
            if hasattr(history, "val_loss") and history.val_loss:
                last_val_loss = history.val_loss[-1] if history.val_loss else None
                if last_val_loss is not None:
                    metadata["current_val_loss"] = float(last_val_loss)

        # Add custom data - sanitize all keys and values
        if config.custom_data:
            custom_data_clean = {}
            for key, value in config.custom_data.items():
                clean_key = str(key)  # Ensure key is string
                if isinstance(value, (int, float)):
                    custom_data_clean[clean_key] = float(value)
                elif isinstance(value, bool):
                    custom_data_clean[clean_key] = value
                elif isinstance(value, str):
                    custom_data_clean[clean_key] = value
                elif isinstance(value, (list, dict)):
                    # Convert complex types to JSON strings
                    import json

                    try:
                        custom_data_clean[clean_key] = json.dumps(value)
                    except (TypeError, ValueError):
                        custom_data_clean[clean_key] = str(value)
                else:
                    custom_data_clean[clean_key] = str(value)
            metadata.update(custom_data_clean)

        return metadata


class CheckpointSaveStrategy(ABC):
    """Base checkpoint strategy"""

    @abstractmethod
    def get_filename(self, config: CheckpointConfig) -> str: ...

    @abstractmethod
    def should_save(
        self, config: CheckpointConfig, trainer_state: Dict[str, Any]
    ) -> bool: ...

    def get_message(self, config: CheckpointConfig, operation: str) -> Tuple[str, str]:
        """Get log message and color"""
        op_text = "saved" if operation == "save" else "loaded"
        return (
            f"{self.__class__.__name__.replace('Strategy', '')} checkpoint {op_text}: epoch {config.epoch}",
            "blue",
        )

    def post_action(
        self, config: CheckpointConfig, callback: Optional[GenericCallback] = None
    ) -> None:
        """Optional post-operation action"""


class BestStrategy(CheckpointSaveStrategy):
    """Strategy for saving the best model based on validation metrics"""

    def get_filename(self, config: CheckpointConfig) -> str:
        return "best_model.pt"

    def should_save(
        self, config: CheckpointConfig, trainer_state: Dict[str, Any]
    ) -> bool:
        return True

    def get_message(self, config: CheckpointConfig, operation: str) -> Tuple[str, str]:
        op_text = "saved" if operation == "save" else "loaded"
        return f"Best model {op_text}", "green"


class LatestStrategy(CheckpointSaveStrategy):
    """Strategy for the latest checkpoint, overwritten every epoch"""

    def get_filename(self, config: CheckpointConfig) -> str:
        return "last_model.pt"

    def should_save(
        self, config: CheckpointConfig, trainer_state: Dict[str, Any]
    ) -> bool:
        return True

    def get_message(self, config: CheckpointConfig, operation: str) -> Tuple[str, str]:
        op_text = "saved" if operation == "save" else "loaded"
        return f"Latest model {op_text}: epoch {config.epoch}", "gray"


class PeriodicStrategy(CheckpointSaveStrategy):
    """Strategy for periodic checkpoints"""

    def __init__(self, frequency: int = 5, keep_last: int = 3):
        self.frequency = frequency
        self.keep_last = keep_last

    def get_filename(self, config: CheckpointConfig) -> str:
        return f"checkpoint_epoch_{config.epoch}.pt"

    def should_save(
        self, config: CheckpointConfig, trainer_state: Dict[str, Any]
    ) -> bool:
        return (config.epoch + 1) % self.frequency == 0

    def get_message(self, config: CheckpointConfig, operation: str) -> Tuple[str, str]:
        op_text = "saved" if operation == "save" else "loaded"
        return f"Periodic checkpoint {op_text}: epoch {config.epoch}", "gray"

    def post_action(
        self,
        config: CheckpointConfig,
        callback: Optional[PeriodicCleanupStrategy] = None,
    ) -> None:
        """Clean up old periodic checkpoints"""
        if callback:
            callback.execute(save_dir=config.save_dir, keep_last=self.keep_last)


class EmergencyStrategy(CheckpointSaveStrategy):
    """Strategy for emergency checkpoints"""

    def get_filename(self, config: CheckpointConfig) -> str:
        return config.filename or f"emergency_epoch_{config.epoch}.pt"

    def should_save(
        self, config: CheckpointConfig, trainer_state: Dict[str, Any]
    ) -> bool:
        return True

    def get_message(self, config: CheckpointConfig, operation: str) -> Tuple[str, str]:
        message = config.message or f"Emergency checkpoint: epoch {config.epoch}"
        return message, "yellow"


class FinalStrategy(CheckpointSaveStrategy):
    """Strategy for final model checkpoint"""

    def get_filename(self, config: CheckpointConfig) -> str:
        return "final_model.pt"

    def should_save(
        self, config: CheckpointConfig, trainer_state: Dict[str, Any]
    ) -> bool:
        return True


class CustomStrategy(CheckpointSaveStrategy):
    """Custom strategy for user-defined checkpoints"""

    def __init__(self, filename_func: Callable[[CheckpointConfig], str]):
        self.filename_func = filename_func

    def get_filename(self, config: CheckpointConfig) -> str:
        return self.filename_func(config)

    def should_save(
        self, config: CheckpointConfig, trainer_state: Dict[str, Any]
    ) -> bool:
        return True


# ================================ LOAD STRATEGIES ================================


class CheckpointLoadStrategy(ABC):
    """Base strategy for loading checkpoints"""

    @abstractmethod
    def find_checkpoint_path(self, config: CheckpointConfig) -> Optional[str]:
        """Find the checkpoint file path based on strategy"""

    def get_message(self, config: CheckpointConfig, filepath: str) -> Tuple[str, str]:
        """Get log message and color for loading"""
        return f"Checkpoint loaded from {os.path.basename(filepath)}", "green"

    def validate_config(self, config: CheckpointConfig) -> bool:
        """Validate configuration for this strategy"""
        return os.path.exists(config.save_dir)


class BestModelLoadStrategy(CheckpointLoadStrategy):
    """Strategy for loading best model checkpoint"""

    def find_checkpoint_path(self, config: CheckpointConfig) -> Optional[str]:
        """Find best model checkpoint"""
        path = os.path.join(config.save_dir, "best_model.pt")
        return path if os.path.exists(path) else None

    def get_message(self, config: CheckpointConfig, filepath: str) -> Tuple[str, str]:
        return "Best model checkpoint loaded", "green"


class LatestPeriodicLoadStrategy(CheckpointLoadStrategy):
    """Strategy for loading latest periodic checkpoint"""

    def find_checkpoint_path(self, config: CheckpointConfig) -> Optional[str]:
        """Find latest periodic checkpoint"""
        files = glob.glob(os.path.join(config.save_dir, "checkpoint_epoch_*.pt"))
        if not files:
            return None

        epoch_files = []
        for file in files:
            try:
                filename = os.path.basename(file)
                epoch = int(
                    filename.replace("checkpoint_epoch_", "").replace(".pt", "")
                )
                epoch_files.append((epoch, file))
            except ValueError:
                continue

        return max(epoch_files)[1] if epoch_files else None

    def get_message(self, config: CheckpointConfig, filepath: str) -> Tuple[str, str]:
        # Extract epoch from filename
        filename = os.path.basename(filepath)
        try:
            epoch = int(filename.replace("checkpoint_epoch_", "").replace(".pt", ""))
            return f"Latest periodic checkpoint loaded: epoch {epoch}", "gray"
        except ValueError:
            return "Latest periodic checkpoint loaded", "gray"


class SpecificEpochLoadStrategy(CheckpointLoadStrategy):
    """Strategy for loading checkpoint from specific epoch"""

    def find_checkpoint_path(self, config: CheckpointConfig) -> Optional[str]:
        """Find checkpoint for specific epoch"""
        if config.epoch is None:
            return None

        path = os.path.join(config.save_dir, f"checkpoint_epoch_{config.epoch}.pt")
        return path if os.path.exists(path) else None

    def validate_config(self, config: CheckpointConfig) -> bool:
        """Validate that epoch is specified"""
        return super().validate_config(config) and config.epoch is not None

    def get_message(self, config: CheckpointConfig, filepath: str) -> Tuple[str, str]:
        return f"Checkpoint loaded for epoch {config.epoch}", "blue"


class LatestAnyLoadStrategy(CheckpointLoadStrategy):
    """Strategy for loading most recent checkpoint of any type"""

    def find_checkpoint_path(self, config: CheckpointConfig) -> Optional[str]:
        """Find most recent checkpoint of any type"""
        patterns = [
            "last_model.pt",
            "checkpoint_epoch_*.pt",
            "best_model.pt",
            "final_model.pt",
            "emergency_*.pt",
        ]
        all_files = []

        for pattern in patterns:
            files = glob.glob(os.path.join(config.save_dir, pattern))
            for file in files:
                try:
                    mtime = os.path.getmtime(file)
                    all_files.append((mtime, file))
                except OSError:
                    continue

        return max(all_files)[1] if all_files else None

    def get_message(self, config: CheckpointConfig, filepath: str) -> Tuple[str, str]:
        return f"Latest checkpoint loaded: {os.path.basename(filepath)}", "blue"


class CustomPatternLoadStrategy(CheckpointLoadStrategy):
    """Strategy for loading checkpoint by custom pattern or specific filename"""

    def find_checkpoint_path(self, config: CheckpointConfig) -> Optional[str]:
        """Find checkpoint by custom pattern or specific filename"""
        if not config.pattern:
            return None

        # Check if pattern is an exact filename
        exact_path = os.path.join(config.save_dir, config.pattern)
        if os.path.exists(exact_path):
            return exact_path

        # Otherwise treat as a glob pattern
        files = glob.glob(os.path.join(config.save_dir, config.pattern))
        return max(files, key=os.path.getmtime) if files else None

    def validate_config(self, config: CheckpointConfig) -> bool:
        """Validate that pattern is specified"""
        return super().validate_config(config) and config.pattern is not None

    def get_message(self, config: CheckpointConfig, filepath: str) -> Tuple[str, str]:
        return f"Checkpoint loaded: {os.path.basename(filepath)}", "cyan"


# ================================ CHECKPOINT SAVER ================================


class CheckpointSaver:
    """Handles checkpoint saving operations"""

    def __init__(self, logger: Optional[GenericObserver] = None):
        self.logger = logger
        self.strategies: Dict[CheckpointType, CheckpointSaveStrategy] = {
            CheckpointType.BEST: BestStrategy(),
            CheckpointType.PERIODIC: PeriodicStrategy(),
            CheckpointType.LATEST: LatestStrategy(),
            CheckpointType.EMERGENCY: EmergencyStrategy(),
            CheckpointType.FINAL: FinalStrategy(),
            CheckpointType.CUSTOM: CustomStrategy(
                lambda c: c.filename or f"custom_{c.epoch}.pt"
            ),
        }

    def register_strategy(
        self, checkpoint_type: CheckpointType, strategy: CheckpointSaveStrategy
    ):
        """Register custom strategy"""
        self.strategies[checkpoint_type] = strategy

    def save(self, config: CheckpointConfig, checkpoint_data: CheckpointData) -> bool:
        """Save checkpoint using appropriate strategy"""
        strategy = self.strategies.get(config.checkpoint_type)
        if not strategy:
            self._log(f"Unknown checkpoint type: {config.checkpoint_type}", "red")
            return False

        # Check if should save
        trainer_state = {"current_epoch": config.epoch}
        if not strategy.should_save(config, trainer_state):
            return False

        try:
            # Generate filepath
            filename = config.filename or strategy.get_filename(config)
            filepath = os.path.join(config.save_dir, filename)

            # Ensure directory exists
            os.makedirs(config.save_dir, exist_ok=True)

            # Save checkpoint
            torch.save(checkpoint_data.to_dict(), filepath)

            # Execute upload callbacks if enabled
            if config.enable_upload and config.upload_callbacks:
                self._execute_upload_callbacks(filepath, config, checkpoint_data)

            # Log success
            message, color = strategy.get_message(config, "save")
            self._log(
                message, color, bold=(config.checkpoint_type == CheckpointType.BEST)
            )

            # Post-save actions
            strategy.post_action(
                config,
                callback=PeriodicCleanupStrategy(logger=self.logger)
                if config.checkpoint_type == CheckpointType.PERIODIC
                else None,
            )

            return True

        except Exception as e:
            self._log(
                f"Error saving {config.checkpoint_type.value} checkpoint: {e}", "red"
            )
            return False

    def _execute_upload_callbacks(
        self, filepath: str, config: CheckpointConfig, checkpoint_data: CheckpointData
    ):
        """Execute all upload callbacks"""
        for callback in config.upload_callbacks:
            try:
                callback.execute(filepath, config, checkpoint_data)
            except Exception as e:
                self._log(
                    f"Upload callback {callback.__class__.__name__} failed: {e}", "red"
                )

    def _log(self, message: str, color: str, bold: bool = False):
        """Log message"""
        if self.logger:
            self.logger.log_message(message, color, bold)


# ================================ CHECKPOINT LOADER ================================


class CheckpointLoader:
    """Handles checkpoint loading operations"""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        logger: Optional[GenericObserver] = None,
    ):
        self.device = device or torch.device("cpu")
        self.logger = logger
        self.strategies: Dict[LoadType, CheckpointLoadStrategy] = {
            LoadType.BEST_MODEL: BestModelLoadStrategy(),
            LoadType.LATEST_PERIODIC: LatestPeriodicLoadStrategy(),
            LoadType.SPECIFIC_EPOCH: SpecificEpochLoadStrategy(),
            LoadType.LATEST_ANY: LatestAnyLoadStrategy(),
            LoadType.CUSTOM_PATTERN: CustomPatternLoadStrategy(),
        }

    def register_strategy(
        self, load_strategy: LoadType, strategy: CheckpointLoadStrategy
    ):
        """Register custom load strategy"""
        self.strategies[load_strategy] = strategy

    def load(self, config: CheckpointConfig) -> Optional[CheckpointData]:
        """Load checkpoint based on configuration"""
        strategy_type = config.load_strategy or LoadType.LATEST_PERIODIC

        strategy = self.strategies.get(strategy_type)
        if not strategy:
            self._log(f"Unknown load strategy: {strategy_type}", "red")
            return None

        # Validate configuration
        if not strategy.validate_config(config):
            self._log(
                f"Invalid configuration for {strategy_type.value} strategy", "red"
            )
            return None

        try:
            filepath = strategy.find_checkpoint_path(config)
            if not filepath:
                self._log(
                    f"No checkpoint found using {strategy_type.value} strategy",
                    "yellow",
                )
                return None

            # Load raw data
            raw_data = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
            checkpoint_data = CheckpointData.from_dict(raw_data)

            message, color = strategy.get_message(config, filepath)
            self._log(message, color)

            return checkpoint_data

        except Exception as e:
            self._log(f"Error loading checkpoint: {e}", "red")
            return None

    def load_model_for_inference(
        self, checkpoint_path: str, model_instance=None, strict: bool = True
    ) -> Optional[torch.nn.Module]:
        """
        Load only the model from a checkpoint for inference purposes.

        Args:
            checkpoint_path (str): Path to the checkpoint file
            model_class: Optional model class to instantiate if needed
            strict (bool): Whether to strictly load the state dict

        Returns:
            torch.nn.Module: The loaded model ready for inference, or None if loading failed
        """
        try:
            if not os.path.exists(checkpoint_path):
                # self._log(f"Checkpoint file not found: {checkpoint_path}", 'red')
                print(f"Checkpoint file not found: {checkpoint_path}")
                return None

            # Load checkpoint data
            # self._log(f"Loading model from checkpoint: {os.path.basename(checkpoint_path)}", 'blue')
            print(f"Loading model from checkpoint: {os.path.basename(checkpoint_path)}")
            raw_data = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Handle different checkpoint formats
            if isinstance(raw_data, dict):
                # Standard checkpoint format with metadata
                if "model_state_dict" in raw_data:
                    model_state_dict = raw_data["model_state_dict"]
                    # self._log("Found model_state_dict in checkpoint", 'green')
                    print("Found model_state_dict in checkpoint")
                elif "state_dict" in raw_data:
                    model_state_dict = raw_data["state_dict"]
                    # self._log("Found state_dict in checkpoint", 'green')
                    print("Found state_dict in checkpoint")
                else:
                    # Assume the entire dict is the model state
                    model_state_dict = raw_data
                    # self._log("Using entire checkpoint as model state", 'yellow')
                    print("Using entire checkpoint as model state")
            else:
                # Direct model object
                if hasattr(raw_data, "state_dict"):
                    model = raw_data
                    model.eval()
                    if torch.cuda.is_available():
                        model = model.cuda()
                    # self._log("Loaded direct model object from checkpoint", 'green')
                    print("Loaded direct model object from checkpoint")
                    return model
                else:
                    # self._log("Invalid checkpoint format", 'red')
                    print("Invalid checkpoint format")
                    return None

            # Fix DDP compatibility (remove 'module.' prefix if present)
            model_state_dict = self._fix_state_dict_compatibility(
                model_state_dict, is_distributed=False
            )

            # If model_class is provided, instantiate it
            if model_instance is not None:
                try:
                    # Check if it's a class, an instance or a factory
                    if isinstance(model_instance, type):
                        # It's a class, instantiate it
                        print(
                            f"Instantiating model from class: {model_instance.__name__}"
                        )
                        model = model_instance()
                    elif isinstance(model_instance, torch.nn.Module):
                        # It's already an instance
                        print(
                            f"Using provided model instance: {type(model_instance).__name__}"
                        )
                        model = model_instance
                    elif callable(model_instance):
                        # Factory receiving the state dict (e.g. to infer the architecture)
                        model = model_instance(model_state_dict)
                        print(f"Model built by factory: {type(model).__name__}")
                    else:
                        raise TypeError(
                            f"Unsupported model_instance type: {type(model_instance)}"
                        )

                    # Rename keys saved by older versions of the model
                    model_state_dict = self._remap_legacy_keys(model_state_dict, model)

                    # model = model_instance
                    incompatible = model.load_state_dict(model_state_dict, strict=False)
                    self._check_incompatible_keys(incompatible, strict=strict)
                    # self._log(f"Model instantiated and loaded with {model_instance}", 'green')
                    print(f"Model instantiated and loaded with {model_instance}")
                except Exception as e:
                    import traceback

                    print(traceback.format_exc())
                    # self._log(f"Error instantiating model with {model_instance}: {e}", 'red')
                    print(f"Error instantiating model with {model_instance}: {e}")
                    return None
            else:
                # Try to reconstruct the model from the checkpoint metadata
                if isinstance(raw_data, dict) and "config" in raw_data:
                    try:
                        # Try to get model info from config
                        config = raw_data["config"]
                        if "model_type" in config:
                            self._log(
                                f"Found model type in config: {config['model_type']}",
                                "blue",
                            )
                        # This would require more specific implementation based on your model architecture
                        # self._log("Model class not provided and cannot reconstruct from checkpoint", 'yellow')
                        print(
                            "Model class not provided and cannot reconstruct from checkpoint"
                        )
                        return None
                    except Exception as e:
                        # self._log(f"Error reading config from checkpoint: {e}", 'yellow')
                        print(f"Error reading config from checkpoint: {e}")
                        return None
                else:
                    # self._log("Model class not provided and no config found in checkpoint", 'yellow')
                    print("Model class not provided and no config found in checkpoint")
                    return None

            # Set model to evaluation mode
            model.eval()

            # Move to appropriate device
            if torch.cuda.is_available():
                model = model.cuda()
                # self._log("Model moved to CUDA", 'green')
                print("Model moved to CUDA")
            else:
                # self._log("Model kept on CPU", 'blue')
                print("Model kept on CPU")

            # self._log("Model successfully loaded for inference", 'green', bold=True)
            print("Model successfully loaded for inference")
            return model

        except Exception as e:
            # self._log(f"Error loading model from checkpoint: {e}", 'red')
            print(f"Error loading model from checkpoint: {e}")
            import traceback

            # self._log(traceback.format_exc(), 'red')
            print(traceback.format_exc())
            return None

    def apply_to_trainer(
        self,
        checkpoint_data: CheckpointData,
        trainer: TrainerInterface,
        config: CheckpointConfig,
    ):
        """Apply checkpoint data to trainer"""
        try:
            # Load model state
            state_dict = self._fix_state_dict_compatibility(
                checkpoint_data.model_state_dict, trainer.is_distributed
            )
            state_dict = self._remap_legacy_keys(state_dict, trainer.model)

            if config.strict_loading:
                trainer.model.load_state_dict(state_dict)
            else:
                trainer.model.load_state_dict(state_dict, strict=False)

            # Load optimizer
            if (
                config.load_optimizer
                and checkpoint_data.optimizer_state_dict
                and trainer.optimizer
            ):
                try:
                    trainer.optimizer.load_state_dict(
                        checkpoint_data.optimizer_state_dict
                    )
                except Exception as e:
                    import traceback

                    self._log(f"Warning: Could not load optimizer state: {e}", "yellow")
                    self._log(traceback.format_exc(), "red")

            # Load scheduler
            if (
                config.load_scheduler
                and checkpoint_data.scheduler_state_dict
                and trainer.scheduler
            ):
                try:
                    trainer.scheduler.load_state_dict(
                        checkpoint_data.scheduler_state_dict
                    )
                except Exception as e:
                    self._log(f"Warning: Could not load scheduler state: {e}", "yellow")

            # Load training state
            if config.load_training_state:
                trainer.start_epoch = checkpoint_data.epoch + 1
                trainer.history = checkpoint_data.history
                trainer.best_val_metric = (
                    checkpoint_data.best_val_metric or trainer.best_val_metric
                )
                # trainer.early_stopping_counter = checkpoint_data.early_stopping_counter

        except Exception as e:
            self._log(f"Error applying checkpoint to trainer: {e}", "red")
            raise

    def _fix_state_dict_compatibility(
        self, state_dict: Dict[str, Any], is_distributed: bool
    ) -> Dict[str, Any]:
        """Fix state dict for DDP compatibility"""
        has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())

        if has_module_prefix and not is_distributed:
            return {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }
        elif not has_module_prefix and is_distributed:
            return {f"module.{key}": value for key, value in state_dict.items()}

        return state_dict

    # Keys renamed across model versions: (legacy pattern, current replacement).
    # - CPE became a module wrapping the depthwise conv in `self.dw`
    # - the block MLP attribute was renamed from `mlp` to `mixer`
    _LEGACY_KEY_RULES = (
        (
            re.compile(
                r"^((?:module\.)?layers\.\d+\.blocks\.\d+)\.cpe([12])\.(weight|bias)$"
            ),
            r"\1.cpe\2.dw.\3",
        ),
        (
            re.compile(r"^((?:module\.)?layers\.\d+\.blocks\.\d+)\.mlp\.mlp\."),
            r"\1.mixer.mlp.",
        ),
    )

    # Keys that may legitimately be unexpected/missing (cached buffers rebuilt at runtime)
    _IGNORED_KEY_SUFFIXES = ("_merge_weight",)

    def _remap_legacy_keys(
        self, state_dict: Dict[str, Any], model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Rename legacy state dict keys to the current model naming.

        A key is renamed only if the model does not know the old name and does
        know the new one, so current checkpoints are never touched.
        """
        model_keys = set(model.state_dict().keys())
        remapped = {}
        renamed = 0
        for key, value in state_dict.items():
            new_key = key
            if key not in model_keys:
                for pattern, replacement in self._LEGACY_KEY_RULES:
                    candidate = pattern.sub(replacement, key)
                    if candidate != key and candidate in model_keys:
                        new_key = candidate
                        break
            if new_key != key:
                renamed += 1
            remapped[new_key] = value

        if renamed:
            print(
                f"Remapped {renamed} legacy checkpoint keys to the current model naming"
            )
            self._log(
                f"Remapped {renamed} legacy checkpoint keys to the current model naming",
                "yellow",
            )
        return remapped

    def _check_incompatible_keys(self, incompatible, strict: bool = True):
        """Report missing/unexpected keys; raise if strict and something relevant is missing."""
        missing = [
            k
            for k in incompatible.missing_keys
            if not k.endswith(self._IGNORED_KEY_SUFFIXES)
        ]
        unexpected = [
            k
            for k in incompatible.unexpected_keys
            if not k.endswith(self._IGNORED_KEY_SUFFIXES)
        ]

        if missing:
            print(
                f"WARNING: {len(missing)} model keys NOT found in checkpoint (left randomly initialized):"
            )
            for k in missing[:20]:
                print(f"  - {k}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        if unexpected:
            print(f"WARNING: {len(unexpected)} checkpoint keys not used by the model:")
            for k in unexpected[:20]:
                print(f"  - {k}")
            if len(unexpected) > 20:
                print(f"  ... and {len(unexpected) - 20} more")
        if not missing and not unexpected:
            print("All checkpoint keys matched the model")

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Checkpoint/model mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys. "
                "Check the architecture flags (e.g. FG mask, condition branch, mixer type)."
            )

    def _log(self, message: str, color: str, bold: bool = False):
        """Log message"""
        if self.logger:
            self.logger.log_message(message, color, bold)


# ================================ DATA BUILDER ================================


class CheckpointDataBuilder:
    """Builds checkpoint data from trainer state"""

    @staticmethod
    def build_from_trainer(
        trainer: TrainerInterface, config: CheckpointConfig
    ) -> CheckpointData:
        """Build checkpoint data from trainer"""
        # Extract model state
        if trainer.is_distributed:
            model_state = trainer.model.module.state_dict()  # type: ignore
        else:
            model_state = trainer.model.state_dict()

        # Build data dictionary
        data = {
            "epoch": config.epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": trainer.optimizer.state_dict()
            if trainer.optimizer
            else None,
            "scheduler_state_dict": trainer.scheduler.state_dict()
            if trainer.scheduler
            else None,
            "val_metrics": config.val_metrics,
            "history": trainer.history,
            "best_val_metric": trainer.best_val_metric,
            "early_stopping_counter": trainer.early_stopping_counter,
            "config": trainer.config.to_dict()
            if hasattr(trainer.config, "to_dict")
            else {},
            "is_distributed": trainer.is_distributed,
            **config.custom_data,
        }

        return CheckpointData(**data)


# ================================ CHECKPOINT MANAGER ================================


class CheckpointManager:
    """Unified checkpoint management with separated Saver and Loader"""

    checkpoint_patterns = {
        "periodic": "checkpoint_epoch_*.pt",
        "latest": "last_model.pt",
        "best": "best_model.pt",
        "final": "final_model.pt",
        "emergency": "emergency_*.pt",
    }

    def __init__(
        self,
        trainer: Optional[TrainerInterface] = None,
        device: Optional[torch.device] = None,
        logger: Optional[GenericObserver] = None,
    ):
        self.trainer = trainer
        self.device = device or (trainer.device if trainer else torch.device("cpu"))
        self.logger = logger

        # Initialize separate components
        self.saver = CheckpointSaver(logger=logger)
        self.loader = CheckpointLoader(device=self.device, logger=logger)

        # Configure periodic strategy with trainer settings
        if trainer and trainer.config:
            frequency = trainer.config.checkpoint_frequency
            self.saver.register_strategy(
                CheckpointType.PERIODIC, PeriodicStrategy(frequency, keep_last=3)
            )

            # Auto-configure Wandb upload if enabled in trainer config
            self._auto_configure_wandb_upload(trainer.config)

    def _auto_configure_wandb_upload(self, config):
        """Auto-configure Wandb upload based on trainer config"""
        if (
            hasattr(config, "wandb_config")
            and config.wandb_config.enabled
            and config.wandb_config.upload_checkpoints
        ):
            try:
                self.configure_wandb_upload(
                    project_name=config.wandb_config.project,
                    run_name=config.wandb_config.name,
                    tags=config.wandb_config.tags,
                    notes=config.wandb_config.notes,
                    artifact_type=config.wandb_config.artifact_type,
                )

                if self.logger:
                    self.logger.log_message(
                        "Wandb checkpoint upload auto-configured", "green"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log_message(
                        f"Warning: Failed to auto-configure Wandb upload: {e}", "yellow"
                    )

    def register_save_strategy(
        self, checkpoint_type: CheckpointType, strategy: CheckpointSaveStrategy
    ):
        """Register custom save strategy"""
        self.saver.register_strategy(checkpoint_type, strategy)

    def register_load_strategy(
        self, load_strategy: LoadType, strategy: CheckpointLoadStrategy
    ):
        """Register custom load strategy"""
        self.loader.register_strategy(load_strategy, strategy)

    def save(self, config: CheckpointConfig) -> bool:
        """Save checkpoint"""
        if not self.trainer:
            raise ValueError("Trainer required for saving")

        # Build checkpoint data
        checkpoint_data = CheckpointDataBuilder.build_from_trainer(
            trainer=self.trainer, config=config
        )

        # Save using saver
        success = self.saver.save(config, checkpoint_data)

        # Update trainer state for best model
        if success and config.checkpoint_type == CheckpointType.BEST:
            # Get the metric type and update best metric
            if hasattr(self.trainer, "best_metric_type") and config.val_metrics:
                current_metric = config.val_metrics.get(self.trainer.best_metric_type)
                if current_metric is not None:
                    self.trainer.best_val_metric = current_metric

        return success

    def load(
        self, config: CheckpointConfig, apply_to_trainer: bool = True
    ) -> Optional[CheckpointData]:
        """Load checkpoint"""
        checkpoint_data = self.loader.load(config)

        if checkpoint_data and apply_to_trainer and self.trainer:
            self.loader.apply_to_trainer(checkpoint_data, self.trainer, config)

        return checkpoint_data

    def auto_resume(self, save_dir: str) -> bool:
        """Auto-resume from best available checkpoint"""
        # Strategies to try (LATEST_ANY first so we resume from the most recent
        # checkpoint, which is last_model.pt saved every epoch)
        strategies = [
            LoadType.LATEST_ANY,
            LoadType.BEST_MODEL,
            LoadType.LATEST_PERIODIC,
        ]

        for strategy in strategies:
            config = CheckpointConfig(
                save_dir=save_dir, load_strategy=strategy, strict_loading=False
            )

            checkpoint_data = self.load(config)
            if checkpoint_data:
                return True

        return False

    def list_checkpoints(self, save_dir: str) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        checkpoints = []
        if not os.path.exists(save_dir):
            return checkpoints

        patterns = self.checkpoint_patterns

        for cp_type, pattern in patterns.items():
            files = glob.glob(os.path.join(save_dir, pattern))
            for file in files:
                try:
                    epoch = None
                    filename = os.path.basename(file)

                    if "epoch_" in filename:
                        try:
                            epoch = int(filename.split("epoch_")[1].split(".")[0])
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid epoch in filename: {filename}"
                            ) from e

                    stat = os.stat(file)
                    checkpoints.append(
                        {
                            "path": file,
                            "type": cp_type,
                            "epoch": epoch,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "filename": filename,
                        }
                    )
                except Exception:
                    continue

        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints

    def configure_wandb_upload(
        self,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        artifact_type: str = "model",
    ) -> "CheckpointManager":
        """
        Configure Wandb upload for this checkpoint manager

        Args:
            project_name: Wandb project name
            run_name: Run name for artifacts
            tags: Tags to apply to artifacts
            notes: Notes for artifacts
            artifact_type: Type of artifacts

        Returns:
            Self for method chaining
        """
        self.wandb_callback = WandbUploadCallback(
            project_name=project_name,
            run_name=run_name,
            tags=tags,
            notes=notes,
            artifact_type=artifact_type,
            logger=self.logger,
        )
        return self

    def enable_wandb_upload(self, config: CheckpointConfig) -> CheckpointConfig:
        """
        Enable Wandb upload for a specific checkpoint configuration

        Args:
            config: Checkpoint configuration to modify

        Returns:
            Modified configuration with Wandb upload enabled
        """
        if hasattr(self, "wandb_callback"):
            config.enable_upload = True
            if self.wandb_callback not in config.upload_callbacks:
                config.upload_callbacks.append(self.wandb_callback)
        else:
            if self.logger:
                self.logger.log_message(
                    "Wandb callback not configured. Call configure_wandb_upload() first.",
                    "yellow",
                )

        return config

    def add_upload_callback(self, callback: "UploadCallback") -> "CheckpointManager":
        """
        Add a custom upload callback

        Args:
            callback: Upload callback to add

        Returns:
            Self for method chaining
        """
        if not hasattr(self, "upload_callbacks"):
            self.upload_callbacks = []
        self.upload_callbacks.append(callback)
        return self

    # Pre confezionati
    def save_best(
        self,
        epoch: int,
        save_dir: str,
        val_metrics: Dict[str, float] | None = None,
        enable_upload: bool = False,
    ) -> bool:
        # Auto-enable upload if wandb is configured
        if not enable_upload and hasattr(self, "wandb_callback"):
            enable_upload = True

        config = CheckpointConfig(
            save_dir=save_dir,
            epoch=epoch,
            checkpoint_type=CheckpointType.BEST,
            val_metrics=val_metrics or {},
            enable_upload=enable_upload,
        )
        if enable_upload:
            config = self.enable_wandb_upload(config)
        return self.save(config)

    def save_periodic(
        self,
        epoch: int,
        save_dir: str,
        val_metrics: Dict[str, float] | None = None,
        enable_upload: bool = False,
    ) -> bool:
        # Auto-enable upload if wandb is configured
        if not enable_upload and hasattr(self, "wandb_callback"):
            enable_upload = True

        config = CheckpointConfig(
            save_dir=save_dir,
            epoch=epoch,
            checkpoint_type=CheckpointType.PERIODIC,
            val_metrics=val_metrics or {},
            enable_upload=enable_upload,
        )
        if enable_upload:
            config = self.enable_wandb_upload(config)
        return self.save(config)

    def save_latest(
        self,
        epoch: int,
        save_dir: str,
        val_metrics: Dict[str, float] | None = None,
        enable_upload: bool = False,
    ) -> bool:
        # Auto-enable upload if wandb is configured
        if not enable_upload and hasattr(self, "wandb_callback"):
            enable_upload = True

        config = CheckpointConfig(
            save_dir=save_dir,
            epoch=epoch,
            checkpoint_type=CheckpointType.LATEST,
            val_metrics=val_metrics or {},
            enable_upload=enable_upload,
        )
        if enable_upload:
            config = self.enable_wandb_upload(config)
        return self.save(config)

    def save_emergency(
        self,
        epoch: int,
        save_dir: str,
        filename: str | None = None,
        message: str | None = None,
        val_metrics: Dict[str, float] | None = None,
        enable_upload: bool = False,
    ) -> bool:
        # Auto-enable upload if wandb is configured
        if not enable_upload and hasattr(self, "wandb_callback"):
            enable_upload = True

        config = CheckpointConfig(
            save_dir=save_dir,
            epoch=epoch,
            checkpoint_type=CheckpointType.EMERGENCY,
            filename=filename,
            message=message,
            val_metrics=val_metrics or {},
            enable_upload=enable_upload,
        )
        if enable_upload:
            config = self.enable_wandb_upload(config)
        return self.save(config)

    def load_best(self, save_dir: str, strict: bool = True) -> Optional[CheckpointData]:
        config = CheckpointConfig(
            save_dir=save_dir, load_strategy=LoadType.BEST_MODEL, strict_loading=strict
        )
        return self.load(config)

    def load_latest(
        self, save_dir: str, strict: bool = True
    ) -> Optional[CheckpointData]:
        config = CheckpointConfig(
            save_dir=save_dir, load_strategy=LoadType.LATEST_ANY, strict_loading=strict
        )
        return self.load(config)

    def load_epoch(
        self, save_dir: str, epoch: int, strict: bool = True
    ) -> Optional[CheckpointData]:
        config = CheckpointConfig(
            save_dir=save_dir,
            epoch=epoch,
            load_strategy=LoadType.SPECIFIC_EPOCH,
            strict_loading=strict,
        )
        return self.load(config)

    def load_from_file(
        self, checkpoint_path: str, load_training_state: bool = True
    ) -> bool:
        """Load checkpoint from specific file path"""
        if not os.path.exists(checkpoint_path):
            return False

        try:
            # Determine strategy based on filename
            filename = os.path.basename(checkpoint_path)
            save_dir = os.path.dirname(checkpoint_path)

            # Automatically determine load strategy based on filename patterns

            if "best_model" in filename:
                config = CheckpointConfig(
                    save_dir=save_dir,
                    load_strategy=LoadType.BEST_MODEL,
                    strict_loading=False,
                    load_training_state=load_training_state,
                )
            elif "checkpoint_epoch_" in filename:
                # Extract epoch for specific loading
                try:
                    epoch_str = filename.replace("checkpoint_epoch_", "").replace(
                        ".pt", ""
                    )
                    epoch = int(epoch_str)
                    config = CheckpointConfig(
                        save_dir=save_dir,
                        epoch=epoch,
                        load_strategy=LoadType.SPECIFIC_EPOCH,
                        strict_loading=False,
                        load_training_state=load_training_state,
                    )
                except ValueError:
                    # Fallback to custom pattern
                    config = CheckpointConfig(
                        save_dir=save_dir,
                        load_strategy=LoadType.CUSTOM_PATTERN,
                        pattern=filename,
                        strict_loading=False,
                        load_training_state=load_training_state,
                    )
            elif "final_model" in filename:
                config = CheckpointConfig(
                    save_dir=save_dir,
                    load_strategy=LoadType.CUSTOM_PATTERN,
                    pattern=filename,
                    strict_loading=False,
                    load_training_state=load_training_state,
                )
            elif "emergency" in filename:
                config = CheckpointConfig(
                    save_dir=save_dir,
                    load_strategy=LoadType.CUSTOM_PATTERN,
                    pattern=filename,
                    strict_loading=False,
                    load_training_state=load_training_state,
                )
            else:
                # Use custom pattern for any other file
                config = CheckpointConfig(
                    save_dir=save_dir,
                    load_strategy=LoadType.CUSTOM_PATTERN,
                    pattern=filename,
                    strict_loading=False,
                    load_training_state=load_training_state,
                )

            # Load checkpoint
            checkpoint_data = self.load(config, apply_to_trainer=True)
            return checkpoint_data is not None

        except Exception as e:
            if self.logger:
                self.logger.log_message(
                    f"Error loading checkpoint from {checkpoint_path}: {e}", "red"
                )
            return False
