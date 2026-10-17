"""
Experiment configuration and management system
"""
from copy import deepcopy
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Type, Any
import os
from datetime import datetime
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

from core.model.net import RSCDMamba
from core.utility.to_dict import dataclass_to_dict

from core.processing.loss import LossType
from core.processing.config import TrainingConfig, DistributedConfig
from core.processing.scheduler import SchedulerConfig
from core.processing.observer import ObserverList, ConsoleLogger, MetricsPlotter, GenericObserver, WandbObserver
from core.processing.builder import TrainerBuilder
from core.processing.training import ModelTrainer
from core.processing.checkpoint import CheckpointManager, WandbUploadCallback


# ================================ DATA CLASSES ================================

class BaseModelConfig(ABC):
    """Base class for all model configurations"""

    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration parameters"""
        pass

    @abstractmethod
    def to_dict(self, exclude_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Get the model type this config is for"""
        pass


@dataclass
class RSCDMambaModelConfig(BaseModelConfig):
    """Configuration for RSCD_Mamba model"""
    img_size: int = 224 # Default image size 
    patch_size: int = 4 # Default patch size
    in_chans: int = 3 # Default input channels
    num_classes: int = 1000 # Default number of classes
    depths: List[int] = field(default_factory=lambda: [2, 2, 9, 2])  # Default depths for each stage
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])  # Default dimensions for each stage
    d_state: int = 1
    dt_init: str = "random"
    mlp_ratio: float = 4.0
    drop_rate: float = 0.  # Default drop rate
    attn_drop_rate: float = 0.  # Default attention drop rate
    drop_path_rate: float = 0.1  # Default drop path rate
    norm_layer: Type = torch.nn.LayerNorm
    patch_norm: bool = True
    use_checkpoint: bool = False
    use_fg_mask: bool = True
    use_condition_branch: bool = True
    num_conditions: int = 6          # Numero di condizioni da predire (per RSCD completo: 6 + ice separato)
    condition_hidden_dim: Optional[int] = None  # Dimensione layer nascosto del branch condizione (default d_model//4)
    condition_dropout: float = 0.1  # Dropout nel branch condizione

    def validate(self) -> bool:
        """Validate RSCD_Mamba specific parameters"""
        if self.drop_rate < 0 or self.drop_rate > 1:
            return False
        return True

    def get_model_type(self) -> str:
        return "rscd_mamba"

    def to_dict(self, exclude_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        return dataclass_to_dict(self, exclude_fields=exclude_fields)

class ModelConfigFactory:
    """Factory for creating model configurations"""

    _configs: Dict[str, Type[BaseModelConfig]] = {
        "rscd_mamba": RSCDMambaModelConfig
    }

    @classmethod
    def create_config(cls, model_type: str, **kwargs) -> BaseModelConfig:
        """Create a model configuration"""
        if model_type.lower() not in cls._configs:
            raise ValueError(f"Unknown model type: {model_type}")

        config_class = cls._configs[model_type.lower()]
        config = config_class(**kwargs)

        if not config.validate():
            raise ValueError(f"Invalid configuration for {model_type}")

        return config

    @classmethod
    def register_config(cls, model_type: str, config_class: Type[BaseModelConfig]):
        """Register a new configuration type"""
        cls._configs[model_type.lower()] = config_class

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get a list of available model types"""
        return list(cls._configs.keys())


@dataclass
class DataConfig:
    """Configuration for data loading
    Attributes:
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the dataset
        train_dataset: Training dataset instance
        val_dataset: Validation dataset instance
        test_dataset: Test dataset instance (optional)
        """
    batch_size: int = 128
    num_workers: int = 4
    shuffle: bool = True

    # Dataset instances (will be set externally)
    train_dataset: Optional[Dataset] = None
    sampler: Optional[torch.utils.data.Sampler] = None
    val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None

    def to_dict(self, exclude_fields: Optional[Set[str]] = None) -> dict:
        """Convert to dictionary representation"""
        # By default, exclude dataset instances and sampler objects as they're not serializable
        if exclude_fields is None:
            exclude_fields = {'train_dataset', 'val_dataset', 'test_dataset', 'sampler'}
        else:
            exclude_fields = exclude_fields.union({'train_dataset', 'val_dataset', 'test_dataset', 'sampler'})
        return dataclass_to_dict(self, exclude_fields=exclude_fields)  # type: ignore[return]


@dataclass
class WandbConfig:
    """Configuration for Wandb integration
    
    Attributes:
        enabled: Whether to enable Wandb logging
        project: Wandb project name
        entity: Wandb entity (team) name
        name: Run name (if None, will use experiment name)
        tags: List of tags for the run
        notes: Notes for the run
        save_code: Whether to save code to Wandb
        upload_checkpoints: Whether to upload checkpoints as artifacts
        artifact_type: Type of artifact for checkpoints
        log_gradients: Whether to log gradients
        log_parameters: Whether to log parameters
        log_freq: Frequency for logging (epochs)
        """
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    save_code: bool = False
    upload_checkpoints: bool = False
    artifact_type: str = "model"
    log_gradients: bool = False
    log_parameters: bool = False
    log_freq: int = 1
    mode: str = "online"  # online, offline, disabled
    
    def to_dict(self, exclude_fields: Optional[Set[str]] = None) -> dict:
        """Convert to dictionary representation"""
        return dataclass_to_dict(self, exclude_fields=exclude_fields)


class ModelType(Enum):
    """Available model types"""
    RSCD_MAMBA = "rscd_mamba"


@dataclass
class ExperimentConfig:
    """Main experiment configuration
    Attributes:
        name: Name of the experiment
        output_dir: Directory to save experiment outputs
        timestamp: Timestamp for the experiment
        use_cuda: Whether to use CUDA for training
        use_distributed: Whether to use distributed training
        model_config: Model configuration
        data_config: Data configuration
        training_config: Training configuration
        scheduler_config: Scheduler configuration
        distributed_config: Distributed configuration
        loss_type: Type of loss function to use
    """
    name: str = "experiment"
    output_dir: str = "data/debug/experiments"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Hardware configuration
    use_cuda: bool = True
    use_distributed: bool = False

    # Main configurations
    model_type: str = ModelType.RSCD_MAMBA.value  # Tipo di modello da usare
    
    # Model configuration
    model_config: BaseModelConfig = field(default_factory=RSCDMambaModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    distributed_config: DistributedConfig = field(default_factory=DistributedConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    loss_type: LossType = LossType.CROSSENTROPY

    def get_experiment_dir(self) -> str:
        """Get a full experiment directory path"""
        return os.path.join(self.output_dir, f"{self.name}")

    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory"""
        return os.path.join(self.get_experiment_dir(), "checkpoints")

    def get_logs_dir(self) -> str:
        """Get logs directory"""
        return os.path.join(self.get_experiment_dir(), "logs")

    def to_dict(self, exclude_fields: Optional[Set[str]] = None) -> dict:
        """Convert to dictionary representation"""
        return dataclass_to_dict(self, exclude_fields=exclude_fields)  # type: ignore[return]


# ================================ STRATEGIES ================================

class DeviceStrategy(ABC):
    """Strategy for device selection"""

    @abstractmethod
    def get_device(self, use_distributed: bool) -> tuple[torch.device, int]:
        """Returns (device, local_rank)"""


class CUDADeviceStrategy(DeviceStrategy):
    """CUDA device strategy"""

    def get_device(self, use_distributed: bool) -> tuple[torch.device, int]:
        if not torch.cuda.is_available():
            return torch.device('cpu'), 0

        if use_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f'cuda:{local_rank}')
        else:
            local_rank = 0
            device = torch.device('cuda:0')

        return device, local_rank


class CPUDeviceStrategy(DeviceStrategy):
    """CPU device strategy"""

    def get_device(self, use_distributed: bool) -> tuple[torch.device, int]:
        return torch.device('cpu'), 0


# ================================ FACTORIES ================================

class DatasetFactory:
    """Factory for creating dataloaders from datasets"""

    @staticmethod
    def create_dataloaders(config: DataConfig):
        """Create DataLoaders from provided datasets"""
        if not all([config.train_dataset, config.val_dataset]):
            raise ValueError("All datasets (train, val) must be provided in DataConfig")

        # Type assertions since we verified they're not None above
        train_dataset = config.train_dataset
        sampler = config.sampler
        val_dataset = config.val_dataset
        test_dataset = config.test_dataset

        assert train_dataset is not None, "Train dataset cannot be None"
        assert val_dataset is not None, "Val dataset cannot be None"

        # Keep workers alive between epochs and prefetch batches to avoid GPU starvation.
        persistent = config.num_workers > 0
        loader_kwargs = dict(
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=persistent,
        )
        if persistent:
            loader_kwargs['prefetch_factor'] = 4

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            #sampler=sampler,
            shuffle=config.shuffle,
            drop_last=True,
            **loader_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **loader_kwargs
        )

        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                **loader_kwargs
            )
        else:
            test_loader = None

        return train_loader, val_loader, test_loader

    @staticmethod
    def get_dataset_info(config: DataConfig):
        """Get dataset information for model creation"""
        if not config.train_dataset:
            raise ValueError("Train dataset must be provided to extract dataset info")

        # Get number of classes
        #test_labels = getattr(config.train_dataset, 'y', None)
        test_labels = config.train_dataset.classes
        if test_labels is None:
            raise ValueError("Dataset must have 'y' attribute for labels")

        num_classes = len(np.unique(test_labels))
        print(f"Detected number of classes: {num_classes}")

        # Get input channels from dataset
        #sample_data = getattr(config.train_dataset, 'X', None)
        sample_data = config.train_dataset[0][0]  # Assuming dataset returns (data, label) tuples
        if sample_data is None:
            raise ValueError("Dataset must have 'X' attribute for features")

        input_channels = sample_data.shape[0] if hasattr(sample_data, 'shape') else sample_data.shape[-1]

        return num_classes, input_channels


class BaseModelStrategy(ABC):
    """Abstract base class for model creation strategies"""

    @abstractmethod
    def create_model(self, config: BaseModelConfig,
                     num_classes: Optional[int],
                     input_channels: Optional[int],
                     device: torch.device) -> torch.nn.Module:
        """Create a model instance"""
        pass

    @abstractmethod
    def validate_config(self, config: BaseModelConfig) -> bool:
        """Validate if config is compatible with this model type"""
        pass
   
   
class RSCDMambaStrategy(BaseModelStrategy):
    """Strategy for creating RSCDMamba models"""

    @staticmethod
    def _resolve_norm_layer(norm_layer):
        """Convert norm_layer to class if it's a string"""
        if norm_layer is None:
            return nn.LayerNorm
        
        if isinstance(norm_layer, type):
            return norm_layer
        
        if isinstance(norm_layer, str):
            norm_layers = {
                "layernorm": nn.LayerNorm,
                "ln": nn.LayerNorm,
                "batchnorm2d": nn.BatchNorm2d,
                "bn": nn.BatchNorm2d,
                "groupnorm": nn.GroupNorm,
                "gn": nn.GroupNorm,
            }
            resolved = norm_layers.get(norm_layer.lower())
            if resolved is None:
                raise ValueError(f"Unknown norm_layer: {norm_layer}")
            return resolved
        
        # Assume it's already a callable
        return norm_layer

    def create_model(
        self,
        config: RSCDMambaModelConfig,
        num_classes: Optional[int],
        input_channels: Optional[int],
        device: torch.device
    ) -> torch.nn.Module:
        norm_layer = self._resolve_norm_layer(config.norm_layer)

        # Mappare i parametri di ModelConfig a quelli di RSCDMamba
        return RSCDMamba(
            img_size=getattr(config, 'img_size', 224),
            patch_size=getattr(config, 'patch_size', 4),
            in_chans=input_channels,
            num_classes=num_classes,
            depths=getattr(config, 'depths', [2, 2, 9, 2]),
            dims=getattr(config, 'dims', [96, 192, 384, 768]),
            d_state=getattr(config, 'd_state', 1),
            dt_init=getattr(config, 'dt_init', "random"),
            mlp_ratio=getattr(config, 'mlp_ratio', 4.0),
            drop_rate=getattr(config, 'drop_rate', config.drop_rate),
            attn_drop_rate=getattr(config, 'attn_drop_rate', config.attn_drop_rate),
            drop_path_rate=getattr(config, 'drop_path_rate', config.drop_path_rate),
            norm_layer=norm_layer,
            patch_norm=getattr(config, 'patch_norm', True),
            use_checkpoint=getattr(config, 'use_checkpoint', False),
            use_fg_mask=getattr(config, 'use_fg_mask', True),
            use_condition_branch=getattr(config, 'use_condition_branch', True),
            num_conditions=getattr(config, 'num_conditions', 6),
            condition_hidden_dim=getattr(config, 'condition_hidden_dim', None),
            condition_dropout=getattr(config, 'condition_dropout', 0.1),
        ).to(device)

    def validate_config(self, config: RSCDMambaModelConfig) -> bool:
        return hasattr(config, 'drop_rate')


class ModelStrategyRegistry:
    """Registry for model creation strategies"""

    def __init__(self):
        self._strategies: Dict[str, BaseModelStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default model strategies"""
        self.register(ModelType.RSCD_MAMBA.value, RSCDMambaStrategy())

    def register(self, model_type: str, strategy: BaseModelStrategy):
        """Register a new model strategy"""
        self._strategies[model_type.lower()] = strategy

    def get_strategy(self, model_type: str) -> BaseModelStrategy:
        """Get strategy for model type"""
        strategy = self._strategies.get(model_type.lower())
        if not strategy:
            available = list(self._strategies.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        return strategy

    def get_available_models(self) -> list:
        """Get a list of available model types"""
        return list(self._strategies.keys())


class ModelFactory:
    """Generic factory for creating models using a Strategy pattern"""

    def __init__(self):
        self._registry = ModelStrategyRegistry()

    def create_model(
        self, 
        model_type: str,
        config: BaseModelConfig,
        num_classes: Optional[int],
        input_channels: Optional[int],
        device: torch.device
    ) -> torch.nn.Module:
        """Create model based on type and configuration"""
        strategy = self._registry.get_strategy(model_type)

        if not strategy.validate_config(config):
            raise ValueError(f"Invalid configuration for model type: {model_type}")

        return strategy.create_model(config, num_classes, input_channels, device)

    def register_model_strategy(self, model_type: str, strategy: BaseModelStrategy):
        """Register a custom model strategy"""
        self._registry.register(model_type, strategy)

    def get_available_models(self) -> list:
        """Get a list of available model types"""
        return self._registry.get_available_models()


class ObserverFactory:
    """Factory for creating observers"""

    @staticmethod
    def create_observers(config: ExperimentConfig) -> ObserverList:
        """Create standard observers"""
        observers = ObserverList()

        logs_dir = config.get_logs_dir()
        os.makedirs(logs_dir, exist_ok=True)

        # Console logger
        console_logger = ConsoleLogger(
            log_file=os.path.join(logs_dir, f'training_{config.timestamp}.log')
        )
        observers.add(console_logger)

        # Metrics plotter
        metrics_plotter = MetricsPlotter(
            save_path=os.path.join(logs_dir, f'training_metrics_{config.timestamp}')
        )
        observers.add(metrics_plotter)

        # Wandb observer (if enabled)
        if config.wandb_config.enabled:
            wandb_observer = WandbObserver(config=deepcopy(config.wandb_config), experiment_config=config)
            observers.add(wandb_observer)

        # Optional observers
        # gradient_monitor = GradientMonitor(threshold=1e-6, check_frequency=40)
        # gradient_flow = GradientFlowMonitor(
        #     log_file=os.path.join(logs_dir, f'gradient_flow_{config.timestamp}.log'),
        #     check_frequency=1
        # )
        # # observers.add(gradient_monitor)
        # observers.add(gradient_flow)

        return observers

    @staticmethod
    def create_checkpoint_manager(config: ExperimentConfig, device: torch.device, logger: Optional[GenericObserver] = None) -> CheckpointManager:
        """Create checkpoint manager with optional Wandb upload support"""
        checkpoint_manager = CheckpointManager(
            trainer=None,  # Will be set later in ModelTrainer
            device=device,
            logger=logger
        )
        
        # Configure Wandb upload if enabled
        if config.wandb_config.enabled and config.wandb_config.upload_checkpoints:
            try:
                checkpoint_manager.configure_wandb_upload(
                    project_name=config.wandb_config.project,
                    run_name=config.wandb_config.name or config.name,
                    tags=config.wandb_config.tags,
                    notes=config.wandb_config.notes,
                    artifact_type=config.wandb_config.artifact_type
                )
                if logger:
                    logger.log_message("Wandb checkpoint upload configured", 'green')
            except Exception as e:
                if logger:
                    logger.log_message(f"Warning: Failed to configure Wandb upload: {e}", 'yellow')
        
        return checkpoint_manager


# ================================ BUILDER ================================

class ExperimentBuilder:
    """Builder for creating experiments"""

    def __init__(self):
        self._config = ExperimentConfig()

    def set_name(self, name: str) -> 'ExperimentBuilder':
        """Set experiment name"""
        self._config.name = name
        return self

    def set_output_dir(self, output_dir: str) -> 'ExperimentBuilder':
        """Set output directory"""
        self._config.output_dir = output_dir
        return self

    def set_datasets(
        self, 
        train_dataset: Dataset,
        sampler: Optional[torch.utils.data.Sampler],
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 128,
        num_workers: int = 4) -> 'ExperimentBuilder':
        """Set dataset instances and configuration"""
        self._config.data_config = DataConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataset=train_dataset,
            sampler=sampler,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
        return self

    def set_model_config(self, model_config: BaseModelConfig) -> 'ExperimentBuilder':
        """Set model configuration"""
        self._config.model_config = model_config

        return self

    def set_training_params(self, training_config: TrainingConfig) -> 'ExperimentBuilder':
        """Set training parameters"""
        self._config.training_config = training_config

        return self

    def set_scheduler(self, scheduler_config: SchedulerConfig) -> 'ExperimentBuilder':
        """Set scheduler configuration"""
        self._config.scheduler_config = scheduler_config

        return self

    def set_distributed_config(self, distributed_config: DistributedConfig) -> 'ExperimentBuilder':
        """Set distributed configuration"""
        self._config.distributed_config = distributed_config
        return self

    def enable_distributed(self, enable: bool = True) -> 'ExperimentBuilder':
        """Enable/disable distributed training"""
        self._config.use_distributed = enable
        return self

    def set_loss_type(self, loss_type: LossType) -> 'ExperimentBuilder':
        """Set loss type"""
        self._config.loss_type = loss_type
        return self

    def set_use_cuda(self, use_cuda: bool) -> 'ExperimentBuilder':
        """Set whether to use CUDA"""
        self._config.use_cuda = use_cuda
        return self

    def set_wandb_config(self, wandb_config: WandbConfig) -> 'ExperimentBuilder':
        """Set Wandb configuration"""
        self._config.wandb_config = wandb_config
        return self

    def build(self) -> 'Experiment':
        """Build the experiment"""
        return Experiment(self._config)


# ================================ MAIN CLASS ================================

class Experiment:
    """Main experiment class that orchestrates the training"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device_strategy = CUDADeviceStrategy() if config.use_cuda else CPUDeviceStrategy()
        self.trainer: Optional[ModelTrainer] = None
        self.device: Optional[torch.device] = None
        self.local_rank: Optional[int] = None

    def setup(self) -> 'Experiment':
        """Set up the experiment"""
        # Setup device
        self.device, self.local_rank = self.device_strategy.get_device(self.config.use_distributed)
        print(f"Using device: {self.device}, Local rank: {self.local_rank}")

        # Create experiment directories
        os.makedirs(self.config.get_experiment_dir(), exist_ok=True)
        os.makedirs(self.config.get_checkpoint_dir(), exist_ok=True)
        os.makedirs(self.config.get_logs_dir(), exist_ok=True)

        # Create dataloaders from provided datasets
        train_loader, val_loader, test_loader = DatasetFactory.create_dataloaders(self.config.data_config)

        # Get dataset info for model creation
        num_classes, input_channels = DatasetFactory.get_dataset_info(self.config.data_config)

        # Create model
        model_factory = ModelFactory()
        model = model_factory.create_model(
            model_type=self.config.model_type,
            config=self.config.model_config,
            num_classes=num_classes,
            input_channels=input_channels,
            device=self.device
        )

        # Create configurations
        training_config = self.config.training_config or TrainingConfig()
        scheduler_config = self.config.scheduler_config or SchedulerConfig()

        # Force config
        training_config.main_device = self.device
        training_config.save_dir = self.config.get_checkpoint_dir()
        training_config.use_distributed = self.config.use_distributed
        training_config.distributed_config = self.config.distributed_config

        # Create observers
        observers = ObserverFactory.create_observers(self.config)
        
        # Get console logger for checkpoint manager
        console_logger = observers.get(ConsoleLogger)
        
        # Create checkpoint manager
        checkpoint_manager = ObserverFactory.create_checkpoint_manager(
            self.config, 
            self.device, 
            console_logger
        )

        # Build trainer
        builder = TrainerBuilder()
        builder.set_model(model)
        builder.set_loss_strategy(self.config.loss_type.value)
        builder.set_scheduler_config(scheduler_config)
        builder.set_checkpoint_manager(checkpoint_manager)

        # Add observers one by one
        for observer in observers:
            builder.add_observer(observer)

        builder.set_train_loader(train_loader)
        builder.set_val_loader(val_loader)
        builder.set_test_loader(test_loader)
        builder.set_config(training_config)

        self.trainer = builder.build()

        return self

    def run(self):
        """Run the experiment"""
        if not self.trainer:
            raise RuntimeError("Experiment not setup. Call setup() first.")

        # Log experiment info
        if self.local_rank == 0:
            self._log_experiment_info()

        # Run training
        return self.trainer.train()

    def _log_experiment_info(self):
        """Log experiment information"""
        if not self.trainer:
            return

        # Find ConsoleLogger in observers
        logger = None
        for observer in self.trainer.observers:
            if isinstance(observer, ConsoleLogger):
                logger = observer
                break

        if not logger:
            return

        logger.log_message(f"Starting experiment: {self.config.name}")
        logger.log_message(f"Experiment directory: {self.config.get_experiment_dir()}")
        logger.log_message(f"Device: {self.device}")
        logger.log_message(f"Model configuration:\n{self.config.model_config}")
        logger.log_message(f"Training epochs: {self.config.training_config.epochs}")
        logger.log_message(f"Learning rate: {self.config.training_config.learning_rate}")
        logger.log_message(f"Batch size: {self.config.data_config.batch_size}")

    def get_trainer(self) -> Optional[ModelTrainer]:
        """Get the trainer instance"""
        return self.trainer

    def save_config(self, path: Optional[str] = None):
        """Save experiment configuration"""
        if self.config.distributed_config.local_rank != 0:
            return

        if path is None:
            path = os.path.join(self.config.get_experiment_dir(), "config.json")

        config_dict = self.config.to_dict()

        # Save to JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4)
