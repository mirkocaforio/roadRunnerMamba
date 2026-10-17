from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader

from .loss import LossStrategy, LossStrategyFactory
from .observer import TrainingObserver, ObserverList
from .training import ModelTrainer, TrainingConfig
from .scheduler import SchedulerConfig, SchedulerFacade
from .checkpoint import CheckpointManager


class TrainerBuilder:
    """Builder pattern for configuring ModelTrainer"""

    def __init__(self):
        """Initialize TrainerBuilder with default values"""
        self._model: Optional[nn.Module] = None
        self._loss_strategy: Optional[LossStrategy] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler_config: Optional[SchedulerConfig] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        self._observers: ObserverList = ObserverList()
        self._config: Optional[TrainingConfig] = None
        self._checkpoint_manager: Optional[CheckpointManager] = None

        self.__scheduler = None

    def reset(self):
        """Reset builder to initial state"""
        self._model = None
        self._loss_strategy = None
        self._optimizer = None
        self._scheduler_config = None
        self._observers = ObserverList()
        self._config = None
        self._checkpoint_manager = None
        self.__scheduler = None
        return self

    def set_train_loader(self, train_loader: DataLoader):
        """Set training DataLoader"""
        self._train_loader = train_loader
        return self

    def set_val_loader(self, val_loader: Optional[DataLoader] = None):
        """Set validation DataLoader"""
        self._val_loader = val_loader
        return self

    def set_test_loader(self, test_loader: Optional[DataLoader] = None):
        """Set test DataLoader"""
        self._test_loader = test_loader
        return self

    def set_model(self, model: nn.Module):
        """Set the model to train"""
        self._model = model
        return self

    def set_loss_strategy(self, strategy: str | LossStrategy):
        """Set loss strategy by name or instance"""
        if isinstance(strategy, str):
            self._loss_strategy = LossStrategyFactory.create_strategy(strategy)
        else:
            self._loss_strategy = strategy
        return self

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set custom optimizer"""
        self._optimizer = optimizer
        return self

    def set_scheduler_config(self, scheduler_config: SchedulerConfig):
        """Set learning rate scheduler configuration"""
        self._scheduler_config = scheduler_config
        return self

    def add_observer(self, observer: TrainingObserver):
        """Add training observer"""
        self._observers.add(observer)
        return self

    def set_config(self, config: TrainingConfig):
        """Set training configuration"""
        self._config = config
        return self

    def set_checkpoint_manager(self, checkpoint_manager: CheckpointManager):
        """Set checkpoint manager"""
        self._checkpoint_manager = checkpoint_manager
        return self

    def build(self) -> 'ModelTrainer':
        """Build the ModelTrainer instance"""
        if self._model is None:
            raise ValueError("Model must be set before building trainer")

        if self._loss_strategy is None:
            raise ValueError("Loss strategy must be set before building trainer")

        if self._train_loader is None:
            raise ValueError("Training DataLoader must be set before building trainer")

        if self._val_loader is None:
            raise ValueError("Validation DataLoader must be set before building trainer")

        # Build scheduler
        if self._scheduler_config is None:
            raise ValueError("Scheduler configuration must be set when using a scheduler strategy")

        self.__scheduler = SchedulerFacade(optimizer=self._optimizer, config=self._scheduler_config).build()

        return ModelTrainer(
            model=self._model,
            loss_strategy=self._loss_strategy,
            optimizer=self._optimizer,
            scheduler=self.__scheduler,
            observers=self._observers,
            train_loader=self._train_loader,
            val_loader=self._val_loader,
            test_loader=self._test_loader,
            config=self._config,
            checkpoint_manager=self._checkpoint_manager
        )
