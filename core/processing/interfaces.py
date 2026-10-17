# Trainer interface
from typing import Optional, Protocol
from torch.utils.data import DataLoader

from torch import nn, optim, device

from .scheduler import SchedulerFacade
from .loss import LossStrategy
from .observer import ObserverList
from .config import TrainingConfig, DistributedConfig
from .object import HistoryDataObject


class TrainerInterface(Protocol):
    """Trainer interface for model training and evaluation"""
    
    # Core components
    model: nn.Module
    loss_strategy: LossStrategy
    optimizer: optim.Optimizer
    scheduler: SchedulerFacade
    device: device
    config: TrainingConfig
    observers: ObserverList
    
    # Training state
    history: HistoryDataObject
    best_val_metric: float
    best_metric_type: str
    early_stopping_counter: int
    hard_stop: bool
    start_epoch: int
    is_distributed: bool

    # Distributed configuration
    distributed_config: DistributedConfig
    
    # Data loaders
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    
    def _should_log(self) -> bool:
        """Check if this process should log"""
        if not self.is_distributed:
            return True
        return self.distributed_config.is_main_process()
    
    def _notify_observers(self, event: str, **kwargs) -> None:
        """Notify all observers of an event"""
        if not self._should_log():  # Only main process should log
            return
        try:
            self.observers.notify(event, **kwargs)
        except Exception as e:
            raise e
