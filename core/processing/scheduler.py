from typing import Dict, Optional, List, Any

import torch
from torch import optim
from timm.scheduler.scheduler_factory import create_scheduler_v2 as timm_create_scheduler

from .observer import GenericObserver

class SchedulerType:
    """
    Enum-like class to define available scheduler types.
    """
    COSINE = "cosine"
    PLATEAU = "plateau"

    @classmethod
    def get_available_schedulers(cls) -> List[str]:
        """
        Returns a list of available scheduler types.
        """
        return [cls.COSINE, cls.PLATEAU]

class SchedulerConfig:
    """
    Configuration class for schedulers.
    """

    def __init__(self,
                 scheduler_type: str = SchedulerType.COSINE,
                 num_epochs: int = 100,
                 min_lr: float = 1e-7,
                 warmup_lr: float = 1e-5,
                 warmup_epochs: int = 0,
                 noise_pct: float = 0.67,
                 noise_std: float = 1.0,
                 cooldown_epochs: int = 0,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 2,
                 cycle_mul: float = 1.0,
                 cycle_decay: float = 0.1,
                 cycle_limit: int = 1,
                 force_new_min: bool = False,
                 force_new_lr: bool = False,
                 optimizer_lr: Optional[float] = None,
                 logger: Optional[GenericObserver] = None,
                 verbose: bool = False):

        self.scheduler_type = scheduler_type
        self.num_epochs = num_epochs
        self.min_lr = min_lr
        self.step_on_epochs = True
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.cooldown_epochs = cooldown_epochs

        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit

        self.force_new_min = force_new_min
        self.force_new_lr = force_new_lr
        self.lr = optimizer_lr
        self.logger = logger

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose  # Abilita il logging per default

    def _pretty_print(self):
        """Pretty print the training configuration"""
        _str = "Scheduler Configuration:\n"
        for key, value in vars(self).items():
            _str += f"{key.replace('_', ' ').capitalize()}: {value}\n"
        return _str

    def __str__(self):
        """String representation of the configuration."""
        return self._pretty_print()

    def get_scheduler_type(self) -> str:
        """Get the type of the scheduler."""
        return self.scheduler_type

    def to_dict(self) -> Dict:
        """Convert dynamically based on scheduler type"""
        if self.scheduler_type == SchedulerType.COSINE:
            config = dict(
                sched=self.scheduler_type,
                num_epochs=self.num_epochs,
                min_lr=self.min_lr,
                step_on_epochs=self.step_on_epochs,
                warmup_lr=self.warmup_lr,
                warmup_epochs=self.warmup_epochs,
                noise_pct=self.noise_pct,
                noise_std=self.noise_std,
                cooldown_epochs=self.cooldown_epochs,
                cycle_mul=self.cycle_mul,
                cycle_decay=self.cycle_decay,
                cycle_limit=self.cycle_limit,
            )
        elif self.scheduler_type == SchedulerType.PLATEAU:
            config = dict(
                mode=self.mode,
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return config


class SchedulerFactory:
    """
    Factory class to create different types of schedulers.
    """
    @staticmethod
    def is_scheduler_available(scheduler_type: str) -> bool:
        """
        Verifica se il tipo di scheduler è disponibile
        Args:
            scheduler_type: Tipo di scheduler da verificare
        Returns:
            True se il tipo di scheduler è disponibile, False altrimenti
        """
        return scheduler_type in SchedulerType.get_available_schedulers()

    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: Optional[str] = None,
                         config: Optional[SchedulerConfig] | Dict[
                             str, Any] = None) -> torch.optim.lr_scheduler.LRScheduler | Any:
        """
        Crea uno scheduler basato sul tipo specificato
        Args:
            optimizer: L'optimizer a cui applicare lo scheduler
            scheduler_type: Tipo di scheduler da creare ('cosine', 'plateau')
            config: Configurazione opzionale per lo scheduler
        Returns:
            Un'istanza dello scheduler appropriato
        """
        if config is None and scheduler_type is not None:
            config = SchedulerConfig(scheduler_type=scheduler_type)
        elif isinstance(config, dict):
            config = SchedulerConfig(**config)

        if config is None:
            raise ValueError("Configuration or scheduler type must be provided.")

        kwargs = config.to_dict()

        if config.get_scheduler_type() == SchedulerType.COSINE:
            _sched, _ = timm_create_scheduler(optimizer=optimizer, **kwargs)
            return _sched

        if config.get_scheduler_type() == SchedulerType.PLATEAU:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

        raise ValueError(f"Unknown scheduler type: {config.get_scheduler_type()}")


class SchedulerFacade(torch.optim.lr_scheduler.LRScheduler):
    """
    Facade to handle different types of schedulers in a unified way.

    This class provides a unified interface to manage different learning rate schedulers.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, optimizer: Optional[torch.optim.Optimizer] = None, config: Optional[SchedulerConfig] = None):
        """
        Initialize the SchedulerFacade.
        Args:
            optimizer: The optimizer to which the scheduler will be applied.
            config: Optional configuration for the scheduler.
        """
        self.optimizer: Optional[torch.optim.Optimizer] = optimizer

        self.config: Optional[SchedulerConfig] = config
        self.logger: Optional[GenericObserver] = config.logger if config and hasattr(config, 'logger') else None
        self.scheduler_factory = SchedulerFactory

        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        self.force_new_min = False
        self.force_new_lr = False

        self._built = False

    def _log(self, message: str, color: str, bold: bool = False):
        """Log message"""
        if self.logger:
            self.logger.log_message(message, color, bold)

    def get_scheduler_type(self) -> str:
        """
        Get the type of the scheduler.
        Returns:
            The type of the scheduler as a string.
        """
        if self.config is None:
            raise RuntimeError("SchedulerFacade must be configured before calling get_scheduler_type().")

        return self.config.get_scheduler_type()

    def _initial_step(self):
        pass

    def get_last_lr(self) -> list[float]:
        return self.get_lr()

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> 'SchedulerFacade':
        """
        Set the optimizer for the scheduler.
        """
        self.optimizer = optimizer
        return self

    def set_config(self, config: SchedulerConfig) -> 'SchedulerFacade':
        """
        Set the configuration for the scheduler.
        """
        self.config = config
        return self

    def build(self) -> 'SchedulerFacade':
        """
        Build the scheduler based on the provided configuration.
        """
        if self.optimizer is None:
            return self

        self.scheduler = self.scheduler_factory.create_scheduler(optimizer=self.optimizer, config=self.config)
        self._built = True
        return self

    def is_built(self) -> bool:
        """
        Check if the scheduler has been built.
        """
        return self._built

    def get_lr(self) -> List[float]:
        """
        Get the current learning rate from the scheduler.
        """
        if not self._built:
            raise RuntimeError("SchedulerFacade must be built before calling get_lr(). Call build() first.")

        if not self.scheduler:
            raise RuntimeError("Scheduler has not been initialized. Call build() first.")

        return self.scheduler.get_last_lr() if hasattr(self.scheduler, 'get_last_lr') else self.scheduler.get_lr()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the state dictionary into the scheduler.
        """
        if not self._built:
            raise RuntimeError("SchedulerFacade must be built before calling load_state_dict(). Call build() first.")

        if not self.scheduler:
            raise RuntimeError("Scheduler has not been initialized. Call build() first.")

        if hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(state_dict)
            
            try:
                if self.config and self.config.force_new_min:
                    self.force_new_min = self.config.force_new_min

                if self.config and self.config.force_new_lr:
                    self.force_new_lr = self.config.force_new_lr
                    new_lr = self.config.lr if self.config else old_lr

                if self.force_new_lr and self.optimizer is not None:
                    old_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
                    new_lr = self.config.lr if self.config else old_lr
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    # Reinizializza i parametri dello scheduler che fanno riferimento all'optimizer
                    if hasattr(self.scheduler, 'base_values') and hasattr(self.scheduler, '_initial_param_group_field'):
                        # Aggiorna i valori iniziali nei param_groups
                        initial_field = self.scheduler._initial_param_group_field
                        for param_group in self.optimizer.param_groups:
                            if initial_field in param_group:
                                param_group[initial_field] = new_lr
                        
                        # Reinizializza base_values basandosi sui nuovi valori iniziali
                        self.scheduler.base_values = [param_group[initial_field] for param_group in self.optimizer.param_groups]
                    
                    self._log(f"Updated optimizer learning rate to {new_lr}", color="yellow", bold=True)
                    if hasattr(self.scheduler, 'base_values') and hasattr(self.scheduler, 'update_groups'):
                        self.scheduler.update_groups(self.scheduler.base_values)
                        self._log(f"Reinitialized scheduler base_values to {self.scheduler.base_values}", color="yellow", bold=True)


                if self.force_new_min and hasattr(self.scheduler, 'min_lr'):
                    old_min_lr = getattr(self.scheduler, 'min_lr', 0.0)
                    new_min_lr = self.config.min_lr if self.config else old_min_lr
                    setattr(self.scheduler, 'min_lr', new_min_lr)
                    
                    # Se il nuovo min_lr è maggiore del precedente, forza l'optimizer a usare il nuovo min_lr
                    if new_min_lr > old_min_lr and self.optimizer is not None:
                        for param_group in self.optimizer.param_groups:
                            if param_group['lr'] < new_min_lr:
                                param_group['lr'] = new_min_lr

                    self._log(f"Updated optimizer learning rate to {new_min_lr}", color="yellow", bold=True)
                                
                elif self.force_new_min and hasattr(self.scheduler, 'lr_min'):
                    old_min_lr = getattr(self.scheduler, 'lr_min', 0.0)
                    new_min_lr = self.config.min_lr if self.config else old_min_lr
                    setattr(self.scheduler, 'lr_min', new_min_lr)

                    #Update warmup epochs if applicable
                    if hasattr(self.scheduler, 'warmup_epochs') and self.config:
                        self.scheduler.warmup_epochs = self.config.warmup_epochs
                        self._log(f"Updated scheduler warmup_epochs to {self.scheduler.warmup_epochs}", color="yellow", bold=True)
                    
                    # Se il nuovo min_lr è maggiore del precedente, forza l'optimizer a usare il nuovo min_lr
                    if new_min_lr > old_min_lr and self.optimizer is not None:
                        for param_group in self.optimizer.param_groups:
                            if param_group['lr'] < new_min_lr:
                                param_group['lr'] = new_min_lr

                    self._log(f"Updated optimizer learning rate to {new_min_lr}", color="yellow", bold=True)
            except Exception as e:
                raise RuntimeError(f"Error while forcing new minimum learning rate: {e}") from e
        else:
            raise RuntimeError("The scheduler does not support loading state dict.")

    def state_dict(self):
        """
        Get the state dictionary of the scheduler.
        """
        if not self._built:
            raise RuntimeError("SchedulerFacade must be built before calling state_dict(). Call build() first.")

        if not self.scheduler:
            raise RuntimeError("Scheduler has not been initialized. Call build() first.")

        if hasattr(self.scheduler, 'state_dict'):
            return self.scheduler.state_dict()

        raise RuntimeError("The scheduler does not support state dict.")

    def step(self, epoch: Optional[int] = None, metric: Optional[float | int] = None):
        """
        Execute a step of the scheduler.
        """
        if not self._built:
            raise RuntimeError("SchedulerFacade must be built before calling step(). Call build() first.")

        if not self.scheduler:
            raise RuntimeError("Scheduler has not been initialized. Call build() first.")

        if self.config and self.config.get_scheduler_type() == SchedulerType.PLATEAU and metric is not None:
            # Per ReduceLROnPlateau, step() accetta una metrica, non un epoch
            self.scheduler.step(metric)  # type: ignore
        elif self.config and self.config.get_scheduler_type() == SchedulerType.COSINE and epoch is not None:
            self.scheduler.step(epoch=epoch)
        else:
            self.scheduler.step()