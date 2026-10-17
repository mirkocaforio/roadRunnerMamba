import json
import os
import traceback
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .checkpoint import CheckpointManager
from .config import DistributedConfig, TrainingConfig
from .errors import OutOfMemoryError
from .events import TrainingEvents
from .interfaces import TrainerInterface
from .loss import LossStrategy
from .object import HistoryDataObject, MetricType, TrainingResult, TrainingStatus
from .observer import ConsoleLogger, ObserverList
from .scheduler import SchedulerConfig, SchedulerFacade, SchedulerType


class ModelTrainer(TrainerInterface):
    """Model trainer"""

    def __init__(
        self,
        model: nn.Module,
        loss_strategy: LossStrategy,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[SchedulerFacade] = None,
        observers: ObserverList = ObserverList(),
        config: Optional[TrainingConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):

        if config is None:
            config = TrainingConfig()

        self.__model_export_name = "model_export.pt"
        self.__model_export_onnx_name = "model_export.onnx"
        self.__model_export_script_name = "model_export_torchscript.pt"

        self.model = model
        self.loss_strategy = loss_strategy
        self.observers = observers or ObserverList()
        self.config = config

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Distributed training setup
        self.is_distributed = self.config.use_distributed
        self.distributed_config: DistributedConfig = self.config.distributed_config

        if self.is_distributed:
            self._setup_distributed()

        # Setup device
        self.device = self._auto_detect_device()

        # Automatic Mixed Precision (bfloat16). Only meaningful on CUDA.
        self.use_amp = (
            bool(getattr(self.config, "use_amp", False)) and self.device.type == "cuda"
        )
        self.amp_dtype = torch.bfloat16
        self._autocast_device = (
            self.device.type if self.device.type in ("cuda", "cpu") else "cuda"
        )

        # Enable cuDNN autotuning (input size is fixed at 224). Keep TF32 OFF for maximum
        # metric fidelity: the residual fp32 matmuls/convs stay at full precision (under AMP
        # most ops already run in bf16, so the speed cost is minimal).
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        self._setup_model()

        # Setup optimizer
        self.optimizer = optimizer or self._create_default_optimizer()

        # Setup scheduler
        self.scheduler: SchedulerFacade = self._setup_scheduler(scheduler)

        # Training state
        self.history = HistoryDataObject()

        self.best_metric_type = MetricType.ACCURACY
        self.best_val_metric = MetricType.get_limit(self.best_metric_type)

        self.early_stopping_counter = 0
        self.hard_stop = False

        # Cached evaluation tensors, used to recompute detailed metrics on best
        self._last_eval_outputs = None
        self._last_eval_targets = None

        # Resume state
        self.start_epoch = 0
        self.last_checkpoint_path = None

        # Initialize CheckpointManager
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            trainer=self,
            device=self.device,
            logger=self.observers.get(observer_class=ConsoleLogger),  # type: ignore
        )
        if self.checkpoint_manager.trainer is None:
            self.checkpoint_manager.trainer = self

        # Try autoresume
        if self.config.auto_resume:
            self._try_auto_resume()

    def _setup_distributed(self):
        """Setup distributed training environment"""
        if not self.distributed_config.is_distributed():
            self.is_distributed = False
            return

        if not torch.cuda.is_available():
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message="Warning: CUDA not available, disabling distributed training",
                color="yellow",
            )
            self.is_distributed = False
            return

        try:
            # Auto-detect from environment if not set
            if self.distributed_config.world_size is None:
                self.distributed_config.auto_detect_from_env()

            # Initialize a process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.distributed_config.backend,
                    init_method=self.distributed_config.init_method,
                    world_size=self.distributed_config.world_size,
                    rank=self.distributed_config.rank,
                )

            # Set local rank for CUDA device
            if self.distributed_config.local_rank is not None:
                torch.cuda.set_device(self.distributed_config.local_rank)

            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Distributed training initialized: rank {self.distributed_config.rank}/{self.distributed_config.world_size}",
                color="green",
            )

        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Failed to initialize distributed training: {e}",
                color="red",
            )
            self.is_distributed = False

    def _setup_distributed_samplers(self):
        """Setup distributed samplers for data loaders"""
        if not self.is_distributed:
            return

        # Create distributed samplers
        train_sampler = DistributedSampler(
            self.train_loader.dataset,
            num_replicas=self.distributed_config.world_size,
            rank=self.distributed_config.rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            self.val_loader.dataset,
            num_replicas=self.distributed_config.world_size,
            rank=self.distributed_config.rank,
            shuffle=False,
        )

        # Preserve the performance kwargs (pin_memory/persistent_workers/prefetch) when
        # rebuilding the loaders with distributed samplers.
        def _perf_kwargs(loader):
            nw = loader.num_workers
            kw = dict(
                num_workers=nw, pin_memory=loader.pin_memory, persistent_workers=nw > 0
            )
            if nw > 0:
                kw["prefetch_factor"] = getattr(loader, "prefetch_factor", None) or 4
            return kw

        # Replace data loaders with distributed versions
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            sampler=train_sampler,
            drop_last=self.train_loader.drop_last,
            **_perf_kwargs(self.train_loader),
        )

        self.val_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=self.val_loader.batch_size,
            sampler=val_sampler,
            drop_last=False,
            **_perf_kwargs(self.val_loader),
        )

        if self.test_loader:
            test_sampler = DistributedSampler(
                self.test_loader.dataset,
                num_replicas=self.distributed_config.world_size,
                rank=self.distributed_config.rank,
                shuffle=False,
            )

            self.test_loader = DataLoader(
                self.test_loader.dataset,
                batch_size=self.test_loader.batch_size,
                sampler=test_sampler,
                drop_last=False,
                **_perf_kwargs(self.test_loader),
            )

    def stop(self):
        """Stop training"""
        self.hard_stop = True
        self._notify_observers(TrainingEvents.ON_TRAINING_STOP)

    def _sync_training_interruption(self) -> bool:
        """
        Synchronize training interruption across all processes.

        Returns:
            bool: True if any process requested to stop training
        """
        if not self.is_distributed:
            return self.hard_stop

        device = self.device

        # Create tensor with current stop state
        stop_tensor = torch.tensor(
            [1 if self.hard_stop else 0], dtype=torch.int, device=device
        )

        # Use all_reduce to check if ANY process wants to stop
        # This will sum all the stop signals across processes
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.SUM)

        # If any process wanted to stop, all processes should stop
        global_stop = stop_tensor.item() > 0

        # Update local state
        if global_stop:
            self.hard_stop = True

        return global_stop

    def _check_stop(self):
        """Check if training should be stopped"""
        return self._sync_training_interruption()

    def _auto_detect_device(self) -> torch.device:
        """Auto-detect best available device"""
        if self.is_distributed and self.distributed_config.local_rank is not None:
            return torch.device(f"cuda:{self.distributed_config.local_rank}")
        elif torch.cuda.is_available():
            return self.config.main_device or torch.device("cuda:0")
        return self.config.main_device or torch.device("cpu")

    def _setup_model(self):
        """Setup model with distributed support"""
        try:
            # Move model to device
            self.model = self.model.to(self.device)

            # Setup distributed samplers if needed
            if self.is_distributed:
                self._setup_distributed_samplers()

            # Wrap model with DistributedDataParallel
            if self.is_distributed and self._can_use_distributed():
                self._setup_distributed_data_parallel()
            else:
                if self.is_distributed:
                    self._notify_observers(
                        TrainingEvents.LOG_MESSAGE,
                        message="Warning: Distributed training requested but not possible, using single GPU",
                        color="yellow",
                    )
                    self.is_distributed = False

        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Error setting up model: {e}",
                color="red",
            )
            raise

    def _can_use_distributed(self) -> bool:
        """Check if distributed training can be used"""
        if not torch.cuda.is_available():
            return False

        if not dist.is_initialized():
            return False

        if self.device.type != "cuda":
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message="Warning: DistributedDataParallel requires CUDA device",
                color="yellow",
            )
            return False

        return True

    def _setup_distributed_data_parallel(self):
        """Setup DistributedDataParallel"""
        self._notify_observers(
            TrainingEvents.LOG_MESSAGE,
            message=f"Setting up DistributedDataParallel on device {self.device}",
            color="yellow",
        )

        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            output_device=self.device.index if self.device.type == "cuda" else None,
            find_unused_parameters=self.distributed_config.find_unused_parameters,
            gradient_as_bucket_view=self.distributed_config.gradient_as_bucket_view,
            static_graph=self.distributed_config.static_graph,
        )

        self._notify_observers(
            TrainingEvents.LOG_MESSAGE,
            message="DistributedDataParallel setup complete",
            color="green",
        )

    def _create_default_optimizer(self) -> torch.optim.Optimizer:
        """Create default optimizer"""
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        # fused=True uses a single fused CUDA kernel for the optimizer step (much faster).
        fused = self.device.type == "cuda"
        return torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay, fused=fused
        )

    def _setup_scheduler(self, scheduler: Optional[SchedulerFacade]) -> SchedulerFacade:
        """Setup learning rate scheduler"""
        if scheduler is None or not isinstance(scheduler, SchedulerFacade):
            _scheduler = self._create_default_scheduler()
        elif (
            isinstance(scheduler, SchedulerFacade)
            and not scheduler.optimizer == self.optimizer
        ):
            _scheduler = scheduler.set_optimizer(self.optimizer).build()
        elif isinstance(scheduler, SchedulerFacade) and not scheduler.is_built():
            _scheduler = scheduler.build()
        elif scheduler is not None:
            _scheduler = scheduler
        else:
            raise ValueError("Invalid scheduler provided")

        if self._should_log():
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Using scheduler: {scheduler.get_scheduler_type()}",  # type: ignore
                color="gray",
            )

        return _scheduler

    def _create_default_scheduler(self):
        """Create default learning rate scheduler"""
        _config = SchedulerConfig(
            scheduler_type=SchedulerType.PLATEAU, mode="min", factor=0.7, patience=2
        )
        _scheduler = SchedulerFacade(optimizer=self.optimizer, config=_config)
        return _scheduler

    def _should_log(self) -> bool:
        """Check if this process should log (only main process in distributed training)"""
        if not self.is_distributed:
            return True
        return self.distributed_config.is_main_process()

    def _notify_observers(self, event: str, **kwargs):
        """Notify all observers of an event (only from main process in distributed training)"""
        if not self._should_log():
            return

        try:
            self.observers.notify(event, **kwargs)
        except Exception as e:
            raise e

    def train(self) -> Optional[TrainingResult]:
        """
        Train the model with distributed support

        Returns:
            Training history and metrics
        """
        try:
            self._notify_observers(TrainingEvents.ON_TRAINING_START)

            epochs = self.config.epochs
            patience = self.config.patience
            save_dir = self.config.save_dir

            if self._should_log():
                os.makedirs(save_dir, exist_ok=True)

            for epoch in range(self.start_epoch, epochs):
                if self._check_stop():
                    self._notify_observers(
                        TrainingEvents.LOG_MESSAGE,
                        message="Training stopped by user",
                        color="red",
                        bold=True,
                    )
                    break

                try:
                    # Set epoch for distributed sampler
                    if self.is_distributed and hasattr(
                        self.train_loader.sampler, "set_epoch"
                    ):
                        self.train_loader.sampler.set_epoch(epoch)  # type: ignore

                    self._notify_observers(
                        TrainingEvents.ON_EPOCH_START, epoch=epoch, total_epochs=epochs
                    )

                    # Training phase
                    train_loss = self._train_epoch(self.train_loader, epoch)

                    if self._check_stop():
                        continue

                    # Validation phase
                    val_loss, val_metrics = self._validate_epoch(self.val_loader, epoch)
                    check_metric = val_metrics.get(
                        self.best_metric_type,
                        MetricType.get_limit(self.best_metric_type),
                    )

                    if self._check_stop():
                        continue

                    # Determine (once) whether this is the best model so far. Only
                    # the main process decides; the value is reused for both the
                    # detailed-metrics logging and the checkpoint/early-stopping logic.
                    is_best = self._should_log() and self._should_save_model(
                        check_metric
                    )

                    # Update history (only on the main process)
                    if self._should_log():
                        # History keeps the base metrics for consistent shape.
                        self._update_history(train_loss, val_loss, val_metrics)

                        # Compute detailed metrics (per-class, confusion matrix) only
                        # when a new best is reached, to keep the log output clean.
                        display_metrics = (
                            self._enrich_detailed_metrics(val_metrics)
                            if is_best
                            else val_metrics
                        )

                        # Prepare metrics for observers
                        epoch_metrics = {
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "train_loss": train_loss,
                            **{f"val_{k}": v for k, v in display_metrics.items()},
                        }
                        self._notify_observers(
                            TrainingEvents.ON_EPOCH_END,
                            epoch=epoch,
                            metrics=epoch_metrics,
                        )

                    # Release cached evaluation tensors
                    self._last_eval_outputs = None
                    self._last_eval_targets = None

                    # Learning rate scheduling
                    if self.scheduler:
                        self.scheduler.step(metric=check_metric, epoch=epoch)

                    # Save the latest checkpoint every epoch (overwrites last_model.pt)
                    if self._should_log():
                        self._save_latest_checkpoint(epoch, save_dir, val_metrics)

                    # Early stopping check (saves best_model.pt when is_best)
                    should_stop = self._handle_early_stopping(
                        epoch, save_dir, is_best, val_metrics, patience
                    )

                    if should_stop:
                        if self._should_log():
                            self._notify_observers(
                                TrainingEvents.LOG_MESSAGE,
                                message=f"Early stopping after {epoch + 1} epochs",
                                color="yellow",
                            )
                        break

                    # Synchronize processes
                    if self.is_distributed:
                        dist.barrier(
                            device_ids=[self.device.index]
                            if self.device.type == "cuda"
                            else None
                        )

                    self._notify_observers(
                        TrainingEvents.DEBUG_GRADIENT_FLOW_MONITOR,
                        model=self.model,
                        optimizer=self.optimizer,
                        batch_idx=epoch,
                        input_batch=self.train_loader.dataset[0][0]
                        .to(self.device, non_blocking=True)
                        .unsqueeze(0),
                        epoch=epoch + 1,
                    )

                except Exception as e:
                    # Handle OutOfMemoryError specifically
                    if isinstance(e, OutOfMemoryError):
                        raise e

                    self._notify_observers(
                        TrainingEvents.LOG_MESSAGE,
                        message=f"Error in epoch {epoch}: {e}",
                        color="red",
                    )

                    if self.config.stop_on_error:
                        raise
                    continue

            final_metrics = {}
            status = TrainingStatus.FAILED

            if not self._check_stop():
                # Load the best model (only on the main process)
                if self._should_log():
                    self._load_best_model(save_dir)

                # Final evaluation (with per-class metrics and confusion matrix)
                if self.test_loader:
                    if self._should_log():
                        test_loss, test_metrics = self._evaluate(
                            self.test_loader, desc="Test Evaluation"
                        )
                        test_metrics = self._enrich_detailed_metrics(test_metrics)
                        self._last_eval_outputs = None
                        self._last_eval_targets = None
                        final_metrics.update(
                            {f"test_{k}": v for k, v in test_metrics.items()}
                        )
                        final_metrics["test_loss"] = test_loss
                        self._save_final_metrics(save_dir, final_metrics)

                if self._should_log():
                    # Save the final model before notifying the observers, so that an
                    # observer failure can never prevent the export
                    self.save_model(save_dir, torch_script=False, onnx=False)
                    self._notify_observers(
                        TrainingEvents.ON_TRAINING_END, final_metrics=final_metrics
                    )

                status = TrainingStatus.COMPLETED

            # Cleanup distributed training
            if self.is_distributed:
                dist.destroy_process_group()

            return TrainingResult(
                history=self.history,
                metrics=final_metrics,
                best_metric=self.best_val_metric,
                metric_type=self.best_metric_type,
                status=status,
            )

        except Exception as e:
            # Handle OutOfMemoryError specifically
            if isinstance(e, OutOfMemoryError):
                self._notify_observers(
                    TrainingEvents.LOG_MESSAGE,
                    message="Out of memory during training, exiting ...",
                    color="red",
                )
                raise e

            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Critical training error: {e}",
                color="red",
            )

            # Cleanup distributed training
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()

            return TrainingResult(
                history=self.history,
                metrics={},
                best_metric=self.best_val_metric,
                metric_type=self.best_metric_type,
                status=TrainingStatus.FAILED,
            )

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        num_batches = len(train_loader)

        if self._should_log():
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        else:
            progress_bar = train_loader

        # Accumulate the loss on-device to avoid a CPU-GPU sync every batch.
        running_loss = torch.zeros((), device=self.device)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            if self._check_stop():
                break

            try:
                inputs, targets = (
                    inputs.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )

                # Forward (optionally under autocast for mixed precision)
                with torch.autocast(
                    device_type=self._autocast_device,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(inputs)
                    loss = self.loss_strategy.compute_loss(outputs, targets)

                # Backward
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore

                # Gradient clipping (optional)
                max_grad_norm = self.config.max_grad_norm
                if max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )

                self.optimizer.step()

                running_loss += loss.detach()

                # Refresh the progress bar periodically (avoids a per-batch .item() sync)
                if (
                    self._should_log()
                    and hasattr(progress_bar, "set_postfix")
                    and (batch_idx % 50 == 0)
                ):
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})  # type: ignore

            except Exception as e:
                if "OutOfMemoryError" in str(e):
                    raise OutOfMemoryError(
                        f"Out of memory during training batch {batch_idx}"
                    ) from e

                # Stampa il traceback completo
                error_traceback = traceback.format_exc()

                self._notify_observers(
                    TrainingEvents.LOG_MESSAGE, message=error_traceback, color="red"
                )

                self._notify_observers(
                    TrainingEvents.LOG_MESSAGE,
                    message=f"Error in batch {batch_idx}: {e}",
                    color="red",
                )

                if self.config.stop_on_batch_error:
                    raise
                continue

        return running_loss.item() / num_batches

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> tuple:
        """Validate for one epoch"""
        return self._evaluate(val_loader, desc=f"Validation Epoch {epoch + 1}")

    def _evaluate(self, data_loader: DataLoader, desc: str = "Evaluation") -> tuple:
        """Evaluate the model on a dataset"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            if self._should_log():
                progress_bar = tqdm(data_loader, desc=desc)
            else:
                progress_bar = data_loader

            for inputs, targets in progress_bar:
                try:
                    inputs, targets = (
                        inputs.to(self.device, non_blocking=True),
                        targets.to(self.device, non_blocking=True),
                    )

                    with torch.autocast(
                        device_type=self._autocast_device,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        outputs = self.model(inputs)
                        loss = self.loss_strategy.compute_loss(outputs, targets)

                    total_loss += loss.item()  # type: ignore
                    # Cast back to float32 on CPU so downstream metrics (sklearn) get fp32.
                    all_outputs.append(outputs.float().cpu())
                    all_targets.append(targets.cpu())

                except Exception as e:
                    self._notify_observers(
                        TrainingEvents.LOG_MESSAGE,
                        message=f"Error in evaluation batch: {e}",
                        color="red",
                    )
                    continue

        # Compute metrics
        metrics = {}
        # Reset cached tensors used to optionally recompute detailed metrics
        self._last_eval_outputs = None
        self._last_eval_targets = None
        if all_outputs and all_targets:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Keep the raw outputs/targets so detailed metrics (per-class,
            # confusion matrix) can be recomputed later only when needed (best).
            self._last_eval_outputs = all_outputs
            self._last_eval_targets = all_targets

            metrics = self.loss_strategy.compute_metrics(
                all_outputs, all_targets, detailed=False
            )

            if metrics is None:
                metrics = {}

        avg_loss = total_loss / len(data_loader)
        metrics["loss"] = avg_loss

        return avg_loss, metrics

    def _update_history(
        self, train_loss: float, val_loss: float, val_metrics: Dict[str, float]
    ):
        """Update training history"""
        self.history.train_loss.append(train_loss)
        self.history.val_loss.append(val_loss)

        for metric_name, value in val_metrics.items():
            if metric_name not in self.history.val_metrics:
                self.history.val_metrics[metric_name] = []  # type: ignore
            self.history.val_metrics[metric_name].append(value)  # type: ignore

    def _should_save_model(self, val_metric: float) -> bool:
        """Check if current model should be saved"""
        if MetricType.get_limit(self.best_metric_type) == -float("inf"):
            return val_metric > self.best_val_metric

        if MetricType.get_limit(self.best_metric_type) == float("inf"):
            return val_metric < self.best_val_metric

        return val_metric < self.best_val_metric

    def _handle_early_stopping(
        self,
        epoch: int,
        save_dir: str,
        is_best: bool,
        val_metrics: Dict[str, float],
        patience: int,
    ) -> bool:
        """
        Handle early stopping in a distributed manner by synchronizing decisions across all processes.

        Args:
            epoch: Current epoch number
            save_dir: Directory to save checkpoints
            is_best: Whether the current model is the best so far (decided by the
                     main process via _should_save_model)
            val_metrics: Dictionary of validation metrics
            patience: Early stopping patience

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Initialize variables for distributed communication
        stop_training = False

        if self.is_distributed:
            # Create tensors for distributed communication
            device = self.device

            # Only main process saves the best model and updates the stop decision
            if self._should_log():
                if is_best:
                    self.best_val_metric = val_metrics.get(
                        self.best_metric_type,
                        MetricType.get_limit(self.best_metric_type),
                    )
                    # Save best model
                    success = self.checkpoint_manager.save_best(
                        epoch, save_dir, val_metrics
                    )
                    if success:
                        self.early_stopping_counter = 0
                    else:
                        self._notify_observers(
                            TrainingEvents.LOG_MESSAGE,
                            message="Failed to save best model checkpoint",
                            color="red",
                        )
                else:
                    self.early_stopping_counter += 1

                # Check if we should stop
                stop_training = self.early_stopping_counter >= patience

            # Broadcast early stopping decision to all processes
            stop_tensor = torch.tensor(
                [1 if stop_training else 0], dtype=torch.int, device=device
            )
            dist.broadcast(stop_tensor, src=0)  # src=0 means rank 0 (main process)

            # Update local variables from broadcasted values
            stop_training = bool(stop_tensor.item())

            # Broadcast
            self._broadcast_training_state()

            # Additional barrier to ensure all processes are synchronized
            dist.barrier(
                device_ids=[self.device.index] if self.device.type == "cuda" else None
            )

        else:
            # Non-distributed case
            if is_best:
                self.best_val_metric = val_metrics.get(
                    self.best_metric_type, MetricType.get_limit(self.best_metric_type)
                )
                success = self.checkpoint_manager.save_best(
                    epoch, save_dir, val_metrics
                )
                if success:
                    self.early_stopping_counter = 0
                else:
                    self._notify_observers(
                        TrainingEvents.LOG_MESSAGE,
                        message="Failed to save best model checkpoint",
                        color="red",
                    )
            else:
                self.early_stopping_counter += 1

            stop_training = self.early_stopping_counter >= patience

        return stop_training

    def _broadcast_training_state(self):
        """
        Broadcast training state (early stopping counter, best metric) across all processes.
        This ensures all processes have consistent state for early stopping decisions.
        """
        if not self.is_distributed:
            return

        device = self.device

        # Create tensors for broadcasting
        state_tensor = torch.tensor(
            [self.early_stopping_counter, self.best_val_metric],
            dtype=torch.float,
            device=device,
        )

        # Broadcast from main process (rank 0) to all other processes
        dist.broadcast(state_tensor, src=0)

        # Update local state from broadcasted values
        if not self._should_log():  # Non-main processes
            self.early_stopping_counter = int(state_tensor[0].item())
            self.best_val_metric = state_tensor[1].item()

    def _save_latest_checkpoint(
        self, epoch: int, save_dir: str, val_metrics: Dict[str, float]
    ):
        """Save (overwrite) the latest checkpoint every epoch"""
        if not self._should_log():
            return

        success = self.checkpoint_manager.save_latest(epoch, save_dir, val_metrics)
        if not success:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Failed to save latest checkpoint for epoch {epoch}",
                color="red",
            )

    def _enrich_detailed_metrics(
        self, val_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Recompute validation metrics with detailed=True (per-class metrics and
        confusion matrix) using the cached evaluation tensors.

        Used only when a new best model is reached, to keep the regular log output
        clean. Falls back to the basic metrics if the cached tensors are missing.
        """
        if self._last_eval_outputs is None or self._last_eval_targets is None:
            return val_metrics

        try:
            detailed_metrics = self.loss_strategy.compute_metrics(
                self._last_eval_outputs, self._last_eval_targets, detailed=True
            )
            if not detailed_metrics:
                return val_metrics

            # Preserve loss (added after compute_metrics in _evaluate)
            if "loss" in val_metrics and "loss" not in detailed_metrics:
                detailed_metrics["loss"] = val_metrics["loss"]
            return detailed_metrics
        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Warning: could not compute detailed metrics: {e}",
                color="yellow",
            )
            return val_metrics

    def _save_final_metrics(
        self, save_dir: str, final_metrics: Dict, filename: str = "test_metrics.json"
    ):
        """Save the final test metrics (including per-class and confusion matrix) as JSON"""

        def _to_json(value):
            if isinstance(value, dict):
                return {str(k): _to_json(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_json(v) for v in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            return value

        try:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_to_json(final_metrics), f, indent=4)
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Test metrics saved to {path}",
                color="green",
            )
        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Warning: could not save test metrics: {e}",
                color="yellow",
            )

    def _cleanup_old_checkpoints(self, save_dir: str, keep_last: int = 3):
        """Clean up old periodic checkpoints, keeping only the most recent ones"""
        try:
            checkpoint_files = []

            for file in os.listdir(save_dir):
                if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
                    try:
                        epoch_num = int(
                            file.replace("checkpoint_epoch_", "").replace(".pt", "")
                        )
                        file_path = os.path.join(save_dir, file)
                        checkpoint_files.append((epoch_num, file_path))
                    except ValueError:
                        continue

            # Sort by epoch number and keep only the most recent
            if len(checkpoint_files) > keep_last:
                checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                files_to_remove = checkpoint_files[keep_last:]

                for _, file_path in files_to_remove:
                    os.remove(file_path)

        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Error cleaning up checkpoints: {e}",
                color="yellow",
            )

    def save_model(self, save_dir: str, torch_script: bool = False, onnx: bool = False):
        """Save the whole nn.Module model to a file"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get the actual model (unwrap from DDP if needed)
        model = self.model.module if self.is_distributed else self.model
        model_path_name = self.__model_export_name

        try:
            if torch_script:
                # Convert to TorchScript
                script_model = torch.jit.script(model)
                script_path_name = self.__model_export_script_name
                torch.save(script_model, os.path.join(save_dir, script_path_name))
                self._notify_observers(
                    TrainingEvents.LOG_MESSAGE,
                    message="Model converted to TorchScript",
                    color="green",
                    bold=True,
                )

            if onnx:
                # Save as ONNX format
                self.save_onnx(save_dir, onnx_path=self.__model_export_onnx_name)

            torch.save(model, os.path.join(save_dir, model_path_name))
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message="Model saved successfully",
                color="green",
                bold=True,
            )
        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Error saving model: {e}",
                color="red",
                bold=True,
            )

    def save_onnx(self, save_dir: str, onnx_path: str = "model.onnx"):
        """Save the model as ONNX format"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            # Get the actual model (unwrap from DDP if needed)
            model = self.model.module if self.is_distributed else self.model
            model.eval()  # type: ignore

            # Create a dummy input based on training data shape
            inp = (next(iter(self.train_loader))[0].to(self.device),)
            torch.cuda.set_device(self.device)
            onnx_program = torch.onnx.export(model, inp, dynamo=True)  # type: ignore
            onnx_program.save(os.path.join(save_dir, onnx_path))  # type: ignore

            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message="Model saved as ONNX successfully",
                color="green",
                bold=True,
            )
        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Error saving model as ONNX: {e}",
                color="red",
                bold=True,
            )

    def _load_best_model(self, save_dir: str):
        """Load the best saved model"""
        checkpoint_data = self.checkpoint_manager.load_best(save_dir, strict=False)
        if checkpoint_data:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message="Best model loaded for final evaluation",
                color="green",
            )
        else:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message="No best model checkpoint found",
                color="yellow",
            )

    def load_checkpoint(
        self, checkpoint_path: str, load_training_state: bool = True
    ) -> bool:
        """Load model from specific checkpoint file using CheckpointManager"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self._notify_observers(
            TrainingEvents.LOG_MESSAGE,
            message=f"Loading checkpoint from {checkpoint_path}",
            color="blue",
        )

        try:
            # Use CheckpointManager's new method for loading specific files
            success = self.checkpoint_manager.load_from_file(
                checkpoint_path, load_training_state=load_training_state
            )

            if success:
                self._notify_observers(
                    TrainingEvents.LOG_MESSAGE,
                    message="Checkpoint loaded successfully",
                    color="green",
                )
            else:
                self._notify_observers(
                    TrainingEvents.LOG_MESSAGE,
                    message="Failed to load checkpoint",
                    color="red",
                )

            return success

        except Exception as e:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Error loading checkpoint: {str(e)[:100]}",
                color="red",
            )
            raise e

    def _try_auto_resume(self):
        """Try to automatically resume from the last checkpoint"""
        if not self._should_log():
            return

        save_dir = self.config.save_dir

        if self.checkpoint_manager.auto_resume(save_dir):
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message=f"Resumed training from epoch {self.start_epoch}",
                color="green",
            )
        else:
            self._notify_observers(
                TrainingEvents.LOG_MESSAGE,
                message="No checkpoint found for auto-resume",
                color="yellow",
            )
