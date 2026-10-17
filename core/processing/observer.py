import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


class TrainingObserver(ABC):
    """Abstract base class for training observers"""

    @abstractmethod
    def on_epoch_start(self, epoch: int, total_epochs: int):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        pass

    @abstractmethod
    def on_training_start(self):
        pass

    @abstractmethod
    def on_training_end(self, final_metrics: Dict[str, Any]):
        pass


class GenericObserver:
    """Generic observer interface for training events"""

    def log_message(self, message: str, color: str = "white", bold: bool = False):
        """Log a custom message"""
        raise NotImplementedError("Subclasses must implement log_message method")


class ObserverList:
    """
    Unified way to manage a list of observers
    """

    def __init__(self):
        self.__observers: List[TrainingObserver] = []

    def add(self, observer: TrainingObserver):
        """Add an observer to the list"""
        if observer not in self.__observers:
            self.__observers.append(observer)
        return self

    def get(self, observer_class: type) -> Optional[TrainingObserver | GenericObserver]:
        """Get an observer by its class type"""
        for observer in self.__observers:
            if isinstance(observer, observer_class):
                return observer
        return None

    def remove(self, observer: TrainingObserver):
        """Remove an observer from the list"""
        if observer in self.__observers:
            self.__observers.remove(observer)

        return self

    def clear(self):
        """Clear all observers"""
        self.__observers.clear()
        return self

    def __len__(self):
        """Get the number of observers"""
        return len(self.__observers)

    def __iter__(self):
        """Iterate over observers"""
        return iter(self.__observers)

    def __getitem__(self, index: int) -> TrainingObserver:
        """Get an observer by index"""
        if index < 0 or index >= len(self.__observers):
            raise IndexError("Observer index out of range")
        return self.__observers[index]

    def __contains__(self, observer: TrainingObserver) -> bool:
        """Check if an observer is in the list"""
        return observer in self.__observers

    def __str__(self):
        """String representation of the observer list"""
        escape = ",\n "
        return f"ObserverList({len(self.__observers)} observers: {escape.join([obs.__class__.__name__ for obs in self.__observers])})"

    def notify(self, event: str, *args, **kwargs):
        """
        Notify all observers of a specific event
        Calls the method with the given name on each observer

        Args:
            event (str): The name of the event to notify observers about
            *args: Positional arguments to pass to the observer method
            **kwargs: Keyword arguments to pass to the observer method
        """
        # Each observer is isolated: a failing observer (e.g. the plotter) must not
        # prevent the others from being notified nor abort the training flow.
        for observer in self.__observers:
            if not hasattr(observer, event):
                continue
            try:
                getattr(observer, event)(*args, **kwargs)
            except Exception as e:
                import traceback

                print(
                    f"[ObserverList] Error in {observer.__class__.__name__}.{event}: {e}\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
                if event != "log_message":
                    logger = self.get(ConsoleLogger)
                    if logger is not None and logger is not observer:
                        try:
                            logger.log_message(
                                f"Warning: observer {observer.__class__.__name__} failed on '{event}': {e}",
                                "yellow",
                            )
                        except Exception:
                            pass


class ConsoleLogger(TrainingObserver, GenericObserver):
    """Observer that logs training progress to console"""

    def __init__(self, log_file=None):
        self.log_file = log_file
        self.epoch_start_time = None
        self.__check_path()

    def __check_path(self):
        """
        Check if the log file path is valid and create directories if necessary.
        """
        try:
            if self.log_file and isinstance(self.log_file, str):
                import os

                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
        except Exception as e:
            raise RuntimeError(f"Error checking log file path: {e}") from e

    def _log(self, message: str, color: str = "white", bold: bool = False):
        """Internal logging method with color support"""
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "gray": "\033[90m",
            "light_gray": "\033[37m",
            "reset": "\033[0m",
            "bold": "\033[1m",
            "underline": "\033[4m",
        }

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        formatted_message = f"{timestamp}\t{colors.get(color, colors['white'])}{message}{colors['reset']}"

        if bold:
            formatted_message = f"{colors['bold']}{formatted_message}{colors['reset']}"

        print(formatted_message)
        if self.log_file and isinstance(self.log_file, str):
            with open(self.log_file, "a+", encoding="utf-8") as log_file:
                print(formatted_message, file=log_file, flush=True)

    def on_epoch_start(self, epoch: int, total_epochs: int):
        self.epoch_start_time = time.time()
        self._log(f"Starting Epoch {epoch + 1}/{total_epochs}", "cyan")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        message = f"Epoch {epoch + 1} completed in {epoch_time:.1f}s\n\t- "
        # message += "\n\t- ".join([f"{k}: {v:.12f}" for k, v in metrics.items()])

        metric_strings = []
        for k, v in metrics.items():
            try:
                # Se v è un dizionario, formattalo ricorsivamente
                if isinstance(v, dict):
                    dict_str = ", ".join(
                        [
                            f"{sub_k}: {sub_v:.12f}"
                            if isinstance(sub_v, (int, float))
                            else f"{sub_k}: {sub_v}"
                            for sub_k, sub_v in v.items()
                        ]
                    )
                    metric_strings.append(f"{k}: {{{dict_str}}}")
                # Se v è un numero, formattalo con 12 decimali
                elif isinstance(v, (int, float)):
                    metric_strings.append(f"{k}: {v:.12f}")
                # Altrimenti usa la rappresentazione stringa standard
                else:
                    metric_strings.append(f"{k}: {v}")
            except (ValueError, TypeError) as e:
                # Fallback se la formattazione fallisce
                metric_strings.append(f"{k}: {str(v)}")

        message += "\n\t- ".join(metric_strings)

        self._log(message, "green")

    def on_training_start(self):
        self._log("Training started", "magenta")

    def on_training_stop(self):
        self._log("Training stop called", "red")

    def on_training_end(self, final_metrics: Dict[str, Any]):
        self._log("Training completed", "magenta")
        for metric, value in final_metrics.items():
            # self._log(f"Final {metric}: {value:.4f}", 'yellow')
            try:
                if isinstance(value, dict):
                    lines = []
                    for k, v in value.items():
                        if isinstance(v, dict):
                            # Per-class metrics: one line per class
                            sub = ", ".join(
                                [
                                    f"{sk}: {sv:.4f}"
                                    if isinstance(sv, float)
                                    else f"{sk}: {sv}"
                                    for sk, sv in v.items()
                                ]
                            )
                            lines.append(f"\t- {k}: {{{sub}}}")
                        elif isinstance(v, (int, float)):
                            lines.append(f"\t- {k}: {v:.4f}")
                        else:
                            lines.append(f"\t- {k}: {v}")
                    self._log(f"Final {metric}:\n" + "\n".join(lines), "yellow")
                elif isinstance(value, np.ndarray):
                    # Confusion matrix (or any other array) on multiple lines
                    self._log(
                        f"Final {metric}:\n{np.array2string(value, max_line_width=200)}",
                        "yellow",
                    )
                elif isinstance(value, (int, float)):
                    self._log(f"Final {metric}: {value:.4f}", "yellow")
                else:
                    self._log(f"Final {metric}: {value}", "yellow")
            except (ValueError, TypeError):
                self._log(f"Final {metric}: {str(value)}", "yellow")

    def log_message(self, message: str, color: str = "white", bold: bool = False):
        """Log a custom message"""
        self._log(message, color, bold=bold)


class MetricsPlotter(TrainingObserver):
    """Observer that plots training metrics"""

    def __init__(self, save_path: str = "training_plots"):
        self.save_path = save_path
        self.__check_path()
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": {}}

    def __check_path(self):
        """
        Check if the log file path is valid and create directories if necessary.
        """
        try:
            if self.save_path and isinstance(self.save_path, str):
                import os

                log_dir = os.path.dirname(self.save_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
        except Exception as e:
            raise RuntimeError(f"Error checking log file path: {e}") from e

    def on_epoch_start(self, epoch: int, total_epochs: int):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        # Store metrics
        if "train_loss" in metrics:
            self.history["train_loss"].append(metrics["train_loss"])
        if "val_loss" in metrics:
            self.history["val_loss"].append(metrics["val_loss"])

        # Store validation metrics (scalars only: per-class dicts and the confusion
        # matrix are logged on best epochs but cannot be plotted as a curve)
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.number)
            ):
                continue
            if key.startswith("val_") and key != "val_loss":
                metric_name = key.replace("val_", "")
                if metric_name not in self.history["val_metrics"]:
                    self.history["val_metrics"][metric_name] = []
                self.history["val_metrics"][metric_name].append(value)

    def on_training_start(self):
        pass

    def on_training_end(self, final_metrics: Dict[str, Any]):
        try:
            self._plot_metrics()
        except Exception as e:
            print(f"[MetricsPlotter] Could not plot training metrics: {e}", flush=True)
        finally:
            plt.close("all")

    def _plot_metrics(self):
        num_plots = 1 + len(self.history["val_metrics"])
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        if num_plots == 1:
            axes = [axes]

        # Plot loss
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        if self.history["val_loss"]:
            axes[0].plot(self.history["val_loss"], label="Validation Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Plot other metrics
        for i, (metric_name, values) in enumerate(
            self.history["val_metrics"].items(), 1
        ):
            if i < len(axes):
                axes[i].plot(values, label=f"Validation {metric_name}")
                axes[i].set_title(f"{metric_name.capitalize()}")
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(metric_name.capitalize())
                axes[i].legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_path}.png")
        plt.close(fig)


class GradientMonitor(TrainingObserver):
    """Observer that monitors gradient norms"""

    def __init__(self, threshold: float = 1e-6, check_frequency: int = 10):
        self.threshold = threshold
        self.check_frequency = check_frequency
        self.step_count = 0

    def on_epoch_start(self, epoch: int, total_epochs: int):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self, final_metrics: Dict[str, Any]):
        pass

    def check_gradients(self, model):
        """Check gradient norms during training"""
        if self.step_count % self.check_frequency != 0:
            self.step_count += 1
            return

        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm

                if grad_norm < self.threshold:
                    print(f"⚠️  Small gradient: {name} = {grad_norm:.8f}")
                elif grad_norm > 1.0:
                    print(f"🚨 Large gradient: {name} = {grad_norm:.4f}")

        total_grad_norm = torch.sqrt(
            torch.tensor(sum(g**2 for g in grad_norms.values()))
        )
        print(f"Total gradient norm: {total_grad_norm:.6f}")

        self.step_count += 1


class GradientFlowMonitor(TrainingObserver):
    """Observer that monitors gradient flow in the model

    :param log_file: File to log gradient flow information.
    :param json_output: If True, save debug data in JSON format.
    :param thresholds: List of thresholds for gradient norms to categorize their status.
    :param check_frequency: Frequency of checks during training (in steps).
    :type log_file: str
    :type json_output: bool
    :type thresholds: list[float] | None
    :type check_frequency: int
    """

    def __init__(
        self,
        log_file: str,
        json_output: bool = False,
        thresholds: list[float] | None = None,
        check_frequency: int = 10,
    ):
        """
        Initialize the GradientFlowMonitor.

        Args:
            log_file (str): File to log gradient flow information.
            json_output (bool): If True, save debug data in JSON format.
            thresholds (list[float] | None): List of thresholds for gradient norms to categorize their status.
            check_frequency (int): Frequency of checks during training (in steps).
        """
        self.thresholds = (
            thresholds
            if thresholds is not None and len(thresholds) == 4
            else [1e-8, 1e-6, 1e-4, 1.0]
        )

        self.log_file = log_file
        self.json_output = json_output

        self.check_frequency = check_frequency
        self.step_count = 0

        self.__check_path()

    def __check_path(self):
        """
        Check if the log file path is valid and create directories if necessary.
        """
        try:
            if self.log_file and isinstance(self.log_file, str):
                import os

                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
        except Exception as e:
            raise RuntimeError(f"Error checking log file path: {e}") from e

    def on_epoch_start(self, epoch: int, total_epochs: int):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self, final_metrics: Dict[str, Any]):
        pass

    def _internal_log(self, message: str, **kwargs):
        """Internal logging method"""

        if self.log_file is not None:
            with open(self.log_file, "t+a", encoding="utf-8") as f:
                f.write(message + "\n")
        else:
            print(message, **kwargs)

    def debug_gradient_flow(
        self,
        model: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        batch_idx: int,
        input_batch: torch.Tensor,
        epoch: Optional[int] = None,
    ):
        """Enhanced gradient flow debugging"""

        if batch_idx % self.check_frequency != 0:
            return

        if model is None or optimizer is None:
            raise ValueError(
                "Model and optimizer must be provided for gradient flow debugging."
            )

        print_args = {
            "flush": True,
        }

        import traceback

        try:
            model.train()

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_batch)

            loss = torch.mean(output)  # Simple loss for debug

            # Backward pass
            loss.backward()
        except Exception as e:
            self._internal_log(f"Error during forward/backward pass: {e}", **print_args)
            self._internal_log(traceback.format_exc(), **print_args)
            return

        # Collect gradient statistics
        gradient_stats = []
        layer_count = 0
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self._internal_log(f"\n{'=' * 30} {timestamp} {'=' * 30}", **print_args)
        self._internal_log(
            f"- Epoch: {epoch if epoch is not None else 'N/A'}", **print_args
        )
        self._internal_log(f"- Loss: {loss.item():.6f}", **print_args)
        self._internal_log(f"{'=' * 80}", **print_args)
        self._internal_log(
            f"{'Layer Name':<40} {'Grad Norm':<15} {'Param Shape':<20} {'Status':<10}",
            **print_args,
        )
        self._internal_log(f"{'-' * 80}", **print_args)

        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight.grad is not None:
                grad_norm = module.weight.grad.norm().item()  # type: ignore
                param_shape = str(list(module.weight.shape))  # type: ignore

                # Determine status
                if grad_norm < self.thresholds[0]:
                    status = "DEAD"
                    color = "🔴"
                elif grad_norm < self.thresholds[1]:
                    status = "VERY_SMALL"
                    color = "🟠"
                elif grad_norm < self.thresholds[2]:
                    status = "SMALL"
                    color = "🟡"
                elif grad_norm > self.thresholds[3]:
                    status = "LARGE"
                    color = "🟣"
                else:
                    status = "OK"
                    color = "🟢"

                self._internal_log(
                    f"{name:<40} {grad_norm:<15.8f} {param_shape:<20} {color} {status}",
                    **print_args,
                )

                gradient_stats.append(
                    {
                        "name": name,
                        "grad_norm": grad_norm,
                        "param_shape": param_shape,
                        "status": status,
                    }
                )

                layer_count += 1

        # Summary statistics
        grad_norms = [stat["grad_norm"] for stat in gradient_stats]
        status_counts = {}
        if grad_norms:
            self._internal_log(f"\n{'=' * 80}", **print_args)
            self._internal_log("SUMMARY STATISTICS", **print_args)
            self._internal_log(f"{'=' * 80}", **print_args)
            self._internal_log(f"Total layers analyzed: {layer_count}", **print_args)
            self._internal_log(
                f"Mean gradient norm: {np.mean(grad_norms):.8f}", **print_args
            )
            self._internal_log(
                f"Std gradient norm: {np.std(grad_norms):.8f}", **print_args
            )
            self._internal_log(
                f"Min gradient norm: {np.min(grad_norms):.8f}", **print_args
            )
            self._internal_log(
                f"Max gradient norm: {np.max(grad_norms):.8f}", **print_args
            )

            # Count by status
            for stat in gradient_stats:
                status = stat["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

            self._internal_log("\nStatus distribution:", **print_args)
            for status, count in status_counts.items():
                percentage = (count / layer_count) * 100
                self._internal_log(
                    f"  {status}: {count} ({percentage:.1f}%)", **print_args
                )

        # Save file if requested
        if self.log_file and self.json_output:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            fname = self.log_file.removesuffix(".json")
            filename = f"{fname}_{timestamp}.json"

            debug_data = {
                "timestamp": timestamp,
                "loss": loss.item(),
                "gradient_stats": gradient_stats,
                "summary": {
                    "total_layers": layer_count,
                    "mean_grad_norm": np.mean(grad_norms) if grad_norms else 0,
                    "std_grad_norm": np.std(grad_norms) if grad_norms else 0,
                    "min_grad_norm": np.min(grad_norms) if grad_norms else 0,
                    "max_grad_norm": np.max(grad_norms) if grad_norms else 0,
                    "status_counts": status_counts,
                },
            }

            with open(filename, "x", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2)

            self._internal_log(
                f"\nGradient debug data saved to: {filename}", **print_args
            )

        self._internal_log(f"{'=' * 80}\n", **print_args)


class ImprovedGradientMonitor(TrainingObserver):
    """
    Observer that monitors and scales small gradients
    This observer checks the gradients of the model parameters and applies scaling
    to those that are below a specified threshold.
    It also tracks the ratio of small gradients and the total gradient norm."""

    def __init__(self, threshold=1e-6, check_frequency=10, apply_scaling=True):
        self.threshold = threshold
        self.check_frequency = check_frequency
        self.apply_scaling = apply_scaling
        self.step_count = 0
        self.gradient_history = []

    def on_epoch_start(self, epoch: int, total_epochs: int):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self, final_metrics: Dict[str, Any]):
        pass

    def check_gradients(self, model):
        if self.step_count % self.check_frequency != 0:
            self.step_count += 1
            return

        grad_norms = {}
        small_grad_count = 0
        total_params = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                total_params += 1

                if grad_norm < self.threshold:
                    small_grad_count += 1
                    if self.apply_scaling:
                        # Applica scaling ai gradienti troppo piccoli
                        param.grad *= 10.0
                        print(
                            f"🔧 Scaled gradient: {name} = {grad_norm:.8f} -> {grad_norm * 10:.8f}"
                        )

        small_grad_ratio = small_grad_count / total_params if total_params > 0 else 0
        total_grad_norm = torch.sqrt(
            torch.tensor(sum(g**2 for g in grad_norms.values()))
        )

        print(
            f"📊 Gradient Stats: {small_grad_count}/{total_params} small ({small_grad_ratio:.2%})"
        )
        print(f"📊 Total gradient norm: {total_grad_norm:.6f}")

        self.gradient_history.append(
            {
                "step": self.step_count,
                "small_ratio": small_grad_ratio,
                "total_norm": total_grad_norm.item(),
            }
        )

        # Allarme se troppi gradienti piccoli
        if small_grad_ratio > 0.5:
            print("🚨 WARNING: >50% of gradients are too small!")

        self.step_count += 1


class WandbObserver(TrainingObserver, GenericObserver):
    """Wandb observer for experiment tracking"""

    def __init__(self, config, experiment_config):
        self.config = config
        self.experiment_config = experiment_config
        self.wandb = None
        self.run = None
        self._is_initialized = False

    def _ensure_wandb(self):
        """Lazy import and initialization of wandb"""
        if self.wandb is None:
            try:
                import wandb

                self.wandb = wandb

                # Check if wandb is properly configured
                try:
                    # Try to access the API key to verify authentication
                    _ = wandb.api.api_key
                except Exception:
                    print("[WandbObserver] Warning: Wandb API key not found.")
                    print(
                        "[WandbObserver] Run 'wandb login' to authenticate, or set WANDB_API_KEY environment variable."
                    )
                    print(
                        "[WandbObserver] You can also run in offline mode with --wandb-mode offline"
                    )
                    if self.config.mode != "offline":
                        return False

                return True
            except ImportError:
                print(
                    "[WandbObserver] Wandb not installed. Install with: pip install wandb"
                )
                return False
        return True

    def _initialize_wandb(self):
        """Initialize wandb run"""
        if self._is_initialized or not self.config.enabled:
            return

        if not self._ensure_wandb():
            return

        try:
            # Validate configuration
            if not self.config.project:
                print(
                    "[WandbObserver] Warning: No project specified. Using default project."
                )

            # Setup run configuration
            run_config = {
                "model_config": self.experiment_config.model_config.to_dict()
                if hasattr(self.experiment_config.model_config, "to_dict")
                else {},
                "training_config": self.experiment_config.training_config.to_dict()
                if hasattr(self.experiment_config.training_config, "to_dict")
                else {},
                "scheduler_config": self.experiment_config.scheduler_config.to_dict()
                if hasattr(self.experiment_config.scheduler_config, "to_dict")
                else {},
                "data_config": self.experiment_config.data_config.to_dict()
                if hasattr(self.experiment_config.data_config, "to_dict")
                else {},
                "experiment_name": self.experiment_config.name,
                "model_type": self.experiment_config.model_type,
                "loss_type": self.experiment_config.loss_type.value
                if hasattr(self.experiment_config.loss_type, "value")
                else str(self.experiment_config.loss_type),
            }

            # Initialize run with better error handling
            init_kwargs = {
                "project": self.config.project,
                "name": self.config.name or self.experiment_config.name,
                "config": run_config,
                "tags": self.config.tags,
                "notes": self.config.notes,
                "save_code": self.config.save_code,
                "resume": "allow",  # Changed from deprecated 'reinit' to 'resume'
            }

            # Add entity only if specified (avoids permission issues with default entity)
            if self.config.entity:
                init_kwargs["entity"] = self.config.entity

            self.run = self.wandb.init(**init_kwargs)

            self._is_initialized = True
            print(f"[WandbObserver] Initialized run: {self.run.name}")

        except Exception as e:
            error_msg = str(e)
            if "permission denied" in error_msg.lower():
                print(
                    f"[WandbObserver] Permission denied. Check project access or try without --wandb-entity"
                )
                print(
                    f"[WandbObserver] You can create a new project or use a different project name"
                )
            elif "does not exist" in error_msg.lower():
                print(
                    f"[WandbObserver] Project or entity does not exist. Check project name and entity"
                )
            else:
                print(f"[WandbObserver] Failed to initialize: {e}")
            print(f"[WandbObserver] Training will continue without Wandb logging")

    def on_training_start(self):
        """Called when training starts"""
        self._initialize_wandb()

    def on_epoch_start(self, epoch: int, total_epochs: int):
        """Called at the start of each epoch"""
        # Nothing to do at epoch start for Wandb
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch"""
        if not self._is_initialized:
            return

        if epoch % self.config.log_freq == 0:
            try:
                # Log all metrics provided
                log_dict = {
                    "epoch": epoch,
                    **metrics,  # Include all metrics passed in
                }

                self.wandb.log(log_dict, step=epoch)

            except Exception as e:
                print(f"[WandbObserver] Failed to log metrics: {e}")

    def on_training_end(self, final_metrics: Dict[str, Any]):
        """Called when training ends"""
        if self._is_initialized and self.run:
            try:
                # Log final metrics if provided
                if final_metrics:
                    self.wandb.log({f"final_{k}": v for k, v in final_metrics.items()})

                self.wandb.finish()
                print("[WandbObserver] Run finished")
            except Exception as e:
                print(f"[WandbObserver] Failed to finish run: {e}")

    def log_message(self, message: str, color: str = None, bold: bool = False):
        """Log message (required by GenericObserver interface)"""
        # Wandb doesn't handle console messages, so we just pass
        pass

    def log_model_parameters(self, model):
        """Log model parameters"""
        if not self._is_initialized or not self.config.log_parameters:
            return

        try:
            self.wandb.watch(model, log="parameters", log_freq=self.config.log_freq)
        except Exception as e:
            print(f"[WandbObserver] Failed to log parameters: {e}")

    def debug_gradient_flow(
        self,
        model: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        batch_idx: int,
        input_batch: torch.Tensor,
        epoch: Optional[int] = None,
    ):
        """Log model gradients to Wandb"""
        if not self._is_initialized or not self.config.log_gradients:
            return

        # Only log gradients at specified frequency to avoid overwhelming Wandb
        if batch_idx % self.config.log_freq != 0:
            return

        try:
            # Use wandb.watch for gradient logging
            self.wandb.watch(model, log="gradients", log_freq=1)

            # Optionally log gradient statistics
            if (
                hasattr(self.config, "log_gradient_stats")
                and self.config.log_gradient_stats
            ):
                grad_stats = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_stats[f"grad_norm/{name}"] = grad_norm

                # Log gradient statistics to wandb
                self.wandb.log(
                    grad_stats, step=epoch if epoch is not None else batch_idx
                )

        except Exception as e:
            print(f"[WandbObserver] Failed to log gradients: {e}")

    def log_gradients(self, model):
        """Legacy method - kept for backward compatibility"""
        if not self._is_initialized or not self.config.log_gradients:
            return

        try:
            self.wandb.watch(model, log="gradients", log_freq=self.config.log_freq)
        except Exception as e:
            print(f"[WandbObserver] Failed to log gradients: {e}")
