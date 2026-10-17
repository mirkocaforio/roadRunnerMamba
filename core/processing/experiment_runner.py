"""
Signal handling and experiment runner utilities
"""
import os
import signal
from typing import Optional, Callable, Any
from contextlib import contextmanager

from .training import ModelTrainer
from .object import TrainingResult
from .experiment import Experiment


class SignalHandler:
    """Handles graceful shutdown signals"""

    def __init__(self):
        self.trainer: Optional[ModelTrainer] = None
        self.rank: int = int(os.environ.get("RANK", 0))
        self.shutdown_callbacks: list[Callable] = []
        self._original_handlers = {}

    def set_trainer(self, trainer: ModelTrainer):
        """Set the trainer instance"""
        self.trainer = trainer

    def add_shutdown_callback(self, callback: Callable):
        """Add a callback to execute on shutdown"""
        self.shutdown_callbacks.append(callback)

    def signal_handler(self, signum: int, _frame: Any = None):
        """Handle shutdown signals gracefully"""
        signal_name = signal.Signals(signum).name

        if self.trainer is not None and self.rank == 0:
            print(f"[{self.rank}] Received signal {signal_name} ({signum}), stopping training...")
            self.trainer.stop()

        # Execute shutdown callbacks
        for callback in self.shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in shutdown callback: {e}")

    def setup_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]

        # Add SIGQUIT only on Unix systems
        if hasattr(signal, 'SIGQUIT'):
            signals_to_handle.append(signal.SIGQUIT)

        for sig in signals_to_handle:
            self._original_handlers[sig] = signal.signal(sig, self.signal_handler)

    def restore_handlers(self):
        """Restore original signal handlers"""
        for sig, original_handler in self._original_handlers.items():
            signal.signal(sig, original_handler)
        self._original_handlers.clear()


@contextmanager
def experiment_context(signal_handler: Optional[SignalHandler] = None):
    """Context manager for running experiments with signal handling"""
    if signal_handler is None:
        signal_handler = SignalHandler()

    signal_handler.setup_handlers()

    try:
        yield signal_handler
    except KeyboardInterrupt:
        print("Experiment interrupted by user")
        raise
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise
    finally:
        # Restore original signal handlers
        signal_handler.restore_handlers()


class ExperimentRunner:
    """High-level experiment runner with signal handling and logging"""

    def __init__(self, logger: Callable, verbose: bool = True):
        self.signal_handler = SignalHandler()
        self.verbose = verbose
        self.logger = logger

    def run_experiment(self, experiment: Experiment, log_results: bool = True):
        """Run an experiment with proper signal handling and logging"""

        with experiment_context(self.signal_handler) as handler:
            try:
                # Setup experiment
                if self.verbose:
                    print(f"Setting up experiment: {experiment.config.name}")

                experiment.setup()

                # Set trainer for signal handling
                trainer = experiment.get_trainer()
                if trainer:
                    handler.set_trainer(trainer)

                local_rank = getattr(experiment, 'local_rank', 0)

                # Run experiment
                if self.verbose and local_rank == 0 and self.logger:
                    self.logger(f"Running experiment: {experiment.config.name}")

                results = experiment.run()

                # Log results
                if log_results and local_rank == 0:
                    self._log_results(results, trainer)

                return results

            except KeyboardInterrupt:
                if self.verbose and self.logger:
                    self.logger("Experiment interrupted by user")
                return None
            except Exception as e:
                local_rank = getattr(experiment, 'local_rank', 0)

                if local_rank == 0 and self.logger:
                    self.logger(f"An error occurred during training: {e}")
                    import traceback
                    self.logger("Full traceback:")
                    self.logger(traceback.format_exc())  # ✅ Stampa lo stack trace completo
                return None

    def _log_results(self, results: TrainingResult | None, trainer):
        """Log experiment results"""

        if not trainer or not results:
            if self.verbose and self.logger:
                self.logger("No trainer or results to log")
            return

        if self.verbose and self.logger:
            self.logger("\n" + "=" * 50)
            self.logger("EXPERIMENT RESULTS")
            self.logger("=" * 50)
            self.logger(str(results))
            self.logger("=" * 50)
