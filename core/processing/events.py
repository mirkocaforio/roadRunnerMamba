"""
Vengono definiti gli eventi standard per il training.
Per adattare un observer a questi eventi, è necessario implementare i metodi corrispondenti ai valori degli eventi.
"""


class TrainingEvents:
    """Training events"""
    ON_TRAINING_START = "on_training_start"
    ON_TRAINING_STOP = "on_training_stop"
    ON_TRAINING_END = "on_training_end"
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_VALIDATION_START = "on_validation_start"
    ON_VALIDATION_END = "on_validation_end"
    ON_TEST_START = "on_test_start"
    ON_TEST_END = "on_test_end"
    DEBUG_GRADIENT_FLOW_MONITOR = "debug_gradient_flow"
    LOG_MESSAGE = "log_message"
    LOG_ERROR = "log_error"
    LOG_WARNING = "log_warning"

    @classmethod
    def get_events(cls):
        """Get all training events"""
        return [value for key, value in vars(cls).items() if not key.startswith('__') and not callable(value)]
