from logging import Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL, getLogger, StreamHandler
import sys

class ColoredFormatter(Formatter):
    """
    Custom formatter to add colors to log messages based on their level.
    """
    grey = '\033[90m'
    yellow = '\033[93m'
    red = '\033[31m'
    bold = '\033[1m'
    blue = '\033[34m'
    white_light = '\033[97m'
    reset = '\033[0m'

    f = "[%(levelname)s][%(name)s] "
    t = "[%(asctime)s]"
    msg = "%(message)s"

    FORMATS = {
        DEBUG: grey + t + reset + white_light + f + reset + msg,
        INFO: grey + t + reset + blue + bold + f + reset + msg,
        WARNING: grey + t + reset + yellow + bold + f + reset + msg,
        ERROR: grey + t + reset + red + f + msg + reset,
        CRITICAL: grey + t + reset + bold + red + f + msg + reset
    }

    def format(self, record):
        """
        Format the log record with color based on the log level.
        :param record:
        :return:
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = Formatter(log_fmt)
        return formatter.format(record)
    

def configure_logging(level=DEBUG,
                      handlers=None,
                      propagate=False):
    """
    Configure logging for the core.extraction module.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        handlers: Handlers list
        propagate: If True, propagate messages to the root logger
    """
    logger = getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = propagate

    for h in list(logger.handlers):
        logger.removeHandler(h)

    if logger.hasHandlers():
        logger.handlers.clear()

    if handlers:
        for handler in handlers:
            logger.addHandler(handler)
    else:
        handler = StreamHandler(stream=sys.stdout)
        formatter = ColoredFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
