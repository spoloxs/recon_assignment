import logging
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

class ColoredFormatter(logging.Formatter):
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG:    RESET + FORMAT + RESET,
        logging.INFO:     GREEN + FORMAT + RESET,
        logging.WARNING:  YELLOW + FORMAT + RESET,
        logging.ERROR:    RED + FORMAT + RESET,
        logging.CRITICAL: BOLD + RED + FORMAT + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

def setup_logger(name="ReconApp"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
        
    return logger
