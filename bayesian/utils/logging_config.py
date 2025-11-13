import logging
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file=None, level=logging.INFO, console=True):
    """Create and return a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Avoid duplicate handlers
    if not logger.handlers:
        if log_file:
            fh = logging.FileHandler(os.path.join(LOG_DIR, log_file), mode='w')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    return logger