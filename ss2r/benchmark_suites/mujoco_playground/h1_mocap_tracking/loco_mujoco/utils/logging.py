import logging
import sys


def setup_logger(name, level=logging.INFO, identifier="[LocoMuJoCo]"):
    """
    Create and return a configured logger.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        identifier (str): Identifier to prepend to all log messages.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(level)

        # Define a custom formatter
        formatter = logging.Formatter(f"{identifier} %(levelname)s: %(message)s")
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger
