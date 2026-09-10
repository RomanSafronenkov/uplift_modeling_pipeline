import sys
import logging

from pathlib import Path


def setup_logging(log_file: Path, logger_name: str) -> logging.Logger:
    """
    Setup logger

    Args:
    - log_file: name of the file where to store logs
    - logger_name: name of the main logger
    """

    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    app_logger = logging.getLogger(logger_name)
    app_logger.setLevel(logging.DEBUG)
    return app_logger
