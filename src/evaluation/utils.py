import logging
import os


def setup_loggers(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    blank_logger = logging.getLogger("blank")
    blank_logger.setLevel(logging.INFO)
    if not blank_logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        blank_logger.addHandler(file_handler)

    return logger, blank_logger
