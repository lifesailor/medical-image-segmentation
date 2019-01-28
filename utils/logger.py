import os, logging


def logger(name, path):
    """
    define logger

    :param name: name of logger
    :param path: path to log
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file handler
    file_handler = logging.FileHandler(path)

    logger.addHandler(file_handler)
    logger.info("starts logging")

    return logger



