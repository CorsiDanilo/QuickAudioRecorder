import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


LOG_FILENAME = "QuickAudioRecorder.log"
_HANDLER_MARKER = "_quick_audio_recorder"


def default_log_folder():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _create_handler(folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, LOG_FILENAME)
    handler = TimedRotatingFileHandler(
        path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    handler._quick_audio_recorder = True
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def configure_logging(log_folder=None):
    requested_folder = os.path.abspath(log_folder or default_log_folder())
    fallback_folder = os.path.abspath(default_log_folder())

    try:
        handler = _create_handler(requested_folder)
        using_fallback = False
    except OSError:
        if requested_folder == fallback_folder:
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception(
                "Unable to create the log file in %s", requested_folder
            )
            return None
        handler = _create_handler(fallback_folder)
        using_fallback = True

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for existing in list(root_logger.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root_logger.removeHandler(existing)
            existing.close()
    root_logger.addHandler(handler)

    logger = logging.getLogger(__name__)
    if using_fallback:
        logger.warning(
            "Unable to use configured log folder %s; using %s",
            requested_folder,
            fallback_folder,
        )
    logger.info("Logging configured: %s", handler.baseFilename)
    return handler.baseFilename
