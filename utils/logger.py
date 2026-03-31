"""
utils/logger.py — Centralized logging for CarIQ
Creates file + console loggers with rotation. All modules use:
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from config import LOG_DIR

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE    = os.path.join(LOG_DIR, "cariq.log")
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root logger configuration (runs once on import)
_root_logger = logging.getLogger("cariq")
if not _root_logger.handlers:
    _root_logger.setLevel(logging.DEBUG)

    # File handler with rotation (5 MB × 3 backups)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    _root_logger.addHandler(fh)

    # Console handler (INFO and above)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    _root_logger.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'cariq' namespace."""
    return _root_logger.getChild(name)
