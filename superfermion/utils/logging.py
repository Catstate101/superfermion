"""
Superfermion Logging — Structured and beautiful console output.
"""

import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def get_logger(name: str = "superfermion") -> logging.Logger:
    """Get a structured logger for Superfermion."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        if _HAS_RICH:
            console = Console()
            handler = RichHandler(
                console=console, 
                rich_tracebacks=True,
                markup=True,
                show_time=True
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            
        logger.addHandler(handler)
        
    return logger


# Default global logger
logger = get_logger()


def set_level(level: str):
    """Set global log level (DEBUG, INFO, WARNING, ERROR)."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)


def debug(msg: str): logger.debug(msg)
def info(msg: str): logger.info(msg)
def warning(msg: str): logger.warning(msg)
def error(msg: str): logger.error(msg)
