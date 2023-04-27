"""Detects tables in images using a custom yolo-v7 model. Parses tables using a standard microsoft/table-transformers model."""
__version__ = "0.4.3"
from .tabledetect import detect_table, parse_table
from . import utils