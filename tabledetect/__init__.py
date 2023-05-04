"""Detects tables in images using a custom yolo-v7 model. Parses tables using a standard microsoft/table-transformers model."""
__version__ = "0.5.4"
from .tabledetect import detect_table, parse_table
from . import utils