"""Detects tables in images using a custom yolo-v7 model"""
__version__ = "0.5.6"
from .tabledetect import detect_table, parse_table
from . import utils