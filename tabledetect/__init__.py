"""Detects tables in images using a custom yolo-v7 model. Very much work in progress"""
__version__ = "0.3.4"
from .tabledetect import detect_table, parse_table
from .helpers import *