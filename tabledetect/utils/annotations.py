# Imports
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path
from tqdm import tqdm
from lxml import etree
from math import ceil

TEST = False
if TEST:
    path_images = r"F:\ml-parsing-project\data\parse_activelearning1_jpg\demos\images"
    path_labels = r"F:\ml-parsing-project\data\parse_activelearning1_jpg\demos\labels_jobmunene"
    path_output = 

# Functions
def visualise_annotation(path_images, path_labels, path_output, annotation_type):
    # Options | Paths
    path_images = Path(path_images); path_labels = Path(path_labels); path_labels = Path(path_labels)
    
    # Options | Annotation type
    colors = ['#FF4136', '#2ECC40', '#0074D9', '#FFDC00', '#B10DC9', '#FF851B', 'black']
    if annotation_type == 'tabledetect':
        labels = ['table-noborders', 'table-fullborders','table-partialborders']
    elif annotation_type == 'tableparse':
        labels = ['table', 'table column','table row', 'table column header', 'table projected row header', 'table spanning cell']
    else:
        raise Exception(f'annotation_type {annotation_type} not supported. Currently implemented: tabledetect, tableparse')
    colorMap = {key: colors[i] for i, key in enumerate(labels)} 

