# Imports
import os, subprocess, sys, shutil
from pathlib import Path
from tabledetect.helpers.flatten_list import flattenList
from tabledetect.helpers.yolo_to_boundingbox import getBoundingBoxesPerFile
from tabledetect.helpers.boundigbox_to_cropped_image import extractCroppedImages
from PIL import Image

# Check if torch and torchvision installed
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('pytorch module not found, go to https://pytorch.org/get-started/locally/ to install the correct version')

# Constants
PATH_PACKAGE = os.path.dirname(os.path.realpath(__file__))
PATH_WEIGHTS = os.path.join(PATH_PACKAGE, 'resources', 'best.pt')
PATH_SCRIPT_DETECT = os.path.join(PATH_PACKAGE, 'yolov7', 'detect.py')
PATH_EXAMPLES = os.path.join(PATH_PACKAGE, 'resources', 'examples')
PATH_PYTHON = sys.executable
PATH_OUT = Path(PATH_PACKAGE).parent / 'out'

# Detect tables
def detect_table(path_input=PATH_EXAMPLES, path_cropped_output=os.path.join(PATH_OUT, 'cropped'), device=None, threshold_confidence=0.5, model_image_size=992, trace='--no-trace', image_format='.png', save_bounding_box_file=True):
    # Parse options
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'
    
    # Detect
    if os.path.exists(PATH_OUT):
        shutil.rmtree(PATH_OUT)
    command = f'{PATH_PYTHON} "{PATH_SCRIPT_DETECT}"' \
                f' --weights {PATH_WEIGHTS}' \
                f' --conf {threshold_confidence}' \
                f' --img-size {model_image_size}' \
                f' --source {path_input}' \
                f' --save-txt --save-conf' \
                f' --project out --name table-detect' \
                f' {trace}'
    subprocess.call(command)

    # Extract bounding boxes   
    bbox_lists_per_file = [getBoundingBoxesPerFile(annotationfile.path) for annotationfile in os.scandir(os.path.join(PATH_OUT, 'table-detect', 'labels'))]

    # Crop images
    extractCroppedImages(bbox_lists_per_file_list=bbox_lists_per_file, outDir=path_cropped_output, imageFormat=image_format, imageDir=path_input, saveBoundingBoxFile=save_bounding_box_file)



    