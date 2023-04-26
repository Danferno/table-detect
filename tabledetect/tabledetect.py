# Imports
import os, subprocess, sys, shutil
from pathlib import Path
from importlib.util import find_spec
from tabledetect.helpers.yolo_to_boundingbox import getBoundingBoxesPerFile
from tabledetect.helpers.boundigbox_to_cropped_image import extractCroppedImages
from tabledetect.helpers.download import downloadRepo, downloadWeights

# Logging
import logging
logger = logging.getLogger(__name__)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(streamHandler)

# Check if torch and torchvision installed
if (find_spec('torch') is None) or (find_spec('torchvision') is None):
    raise ModuleNotFoundError('pytorch modules not found, go to https://pytorch.org/get-started/locally/ to install the correct version')

# Constants
PATH_PACKAGE = os.path.dirname(os.path.realpath(__file__))
PATH_WEIGHTS = os.path.join(PATH_PACKAGE, 'resources', 'tabledetect.pt')
PATH_WEIGHTS_URL = 'https://www.dropbox.com/s/al2xt5g9vg7g1rk/tabledetect.pt?dl=1'
PATH_EXAMPLES_DETECT = os.path.join(PATH_PACKAGE, 'resources', 'examples_detect')
PATH_EXAMPLES_PARSE = os.path.join(PATH_PACKAGE, 'resources', 'examples_parse')
PATH_PYTHON = sys.executable
PATH_OUT = os.path.join(PATH_PACKAGE, 'resources', 'examples_out')

PATH_ML_MODEL = os.path.join(PATH_PACKAGE, 'yolov7-main')
PATH_SCRIPT_DETECT = os.path.join(PATH_PACKAGE, 'yolov7-main', 'detect_codamo.py')
if not os.path.exists(PATH_SCRIPT_DETECT):
    downloadRepo(url='https://github.com/Danferno/yolov7/archive/master.zip', destination=PATH_PACKAGE)

# Detect tables
def detect_table(path_input=PATH_EXAMPLES_DETECT, path_output=PATH_OUT, path_weights=PATH_WEIGHTS,
                 device='', threshold_confidence=0.5, model_image_size=992, trace='--no-trace',
                 image_format='.png', save_bounding_box_file=True, verbosity=logging.INFO,
                 deskew=True):
    # Parse options
    logger.setLevel(verbosity)
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'
    if not os.path.exists(path_weights):
        if os.path.basename(path_weights) == 'tabledetect.pt':
            downloadWeights(url=PATH_WEIGHTS_URL, destination=path_weights)
        else:
            raise FileNotFoundError(f'Weights not found at: {path_weights}')

    # Detect
    logger.info('Detecting objects in your source files')
    pathMlOutput = os.path.join(path_output, 'out')
    if os.path.exists(pathMlOutput):
        shutil.rmtree(pathMlOutput)
    os.makedirs(path_output, exist_ok=True)
    command = f'{PATH_PYTHON} "{PATH_SCRIPT_DETECT}"' \
                f' --weights {path_weights}' \
                f' --conf {threshold_confidence}' \
                f' --img-size {model_image_size}' \
                f' --source {path_input}' \
                f' --save-txt --save-conf' \
                f' --project out --name table-detect' \
                f' {device}' \
                f' {trace}'
    subprocess.run(command, check=True, cwd=path_output)
    logger.info(f'path_output: {path_output}')

    # Extract bounding boxes
    logger.info('Extracting bounding box information from the YOLO files')
    bbox_lists_per_file = [getBoundingBoxesPerFile(annotationfile.path) for annotationfile in os.scandir(os.path.join(path_output, 'out', 'table-detect', 'labels'))]

    # Crop images
    logger.info('Extracting cropped images and saving single bounding box json file')
    path_cropped_output = os.path.join(path_output, 'out', 'table-detect', 'cropped')
    extractCroppedImages(bbox_lists_per_file_list=bbox_lists_per_file, outDir=path_cropped_output, imageFormat=image_format, imageDir=path_input, saveBoundingBoxFile=save_bounding_box_file)

def parse_table(path_input):
    ...
    