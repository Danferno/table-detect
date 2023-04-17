# Imports
import os, subprocess, sys, shutil
from pathlib import Path
from tabledetect.helpers.yolo_to_boundingbox import getBoundingBoxesPerFile
from tabledetect.helpers.boundigbox_to_cropped_image import extractCroppedImages
from tabledetect.helpers.download import downloadRepo, downloadWeights
import logging; logging.basicConfig(level=logging.INFO)

# Check if torch and torchvision installed
try:
    import torch
    import torchvision
except ModuleNotFoundError:
    raise ModuleNotFoundError('pytorch module not found, go to https://pytorch.org/get-started/locally/ to install the correct version')

# Constants
PATH_PACKAGE = os.path.dirname(os.path.realpath(__file__))
PATH_WEIGHTS = os.path.join(PATH_PACKAGE, 'resources', 'tabledetect.pt')
PATH_WEIGHTS_URL = 'https://www.dropbox.com/s/k1iuhwk2k62uifb/tabledetect.pt?dl=1'
PATH_EXAMPLES = os.path.join(PATH_PACKAGE, 'resources', 'examples')
PATH_PYTHON = sys.executable
PATH_OUT = os.path.join(PATH_PACKAGE, 'resources', 'examples_out')

PATH_ML_MODEL = os.path.join(PATH_PACKAGE, 'yolov7-main')
PATH_SCRIPT_DETECT = os.path.join(PATH_PACKAGE, 'yolov7-main', 'detect_codamo.py')
if not os.path.exists(PATH_SCRIPT_DETECT):
    downloadRepo(url='https://github.com/Danferno/yolov7/archive/master.zip', destination=PATH_PACKAGE)

if not os.path.exists(PATH_WEIGHTS):
    downloadWeights(url=PATH_WEIGHTS_URL, destination=PATH_WEIGHTS)


# Detect tables
def detect_table(path_input=PATH_EXAMPLES, path_cropped_output=os.path.join(PATH_OUT, 'cropped'), device=None, threshold_confidence=0.5, model_image_size=992, trace='--no-trace', image_format='.png', save_bounding_box_file=True, verbosity=logging.INFO):
    # Parse options
    logging.basicConfig(level=verbosity)
    logging.debug('Checking if torch is properly installed')
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'

    # Detect
    logging.info('Detecting objects in your source files')
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
    subprocess.run(command, check=True)

    # Extract bounding boxes
    logging.info('Extracting bounding box information from the YOLO files')
    bbox_lists_per_file = [getBoundingBoxesPerFile(annotationfile.path) for annotationfile in os.scandir(os.path.join(PATH_OUT, 'table-detect', 'labels'))]

    # Crop images
    logging.info('Extracting cropped images and saving single bounding box json file')
    extractCroppedImages(bbox_lists_per_file_list=bbox_lists_per_file, outDir=path_cropped_output, imageFormat=image_format, imageDir=path_input, saveBoundingBoxFile=save_bounding_box_file)



    