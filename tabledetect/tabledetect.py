# Imports
import os, subprocess, sys, shutil
from pathlib import Path
from importlib.util import find_spec
from tabledetect.helpers.yolo_to_boundingbox import getBoundingBoxesPerFile
from tabledetect.helpers.boundigbox_to_cropped_image import extractCroppedImages
from tabledetect.helpers.download import downloadRepo, downloadWeights
from tabledetect.helpers.args_classes import StructureArgs

from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
from deskew import determine_skew
import json

# Logging
import logging
logger = logging.getLogger(__name__)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(streamHandler)

# Check if torch and torchvision installed
if (find_spec('torch') is None) or (find_spec('torchvision') is None):
    import GPUtil; gpuCount = len(GPUtil.getAvailable())
    if (os.name == 'nt') and (gpuCount>0):          suggestedPipCommand = 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
    elif (os.name == 'nt') and (gpuCount == 0):     suggestedPipCommand = 'pip install torch torchvision torchaudio'
    else:                                           suggestedPipCommand = ''
    raise ModuleNotFoundError(f'pytorch modules not found, go to https://pytorch.org/get-started/locally/ to install the correct version\nOur best guess: {suggestedPipCommand}')

# Constants
PATH_PACKAGE = os.path.dirname(os.path.realpath(__file__))
PATH_PYTHON = sys.executable
PATH_OUT = os.path.join(PATH_PACKAGE, 'resources', 'examples_out')

PATH_WEIGHTS_DETECT = os.path.join(PATH_PACKAGE, 'resources', 'tabledetect.pt')
PATH_WEIGHTS_DETECT_URL = 'https://www.dropbox.com/s/al2xt5g9vg7g1rk/tabledetect.pt?dl=1'
PATH_EXAMPLES_DETECT = os.path.join(PATH_PACKAGE, 'resources', 'examples_detect')
PATH_ML_MODEL_DETECT = os.path.join(PATH_PACKAGE, 'yolov7-main')
PATH_SCRIPT_DETECT = os.path.join(PATH_PACKAGE, 'yolov7-main', 'detect_codamo.py')
PATH_OUT_DETECT = os.path.join(PATH_OUT)

PATH_WEIGHTS_PARSE = os.path.join(PATH_PACKAGE, 'resources', 'tablestructure.pth')
PATH_WEIGHTS_PARSE_URL = 'https://pubtables1m.blob.core.windows.net/model/pubtables1m_structure_detr_r18.pth'
PATH_EXAMPLES_PARSE = os.path.join(PATH_PACKAGE, 'resources', 'examples_parse')
PATH_ML_MODEL_PARSE = os.path.join(PATH_PACKAGE, 'table-transformer-main')
PATH_SCRIPT_PARSE = os.path.join(PATH_PACKAGE, 'table-transformer-main', 'src')
PATH_CONFIG_PARSE = os.path.join(PATH_SCRIPT_PARSE, 'structure_config.json')
PATH_OUT_PARSE = os.path.join(PATH_OUT, 'out', 'table-structure')

# Detect tables
def detect_table(path_input=PATH_EXAMPLES_DETECT, path_output=PATH_OUT_DETECT, path_weights=PATH_WEIGHTS_DETECT,
                 device='', threshold_confidence=0.5, model_image_size=992, trace='--no-trace',
                 image_format='.png', save_bounding_box_file=True, verbosity=logging.INFO,
                 deskew=True):
    # Parse options
    logger.setLevel(verbosity)
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'

    # Download weights and scripts
    if not os.path.exists(path_weights):
        if os.path.basename(path_weights) == 'tabledetect.pt':
            downloadWeights(url=PATH_WEIGHTS_DETECT_URL, destination=path_weights)
        else:
            raise FileNotFoundError(f'Weights not found at: {path_weights}')
    if not os.path.exists(PATH_SCRIPT_DETECT):
        downloadRepo(url='https://github.com/Danferno/yolov7/archive/master.zip', destination=PATH_PACKAGE)

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

def parse_table(path_input=PATH_EXAMPLES_PARSE, path_weights=PATH_WEIGHTS_PARSE, path_config=PATH_CONFIG_PARSE, path_output=PATH_OUT_PARSE, deskew=True, padding=20, device=None, image_format='.png', verbosity=logging.INFO):
    # Options
    # Options | Verbosity
    logger.setLevel(verbosity)

    # Options | Download weights and scripts
    if not os.path.exists(path_weights):
        if os.path.basename(path_weights) == 'tablestructure.pth':
            downloadWeights(url=PATH_WEIGHTS_PARSE_URL, destination=path_weights)
        else:
            raise FileNotFoundError(f'Weights not found at: {path_weights}')
    if not os.path.exists(PATH_SCRIPT_PARSE):
        downloadRepo(url='https://github.com/Danferno/table-transformer/archive/master.zip', destination=PATH_PACKAGE)

    # Options | Detect GPU
    if not device:
        import torch.cuda
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Options | Create output folder
    os.makedirs(path_output, exist_ok=True)

    # options | Image format dot
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'
    
    # Import parser
    sys.path.append(PATH_SCRIPT_PARSE)
    os.chdir(PATH_SCRIPT_PARSE); from inference import TableExtractionPipeline, output_result
    TableStructurer = TableExtractionPipeline(str_model_path=path_weights, str_config_path=path_config, str_device=device)
    args = StructureArgs(out_dir=path_output)

    # Parse images
    inputPaths = list(os.scandir(path_input))
    for dirEntry in tqdm(inputPaths, desc='Parsing table images'):      # dirEntry = inputPaths[0]
        # Parse | Correct skew
        image = Image.open(dirEntry.path)
        if deskew:
            skewAngle = determine_skew(np.array(image.convert('L')), min_deviation=0.25)
            if skewAngle != 0.0: image = image.rotate(skewAngle, expand=True, fillcolor='white')
        
        # Parse | Pad
        if padding:
            size_padded = tuple(dimension + 2*padding for dimension in image.size)
            image = ImageOps.pad(image, size=size_padded, color='white')

        # Parse | Parse
        image = image.convert('RGB')
        extractedTable = TableStructurer.recognize(img=image, tokens=[], out_cells=True, out_objects=True)
        for key, val in extractedTable.items():
            output_result(key, val, args=args, img=image, img_file=dirEntry.name, img_format=image_format)

