# Imports
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageColor
from pathlib import Path
from tqdm import tqdm
from lxml import etree
from math import ceil
import time
import logging
from typing import Literal, TypedDict

class AnnotationFormat(TypedDict):
    labelFormat: Literal['voc', 'yolo']
    labels: list
    classMap: dict
    split_annotation_types: bool
    show_labels: bool
    as_area: bool

TEST = False
if TEST:
    annotation_type = 'tableparse'
    path_images = rf"F:\ml-parsing-project\table-detect\tabledetect\resources\examples_visualise\{annotation_type}\images"
    path_labels = rf"F:\ml-parsing-project\table-detect\tabledetect\resources\examples_visualise\{annotation_type}\labels"
    path_output = rf"F:\ml-parsing-project\table-detect\tabledetect\resources\examples_visualise\{annotation_type}\images_annotated"
    

# Helper functions
def __guessProperFontsize(x0, x1, y0, y1):
    distance_x = (x1-x0)/2
    width = ceil(distance_x/200)
    fontsize = width * 10
    return width, fontsize

def __voc_to_pilBoxes(vocPath, colorMap):
    pilBoxes = []
    for _, el in etree.iterparse(vocPath, tag=['object']):      # _, el = next(etree.iterparse(vocPath, tag=['object']))
        bbox = el.find('bndbox')
        x0 = float(bbox.find('xmin').text)
        y0 = float(bbox.find('ymin').text)
        x1 = float(bbox.find('xmax').text)
        y1 = float(bbox.find('ymax').text)
        label = el.find('name').text
        width, fontsize = __guessProperFontsize(x0, x1, y0, y1)

        pilBoxes.append({'xy': [x0, y0, x1, y1], 'outline': colorMap[label], 'width': width,
                         'label': {'text': label, 'size': fontsize},})
    return pilBoxes

def __yolo_to_pilBox(yoloPath, targetImage, classMap, colorMap):  
    targetWidth, targetHeight = targetImage.size
    pilBoxes = []
    with open(yoloPath, 'r') as yoloFile:
        for annotationLine in yoloFile:         # annotationLine = yoloFile.readline()
            cat, xc, yc, w, h = [float(string.strip('\n')) for string in annotationLine.split(' ')]
            
            x0 = (xc - w/2) * targetWidth
            x1 = (xc + w/2) * targetWidth
            y0 = (yc - h/2) * targetHeight
            y1 = (yc + h/2) * targetHeight
            width, fontsize = __guessProperFontsize(x0, x1, y0, y1)

            label = classMap[cat]

            pilBoxes.append({'xy': [x0, y0, x1, y1], 'outline': colorMap[label], 'width': width,
                            'label': {'text': label, 'size': fontsize},})
    return pilBoxes

def __annotation_to_pilBox(sourcePath:Path, labelFormat, targetImage, classMap, colorMap):
    match labelFormat:
        case 'voc':
            label_extension = '.xml'
            pilBoxes = __voc_to_pilBoxes(vocPath=sourcePath.with_suffix(label_extension), colorMap=colorMap)
        case 'yolo':
            label_extension = '.txt'
            pilBoxes = __yolo_to_pilBox(yoloPath=sourcePath.with_suffix(label_extension), targetImage=targetImage, classMap=classMap, colorMap=colorMap)
        case 'coco':
            ...
    
    return pilBoxes

def __visualise(image, annotations, min_width, show_labels, as_area):
    overlay = Image.new('RGBA', image.size, (0,0,0,0))
    annotatedImg = ImageDraw.Draw(overlay, 'RGBA')
    for annotation in annotations:      # annotation = annotations[0]
        label = annotation.get('label')
        label_color = annotation.get('outline') or 'black'
        label_color_transparent = ImageColor.getrgb(annotation['outline']) + (50,)

        label_font = ImageFont.truetype('arial.ttf', size=label['size'])
        label_text = label['text']

        label_pos_x = (annotation['xy'][0] + annotation['xy'][2] - annotatedImg.textlength(text=label_text, font=label_font))/2
        label_pos_y = (annotation['xy'][1] + annotation['xy'][3] - label['size'])/2
        label_bbox = annotatedImg.textbbox(xy=(label_pos_x, label_pos_y), text=label_text, font=label_font)
        if show_labels:
            annotatedImg.rectangle(xy=label_bbox, fill=(255, 255, 255, 128))
            annotatedImg.text(xy=(label_pos_x, label_pos_y), text=label_text, fill=label_color, font=label_font)

        if as_area:
            annotatedImg.rectangle(xy=annotation['xy'], fill=label_color_transparent, width=annotation['width'])
        else:
            annotatedImg.rectangle(xy=annotation['xy'], outline=label_color_transparent, width=annotation['width'])

    if min_width:
        scale = max(800/image.width, 1)
        image = ImageOps.scale(image=image, factor=scale)

    return Image.alpha_composite(image, overlay).convert('RGB')


# Functions
def visualise_annotation(path_images, path_labels, path_output, annotation_type:Literal['tabledetect', 'tableparse']=None, annotation_format:AnnotationFormat={},
                         split_annotation_types=None, show_labels=True, as_area=False, min_width=800,
                         n_workers=-1, verbosity=logging.INFO):
    # Options | Paths
    path_images = Path(path_images); path_labels = Path(path_labels); path_output = Path(path_output)
    os.makedirs(path_output, exist_ok=True)

    # Options | Logging
    logger = logging.getLogger(__name__); logger.setLevel(verbosity)
    
    # Options | Annotation type
    colors = ['#FF4136', '#2ECC40', '#0074D9', '#FFDC00', '#B10DC9', '#FF851B', 'black']
    if annotation_type == 'tabledetect':
        labelFormat = 'yolo'
        labels = ['table-noborders', 'table-fullborders','table-partialborders']
        classMap = {0: 'table-noborders', 1: 'table-fullborders', 2: 'table-partialborders' }
        split_annotation_types = split_annotation_types or False
    elif annotation_type == 'tableparse':
        labelFormat = 'voc'
        labels = ['table', 'table column','table row', 'table column header', 'table projected row header', 'table spanning cell']
        classMap = None
        split_annotation_types = split_annotation_types or True
    elif not annotation_type:
        try:
            labelFormat = annotation_format['labelFormat']
            labels = annotation_format['labels']
            classMap = annotation_format.get('classMap')
            split_annotation_types = annotation_format.get('split_annotation_types') or False
            
            show_labels = annotation_format.get('show_labels')
            if show_labels is None:
                show_labels = True
            as_area = annotation_format.get('as_area')
            if as_area is None:
                as_area = True
        except KeyError:
            raise KeyError('If -annotation_type- is not specified, please include a -annotation_format- dict.')
    else:
        raise Exception(f'annotation_type {annotation_type} not supported. Currently implemented: tabledetect, tableparse')
    
    colorMap = {key: colors[i] for i, key in enumerate(labels)}         # this will cause an error for large label lists
    print(show_labels)

    # Annotation function
    def annotateImage(imagePath):
        img = Image.open(imagePath).convert('RGBA')
        filename = os.path.splitext(os.path.basename(imagePath))[0]
        imageExtension = os.path.splitext(os.path.basename(imagePath))[-1]
        try:
            annotations = __annotation_to_pilBox(sourcePath=path_labels / filename, labelFormat=labelFormat, targetImage=img, classMap=classMap, colorMap=colorMap)
        except FileNotFoundError:
            annotations = []
        if annotations:
            if split_annotation_types:
                for annotationType in colorMap:        # annotationType = list(colorMap.keys())[1]
                    baseImg = img.copy()
                    relevantAnnotations = list(filter(lambda item: item['label']['text'] == annotationType, annotations))
                    if relevantAnnotations:
                        baseImg = __visualise(image=baseImg, annotations=relevantAnnotations, min_width=min_width, show_labels=show_labels, as_area=as_area)
                        name = f'{filename}_{annotationType.replace(" ", "").upper()}{imageExtension}'
                        baseImg.save(path_output / name)
            else:
                baseImg = __visualise(image=img, annotations=annotations, min_width=min_width, show_labels=show_labels, as_area=as_area)           
                name = f'{filename}_annotated{imageExtension}'
                baseImg.save(path_output / name)
            return True
        else:
            return False
        
    # Annotate
    imagePaths = list(os.scandir(path_images))
    if n_workers == 1:
        start = time.time()
        results = [annotateImage(imagePath=imagePath.path) for imagePath in tqdm(imagePaths, desc='Annotating images')]     # imagePath = imagePaths[0].path           
        logger.info(f'Annotating {len(results)} images took {time.time()-start:.0f} seconds. {results.count(True)} images contained labels.')
    else:
        start = time.time()
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(delayed(annotateImage)(imagePath=imagePath.path) for imagePath in imagePaths)
        logger.info(f'Annotating {len(results)} images took {time.time()-start:.0f} seconds. {results.count(True)} images contained labels.')


if __name__ == '__main__':
    # annotation_type = 'tableparse'
    # annotation_type = 'tabledetect'
    # path_images = rf"F:\ml-parsing-project\table-detect\tabledetect\resources\examples_visualise\{annotation_type}\images"
    # path_labels = rf"F:\ml-parsing-project\table-detect\tabledetect\resources\examples_visualise\{annotation_type}\labels"
    # path_output = rf"F:\ml-parsing-project\table-detect\tabledetect\resources\examples_visualise\{annotation_type}\images_annotated"
    annotation_type = 'tableparse'
    path_images = rf"F:\ml-parsing-project\data\parse_activelearning1_jpg\demos\images_will"
    path_labels = rf"F:\ml-parsing-project\data\parse_activelearning1_jpg\demos\labels_will"
    path_output = rf"F:\ml-parsing-project\data\parse_activelearning1_jpg\demos\images_annotated_will"
    visualise_annotation(annotation_type=annotation_type, path_images=path_images, path_labels=path_labels, path_output=path_output, n_workers=1, as_area=False)

            



