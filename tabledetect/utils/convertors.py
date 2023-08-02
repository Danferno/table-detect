# Import
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from lxml import etree
import general

# Constants
TQDM_OPTIONS_FILES = dict(smoothing=0.1, desc='Looping over files')
IMPLEMENTED = ['voc']

# Existing to codamo
def yolo_to_codamo(path_label, path_image, classMap):
    ''' Convert yolo annotation at :path_label: for image at :path_image: to
    a codamo-style annotation dictionary using :classMap: to convert the 
    integer classes of yolo to a meaningful label.
    '''

    # Convert to universal
    image_width, image_height = Image.open(path_image).size     
    bboxes = []
    with open(path_label, 'r') as f_in:
        for line in f_in:
            # Read line
            try:
                labelClass, xcenter, ycenter, width, height, conf = map(float, line.strip().split(' '))
            except ValueError:
                labelClass, xcenter, ycenter, width, height = map(float, line.strip().split(' '))
                conf = None
        
            bboxes.append({'labelClass': classMap[int(labelClass)],
                           'x0': round((xcenter - width / 2)*image_width),
                           'x1': round((xcenter + width / 2)*image_width),
                           'y0': round((ycenter - height / 2)*image_height),
                           'y1': round((ycenter + height / 2)*image_height),
                           'conf': conf})
            
    # Annotation
    annotation_codamo = {
        'image_width': image_width,
        'image_height': image_height,
        'bboxes': bboxes
    }

    return annotation_codamo

# Codamo to existing
def codamo_to_voc(annotation_codamo, path_output):
    ''' Convert codamo-style annotation dictionary to voc-style xml annotation.
    '''
    # Write XML
    xml = etree.Element('annotation')
    xml_size = etree.SubElement(xml, 'size')
    xml_width  = etree.SubElement(xml_size, 'width');   xml_width.text = str(annotation_codamo['image_width'])
    xml_height = etree.SubElement(xml_size, 'height'); xml_height.text = str(annotation_codamo['image_height'])
   
    # Add table bbox
    for bbox in annotation_codamo['bboxes']:        # temp = iter(annotation_codamo['bboxes']); bbox = next(temp)
        xml_object = etree.SubElement(xml, 'object')
        xml_label = etree.SubElement(xml_object, 'name'); xml_label.text = bbox['labelClass']
        xml_table_bbox = etree.SubElement(xml_object, 'bndbox')
        x0 = etree.SubElement(xml_table_bbox, 'xmin'); x0.text = str(bbox['x0'])
        x1 = etree.SubElement(xml_table_bbox, 'xmax'); x1.text = str(bbox['x1'])
        y0 = etree.SubElement(xml_table_bbox, 'ymin'); y0.text = str(bbox['y0'])
        y1 = etree.SubElement(xml_table_bbox, 'ymax'); y1.text = str(bbox['y1'])

    # Save
    tree = etree.ElementTree(xml)
    tree.write(path_output, pretty_print=True, xml_declaration=False, encoding='utf-8')


# Code
def yolo_to_x(output_format, path_labels, path_images, classMap, path_output,
              image_format='.png',
              replace_dirs='warn', n_workers=1):
    # Parameters
    path_labels = Path(path_labels)
    path_images = Path(path_images)
    path_output =  Path(path_output)
    general.makeDirs(path_output, replaceDirs=replace_dirs)

    # Function
    def parse_single(labelFile):
        filename = os.path.splitext(os.path.basename(labelFile))[0]
        path_image = path_images / f'{filename}{image_format}'
        annotation_codamo = yolo_to_codamo(path_label=labelFile, path_image=path_image, classMap=classMap)

        if output_format == 'voc':
            codamo_to_voc(annotation_codamo=annotation_codamo, path_output=path_output / f'{filename}.xml')
        else:
            raise ValueError(f'{output_format} not implemented. Only {IMPLEMENTED} are currently supported as output_format options.')
    def parse_list(labelFileList):
        return [parse_single(labelFile) for labelFile in labelFileList]

    # Get labels
    labelFiles = [entry.path for entry in os.scandir(path_labels)]
    if n_workers == 1:
        for labelFile in tqdm(labelFiles, **TQDM_OPTIONS_FILES):        # temp = iter(labelFiles); labelFile = next(temp)
            parse_single(labelFile)
    else:
        # Start parallel client
        print('Starting dask client for parallel processing')
        import psutil
        from dask.distributed import Client, LocalCluster, Worker
        n_cores = psutil.cpu_count()
        workers_to_use = n_workers if n_workers > 0 else n_cores + n_workers
        client = Client(LocalCluster(n_workers=workers_to_use, threads_per_worker=1, worker_class=Worker))

        # Starting jobs
        labelFileLists = general.chunker(labelFiles, n=min(workers_to_use*3, len(labelFiles)), toList=True, shuffle=True)
        _ = general.daskProgressBar(client=client, iterable=labelFileLists, function=parse_list, tqdmkwargs=TQDM_OPTIONS_FILES)

# Test
if __name__ == '__main__':
    path_labels_yolo = r"F:\ml-parsing-project\data\detect_activelearning1_png\selected_labels"
    classMap_yolo = {0: 'table'}
    path_images_yolo = r"F:\ml-parsing-project\data\detect_activelearning1_png\selected"
    path_out_yoloToVoc = r"F:\ml-parsing-project\data\detect_activelearning1_png\selected_labels_voc"
    n_workers = -2
    
    yolo_to_x(output_format='voc', path_labels=path_labels_yolo, path_images=path_images_yolo, classMap=classMap_yolo, path_output=path_out_yoloToVoc, n_workers=n_workers)