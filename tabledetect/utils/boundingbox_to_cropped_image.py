import os
from PIL import Image
import json

def yoloBboxToPillowBbox(yoloBbox, image_width, image_height):
    # Convert YOLO coordinates to pixel coordinates
    x_center, y_center, width, height = yoloBbox['x_center'], yoloBbox['y_center'], yoloBbox['width'], yoloBbox['height']
    left = int((x_center - width / 2) * image_width)
    top = int((y_center - height / 2) * image_height)
    right = int(left + width * image_width)
    bottom = int(top + height * image_height)
    
    return (left, top, right, bottom)
    
def extractCroppedImages(bbox_lists_per_file_list, outDir, imageFormat, imageDir):
    # Create folders
    uniqueLabelIds = set(bbox['labelId'] for item in bbox_lists_per_file_list for _, bbox_list in item.items() for bbox in bbox_list)
    os.makedirs(os.path.join(outDir), exist_ok=True)
    [os.makedirs(os.path.join(outDir, str(labelId)), exist_ok=True) for labelId in uniqueLabelIds]

    # Loop over source images
    boundingBoxInfos = []
    for bbox_lists_per_file in bbox_lists_per_file_list: 
        for filename, bbox_list in bbox_lists_per_file.items():
            imageName = f'{filename}{imageFormat}'
            imagePath = os.path.join(imageDir, imageName)
            inputImage = Image.open(imagePath)

            # Loop over object in source image
            for i, bbox in enumerate(bbox_list):
                pathOut = os.path.join(outDir, str(bbox['labelId']), f'{filename}_{i}{imageFormat}')
                
                pillowBbox = yoloBboxToPillowBbox(yoloBbox=bbox, image_width=inputImage.width, image_height=inputImage.height)
                cropped = inputImage.crop(pillowBbox)
                cropped.save(pathOut)

                boundingBoxInfos.append({'sourceName': filename, 'labelId': bbox['labelId'], 'table_num': i, 'pillowBbox': pillowBbox})

    pathBoundingBoxFile = os.path.join(outDir, 'boundingBoxData.json')
    with open(pathBoundingBoxFile, 'w') as boundingBoxFile:
        json.dump(boundingBoxInfos, boundingBoxFile)