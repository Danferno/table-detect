import os
def getBoundingBoxesPerFile(yoloAnnotationsFilePath):
    with open(yoloAnnotationsFilePath) as yoloAnnotationsFile:
        filename = os.path.splitext(os.path.basename(yoloAnnotationsFilePath))[0]
        bbox_lists_per_file = {filename: []}
        for line in yoloAnnotationsFile:
            yoloAnnotation = line.strip()
            labelId, x_center, y_center, width, height, conf = map(float, yoloAnnotation.split(' '))
            bbox_lists_per_file[filename].append({
                                'labelId': int(labelId),
                                'x_center': x_center, 'y_center': y_center,
                                'width': width, 'height': height,
                                'confidence': conf})
    return bbox_lists_per_file