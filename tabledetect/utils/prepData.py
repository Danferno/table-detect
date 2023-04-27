from tqdm import tqdm
import random
import os, sys
import fitz
import time
import logging
import numpy as np
from pathlib import Path
import json
import math
import shutil

TEST = False
if TEST:
    path_input = r"F:\datatog-junkyard\samples_pdfModel_pdfFiles"
    path_output = r"F:\ml-parsing-project\data\detect_activelearning1_jpg\all"

    path_labels = Path(r"F:\ml-parsing-project\data\detect_activelearning1_jpg\detected\out\table-detect/labels")
    path_images = Path(r"F:\ml-parsing-project\data\detect_activelearning1_jpg\all")
    path_output = r"F:\ml-parsing-project\data\detect_activelearning1_jpg\selected"

def pdfToImages(path_input, path_output, image_format='.jpg', sample_size_pdfs=100, dpi=150, keep_transparency=False, n_workers=-1, verbosity=logging.INFO):
    # Options
    # Options | Image format dot
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'

    # Options | Create output folder
    os.makedirs(path_output, exist_ok=True)

    # Logging
    logger = logging.getLogger(__name__); logger.setLevel(verbosity)

    # Draw sample
    pdfPaths = random.sample([pdfPath.path for pdfPath in os.scandir(path_input)], k=sample_size_pdfs)

    # Split
    def splitPdf(pdfPath, path_output):
        try:
            with open(pdfPath) as pdfFile:
                doc = fitz.open(pdfFile)
            filename = os.path.basename(pdfPath).replace('.pdf', '')
            for page in doc:
                page.get_pixmap(alpha=keep_transparency, dpi=dpi).save(os.path.join(path_output, f"{filename}-p{page.number}{image_format}"))
            return True
        except fitz.fitz.FileDataError:
            return False
        
    if n_workers == 1:
        start = time.time()
        results = [splitPdf(pdfPath=pdfPath, path_output=path_output) for pdfPath in tqdm(pdfPaths, desc='Splitting PDF files')]
        logger.info(f'Splitting {len(results)} PDFs took {time.time()-start:.0f} seconds. {results.count(False)} PDFs raised an error.')

    else:
        from joblib import Parallel, delayed
        start = time.time()
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=verbosity)(delayed(splitPdf)(pdfPath, path_output) for pdfPath in pdfPaths)
        logger.info(f'Splitting {len(results)} PDFs took {time.time()-start:.0f} seconds. {results.count(False)} PDFs raised an error.')

def sampleImages(path_labels, path_images, path_output, percentile_of_scores=15, sample_size=6000, inverse_scaling_factor=5, share_sample_empty=0.05, verbosity=logging.INFO):
    '''
    percentile_of_scores: Percentile of confidence scores to take, rounded down.
        Some models output multiple confidence scores per image, this option determines
        which score to use. Set to 0 to use the lowest. By default we take the 15th percentile,
        this excludes the very lowest confidence scores while still basing the sampling probability
        on a relatively bad prediction within the image.
    '''
    # Options | Paths
    path_images = Path(path_images)
    path_output = Path(path_output)

    # Options | Sampling probability function
    def confidenceScore_to_samplingProbability(confidenceScore):
        return 0.01 + (1 - 0.01) * math.exp(-inverse_scaling_factor*confidenceScore)
    
    # Options | Logger
    logger = logging.getLogger(__name__); logger.setLevel(verbosity)
    

    # Sample | notEmpty | Calculate sampling probability per file based on confidence score
    labelFilePaths = list(os.scandir(path_labels))
    notEmpty_samplingProbability = {}
    for labelFilePath in tqdm(labelFilePaths, desc='Calculating sampling probabilities'):                                # labelFilePath = next(labelFilePaths)
        try:
            with open(labelFilePath.path, 'r') as labelFile:    
                confidenceScores = [float(line.split(' ')[-1].strip('\n')) for line in labelFile]
                relevantConfidenceScore = np.percentile(confidenceScores, q=percentile_of_scores, method='inverted_cdf')

            notEmpty_samplingProbability[labelFilePath.name] = confidenceScore_to_samplingProbability(relevantConfidenceScore)
        except UnicodeDecodeError:
            pass
    
    # Sample | notEmpty | Collect sample
    sample_size_notEmpty = math.ceil(sample_size * (1-share_sample_empty))
    notEmpty_pool = [os.path.splitext(filename)[0] for filename in notEmpty_samplingProbability.keys()]
    notEmpty_poolsize = len(notEmpty_pool)
    if sample_size_notEmpty > notEmpty_poolsize:
        sample_notEmpty = notEmpty_pool
        sample_size_initial = sample_size
        sample_size = math.ceil(notEmpty_poolsize / (1-share_sample_empty))
        sample_size_notEmpty = notEmpty_poolsize
        logger.warning(f'''Sample size of {sample_size_initial} is too large for the number of images with predictions ({notEmpty_poolsize})
                       given the share allocated to non-empty predictions (1-{share_sample_empty:.0%}={1-share_sample_empty:.0%}).
                       You will either want to
                            a) reduce the sample size, e.g. to 10% of non-empty images ({notEmpty_poolsize/10:.0f})
                            b) supply more images
                        For now, I will reduce the sample size proportionally such that all non-empty images are kept.
                        New sample size target: {sample_size}''')
    else:    
        notEmpty_probs = np.asarray(list(notEmpty_samplingProbability.values()))
        notEmpty_probs = np.divide(notEmpty_probs, np.sum(notEmpty_probs)) 

        sample_notEmpty = np.random.choice(a=notEmpty_pool, size=sample_size_notEmpty, replace=False, p=notEmpty_probs)

    
    # Sample | Empty | Collect sample
    if share_sample_empty > 0:
        empty_pool = list(set([os.path.splitext(dirEntry.name)[0] for dirEntry in os.scandir(path_images) if dirEntry.is_file()]) - set(notEmpty_pool))
        empty_poolsize = len(empty_pool)
        sample_size_empty = sample_size - sample_size_notEmpty

        if sample_size_empty > empty_poolsize:
            sample_empty = empty_pool
            sample_size_empty_initial = sample_size_empty
            sample_size_empty = empty_poolsize
            logger.warning(f'''Empty-image sample size of {sample_size_empty_initial} is too large for the number of images without predictions ({empty_poolsize})
                       given the share of the total sample allocated to empty predictions ({share_sample_empty:.0%}).
                       You will either want to
                            a) reduce the sample size
                            b) supply more images
                            c) reduce the share allocated to empty images
                        For now, I will reduce the empty-image sample size proportionally such that all empty images are kept.
                        New empty-image sample size target: {sample_size_empty}''')

        else:
            sample_empty = np.random.choice(a=empty_pool, size=sample_size_empty, replace=False)
    else:
        sample_empty = set()

    # Sample | Combined
    sample = list(sample_notEmpty) + list(sample_empty)

    # Sample | Collect images
    imageFormat = os.path.splitext(next(os.scandir(path_images)))[-1]
    os.makedirs(path_output, exist_ok=True)
    for filename in tqdm(sample, desc='Copying files from path_images to path_output'):
        filename_full = f'{filename}{imageFormat}'
        _ = shutil.copyfile(src=path_images / filename_full,
                        dst=path_output / filename_full)


    






        