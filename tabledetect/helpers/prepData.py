from tqdm import tqdm
import random
import os, sys
import fitz
import time
import logging

TEST = False
if TEST:
    path_in = r"F:\datatog-junkyard\samples_pdfModel_pdfFiles"
    path_output = r"F:\ml-parsing-project\data\detect_activelearning1_jpg\all"

def pdfToImages(path_in, path_output, image_format='.jpg', sample_size=100, dpi=150, keep_transparency=False, n_workers=-1, verbosity=logging.INFO):
    # Options
    # Options | Image format dot
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'

    # Options | Create output folder
    os.makedirs(path_output, exist_ok=True)

    # Logging
    logger = logging.getLogger(__name__); logger.setLevel(verbosity)
    logger.info = print

    # Draw sample
    pdfPaths = random.sample([pdfPath.path for pdfPath in os.scandir(path_in)], k=sample_size)

    # Split
    def splitPdf(pdfPath, path_output):
        try:
            with open(pdfPath) as pdfFile:
                doc = fitz.open(pdfFile)
            filename = os.path.basename(pdfPath).replace('pdf', '')
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
        

    