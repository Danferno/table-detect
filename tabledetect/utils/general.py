import os, shutil
import click
import random
import numpy as np
from tqdm import tqdm

# Parameter parsing
def ensure_image_format_has_dot(image_format):
    if not image_format.startswith('.'):
        image_format = f'.{image_format}'

    return image_format

# Folder creation
def replaceDir(path):
    shutil.rmtree(path)
    os.makedirs(path)

def makeDirs(path, replaceDirs='warn'):
    try:
        os.makedirs(path)
    except FileExistsError:
        if replaceDirs == 'warn':
            if click.confirm(f'This will remove folder {path}. Are you sure you are okay with this?'):
                replaceDir(path)
            else:
                raise InterruptedError(f'User was not okay with removing {path}')
        elif replaceDirs:
            replaceDir(path)
        else:
            raise FileExistsError
        
# Parallel tools
def chunker(lst, n, toList=False, shuffle=False, forceLength=False, **kwargs):
    if shuffle: random.shuffle(lst)
    if len(lst) == 0: raise IndexError('List is empty!')
    if not forceLength:
        n = min(n, len(lst))
    arrayOfArrays = np.array_split(lst, n, **kwargs)
    if toList: 
        listOfLists = [list(subArray) for subArray in arrayOfArrays]
        return listOfLists
    else:
        return arrayOfArrays
    
def daskProgressBar(client, iterable, function, kwargs={}, tqdmkwargs={}) -> list:
    from dask.distributed import as_completed
    futuresList = [client.submit(function, arg, **kwargs) for arg in iterable]
    resultsList = [f.result() for f in tqdm(as_completed(futuresList), total=len(futuresList), **tqdmkwargs)]
    return resultsList