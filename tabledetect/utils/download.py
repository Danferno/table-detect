import logging
import requests
import zipfile as zf
from io import BytesIO
def downloadRepo(url, destination):
    logging.info(f'ML model repository not found.\nDownloading from: {url}\nDestination: {destination}')
    response = requests.get(url)

    with zf.ZipFile(BytesIO(response.content)) as zipfile:
        zipfile.extractall(destination)
    logging.info(f'Finished downloading ML model repository.')

def downloadWeights(url, destination):
    logging.info(f'ML weights not found.\nDownloading from: {url}\nDestination: {destination}')
    response = requests.get(url)

    with open(destination, 'wb') as out:
        out.write(response.content)
    logging.info(f'Finished downloading ML weights.')