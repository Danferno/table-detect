import os, shutil
import click

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