import pathlib

def path_exists(path):
    return pathlib.Path(path).exists()
