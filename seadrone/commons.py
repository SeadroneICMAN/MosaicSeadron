import inspect

from typing import Callable, Mapping, Any, List
from pathlib import Path


def get_default_params(func : Callable) -> Mapping[str, Any]:
    """This function returns the default parameters of a given function

    Args:
        func (Callable): function to read its parameters

    Returns:
        Mapping[str, Any]: default parameters
    """

    signature = inspect.signature(func)
    params = signature.parameters

    defaults = {}
    for param in params.values():
        if param.default != inspect.Parameter.empty:
            defaults[param.name] = param.default

    return defaults


def overwrite_dict_a_with_dict_b(a : dict, b : dict) -> dict:
    """Given to dictionaries of different size, this function returns the first dict with the values of the second dict 
    for those keys that are in the two dicts.

    Args:
        a (dict): first dict
        b (dict): second dict

    Returns:
        dict: first dict overwritten with values of the second dict
    """

    for key in set(a.keys()).intersection(set(b.keys())):
        a[key] = b[key]
        
    return a


def search_raster_paths(in_folder : str, raster_paths : List[str] = []) -> List[str]:
    """_summary_

    Args:
        in_folder (str): Root folder to search
        raster_paths (List[str], optional): A list of full paths that are .tif files. Defaults to [].

    Returns:
        List[str]: A list of full paths that are .tif files
    """

    folder = Path(in_folder)

    for i in folder.iterdir():
        if 'tif' in i.suffix and not str(i) in raster_paths:
            raster_paths.append(str(i))
        elif i.is_dir():
            search_raster_paths(str(i))
    
    return raster_paths


def get_first_file_extension(folder_path : str) -> str:
        """this function return the extension of the first file from a folder

        Args:
            folder_path (str): _description_

        Returns:
            str: _description_
        """

        folder = Path(folder_path)
        
        # Get a list of all files in the folder
        files = list(folder.glob('*'))
        
        if not files:
            return None  # No files in the folder
        
        # Get the first file's extension
        first_file = files[0]
        file_extension = first_file.suffix
        
        return file_extension