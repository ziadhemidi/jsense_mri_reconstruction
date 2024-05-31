import copy
import os
import re
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F

THIS_SCRIPT_DIR = str(Path(__file__).parent)


class DotDict(dict):
    """dot.notation access to dictionary attributes
    See https://stackoverflow.com/questions/49901590/python-using-copy-deepcopy-on-dotdict
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_notebook():
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return False


def extract_accleration_factor(string, word: str = "AccFactor"):
    '''Extracts the acceleration factor from a string.
    Args:
        string (str): String to extract the acceleration factor from.
        word (str): Word to search for in the string.
    Returns:
        acceleration_factor (str): Acceleration factor.
    '''
    
    pattern = r"\b" + re.escape(word) + r"\D*(\d{2})"
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None

def extract_patient_id(filename):
    '''
    Extracts the patient ID from a filename.
    Args:
        filename (str): path to extract the patient ID from.
        Returns:
        patient_id (str): Patient ID.
    '''
    return re.search(r"(?<=P)\d+", filename).group(0)

def create_directory_structure(root_directory, coil, dataset, acceleration):
    """Creates a dictionary with the directory structure of the dataset.
    Args:
        root_directory (str): Root path of the dataset.
        coil (str): Coil configuration. One of 'SingleCoil', 'MultiCoil'.
        dataset (str): Dataset. One of 'TrainingSet', 'ValidationSet'.
        acceleration (str): Acceleration factor. One of 'AccFactor08', 'AccFactor10', 'AccFactor04'.
    Returns:
        directory_structure (dict): Dictionary with the directory structure of the dataset.
    """

    directory_structure = {}

    for root, directories, files in os.walk(root_directory):
        current_directory = directory_structure
        path = os.path.relpath(root, root_directory).split(
            os.sep
        )  # Get relative path from root_directory

        for directory in path:
            if directory != ".":
                current_directory = current_directory.setdefault(
                    os.path.basename(directory), {}
                )

        for file in files:
            filename = os.path.splitext(file)[0]  # Remove file extension
            current_directory[filename] = os.path.join(root, file)
            
    directory_structure = directory_structure[coil]["Cine"][dataset]
    if dataset == "TrainingSet":
        if acceleration != "FullSample":
            directory_structure_1 = directory_structure[acceleration]
            directory_structure_2 = directory_structure["FullSample"]
            directory_structure = {
                acceleration: directory_structure_1,
                "FullSample": directory_structure_2,
            }
        else:
            directory_structure = directory_structure[acceleration]

    elif dataset == "ValidationSet":
        if acceleration != "FullSample":
            directory_structure = directory_structure[acceleration]
        else:
            ValueError("ValidationSet does not have Fully Sampled Data")

    return directory_structure


def extract_file_paths(
    root_path, coil_info, dataset, acceleration_factor, cardiac_view="cine_sax"
):
    """Extracts the file paths for the given dataset and coil configuration.
    Args:
        root_path (str): Root path of the dataset.
        coil_info (str): Coil configuration. One of 'SingleCoil', 'MultiCoil'.
        dataset (str): Dataset. One of 'TrainingSet', 'ValidationSet'.
        acceleration_factor (str): Acceleration factor. One of 'AccFactor08', 'AccFactor10', 'AccFactor12'.
        cardiac_view (str): Cardiac view. One of 'cine_sax', 'cine_lax'.
    Returns:
        paths (list): List of file paths of the corresponding undersampeled data.
        path_full (list): List of file paths for the corresponding fully sampled data.
        path_mask (list): List of file paths for the corresponding sampling masks.
    """
    directory_dict = create_directory_structure(
        root_path, coil_info, dataset, acceleration_factor
    )
    paths = []
    path_mask = []
    path_full = []

    def traverse_directory(directory_dict, current_path):
        for key, value in directory_dict.items():
            if isinstance(value, dict):
                # It's a nested directory
                traverse_directory(value, current_path + "/" + key)
            else:
                # It's a file
                if cardiac_view in value and "mask" not in value:
                    if "FullSample" not in value:
                        paths.append(value)
                    else:
                        path_full.append(value)

                elif cardiac_view + "_mask" in value:
                    path_mask.append(value)
                else:
                    pass

    traverse_directory(directory_dict, "")
    return [paths, path_full, path_mask]


def merge_config_dicts(sweep_config_dict, config_dict):
    """Merges the sweep_config_dict and the config_dict.
    Args:
        sweep_config_dict (dict): Dictionary with the sweep configuration.
        config_dict (dict): Dictionary with the configuration.
    Returns:
        merged_sweep_config_dict (dict): Dictionary with the merged configuration.
    """
    cp_config_dict = copy.deepcopy(config_dict)

    for del_key in sweep_config_dict["parameters"].keys():
        if del_key in cp_config_dict:
            del cp_config_dict[del_key]
    merged_sweep_config_dict = copy.deepcopy(sweep_config_dict)

    for key, value in cp_config_dict.items():
        merged_sweep_config_dict["parameters"][key] = dict(value=value)
    # Convert enum values in parameters to string. They will be identified by their numerical index otherwise
    for key, param_dict in merged_sweep_config_dict["parameters"].items():
        if "value" in param_dict and isinstance(param_dict["value"], Enum):
            param_dict["value"] = str(param_dict["value"])
        if "values" in param_dict:
            param_dict["values"] = [
                str(elem) if isinstance(elem, Enum) else elem
                for elem in param_dict["values"]
            ]

        merged_sweep_config_dict["parameters"][key] = param_dict
    return merged_sweep_config_dict

def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, "r") as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                new_value = v[()][:]
                if np.array(new_value).dtype == np.float64:
                    data[k] = new_value
                else:
                    data[k] = new_value["real"] + 1j * new_value["imag"]

            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data


def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            new_value = v[()][:]
            if new_value.dtype == np.float64:
                data[k] = new_value
            else:
                data[k] = new_value["real"] + 1j * new_value["imag"]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data

def get_script_dir():
    if in_notebook():
        return os.path.abspath('')
    else:
        return os.path.dirname(os.path.realpath(__file__))

def set_env_vars():
    """Sets the environment variables for the CMRxRecon dataset."""
    os.environ["CMRXRECON_CACHE_PATH"] = str(
        Path(THIS_SCRIPT_DIR, "../../.cache").resolve()
    )


if __name__ == "__main__":
    set_env_vars()
