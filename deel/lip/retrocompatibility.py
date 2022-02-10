import json
import os
import shutil

import h5py

from .normalizers import DEFAULT_EPS_BJORCK, DEFAULT_EPS_SPECTRAL

deprecated_args = {
    "niter_spectral": ("eps_spectral", DEFAULT_EPS_SPECTRAL),
    "niter_bjorck": ("eps_bjorck", DEFAULT_EPS_BJORCK),
}


def _replace_key(dictionary, old_key, new_key, new_val):
    """Replace all occurences of a key in a nested dict/list structure.

    The input dictionary can contain nested lists or dictionaries. This function
    replaces the old_key with the new_key and assigns the new_val value to it.

    Args:
        dictionary: a dictionary containing nested lists and dictionaries.
        old_key: the key to remove.
        new_key: the new key to add.
        new_val: the value to assign to the new key.
    """

    if isinstance(dictionary, dict):
        for k in list(dictionary.keys()):
            v = dictionary[k]
            if k == old_key:
                del dictionary[k]
                dictionary[new_key] = new_val
            elif isinstance(v, dict):
                _replace_key(v, old_key, new_key, new_val)
            elif isinstance(v, list):
                for d in v:
                    _replace_key(d, old_key, new_key, new_val)


def upgrade_model(src_filepath, dst_filepath):
    """Upgrade source model by replacing deprecated arguments with the new equivalent
    ones. This is for retrocompatibility between deel-lip versions 1.2.0 and 1.3.0:

    - source model was saved with deel-lip < 1.2.0, but cannot be loaded with 1.3.0.
    - upgraded model can be loaded with deel-lip > 1.3.0.

    Args:
        src_filepath: path to the old h5 model (< 1.2.0).
        dst_filepath: path where to save the new upgraded h5 model.
    """

    if not h5py.is_hdf5(src_filepath):
        raise ValueError(
            "Only Keras .h5 models are supported for retrocompatibility conversion"
        )

    # Copy the source model and load the copied h5 file
    shutil.copy(src_filepath, dst_filepath)
    f = h5py.File(dst_filepath, mode="r+")

    # Get model configuration
    model_config = f.attrs.get("model_config")
    if model_config is None:
        raise ValueError("No model found in config file.")
    json_model = json.loads(model_config)

    # Check if deprecated arguments are present in model configuration
    found_deprecated_arg = False
    for deprecated_arg in deprecated_args.keys():
        if f'"{deprecated_arg}"' in model_config:
            print(f'Found "{deprecated_arg}" in model configuration.')
            found_deprecated_arg = True

    if found_deprecated_arg:
        # Replace deprecated arguments with their new equivalent
        for old_arg, new_arg in deprecated_args.items():
            _replace_key(json_model, old_arg, new_arg[0], new_arg[1])

        # Update model configuration in the h5py.File
        f.attrs["model_config"] = json.dumps(json_model).encode("utf8")
        f.close()

        print(f"Model upgraded and saved to {dst_filepath}")
    else:
        # If no upgrade required, delete the dst_filepath
        os.remove(dst_filepath)
        print("No deprecated arguments were found. No operation is done")
