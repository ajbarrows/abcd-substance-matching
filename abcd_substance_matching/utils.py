import pathlib

import yaml


def load_yaml(fpath: str) -> dict:
    """Load YAML configuration file.

    Args:
        fpath: Path to the YAML file.

    Returns:
        Dictionary containing the parsed YAML content.
    """
    fpath = pathlib.PurePath(fpath)
    with open(fpath, 'r') as file:
        conf = yaml.safe_load(file)

        for key, value in conf.items():
            if isinstance(value, str):
                conf[key] = value.encode().decode('unicode_escape')

    return conf
