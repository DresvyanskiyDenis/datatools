import os
import sys
from typing import Dict, Union


def load_config_file(path_to_config_file) -> Dict[str, Union[str, int, float, bool]]:
    sys.path.insert(1, str(os.path.sep).join(path_to_config_file.split(os.path.sep)[:-1]))
    name = os.path.basename(path_to_config_file).split(".")[0]
    config = __import__(name)
    # convert it to dict
    config = vars(config)
    # exclude all system varibles
    config = {key: value for key, value in config.items() if not key.startswith("__")}
    return config