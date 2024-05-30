from typing import Any


def deepin_update(dict1: dict[Any, Any], dict2: dict[Any, Any]):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for key in dict2:
        if (
            key in dict1
            and isinstance(dict1[key], dict)
            and isinstance(dict2[key], dict)
        ):
            dict1[key] = deepin_update(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]

    return dict1
