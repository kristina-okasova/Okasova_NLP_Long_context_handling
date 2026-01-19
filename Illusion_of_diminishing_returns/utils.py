import json
import logging
from argparse import Namespace
from dataclasses import asdict, is_dataclass


def safe_json(obj):
    if isinstance(obj, Namespace):
        return vars(obj)
    if is_dataclass(obj):
        return asdict(obj) # type: ignore
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_json_to_file(data, filepath, indent=4):
    """Saves a dictionary to a JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=safe_json)
        logging.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}", exc_info=True)

