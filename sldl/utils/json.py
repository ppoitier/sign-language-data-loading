import json


def from_json_to_dict(filepath: str):
    """Load a JSON file into a Python object.

    Args:
        filepath: Path to the JSON file.

    Returns:
        The parsed JSON content.
    """
    with open(filepath, "r", encoding='utf-8') as f:
        return json.load(f)
