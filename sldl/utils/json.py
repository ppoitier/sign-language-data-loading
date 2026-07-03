import json


def from_json_to_dict(filepath: str):
    with open(filepath, "r", encoding='utf-8') as f:
        return json.load(f)
