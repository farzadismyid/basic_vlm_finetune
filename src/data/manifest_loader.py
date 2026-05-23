import json
from pathlib import Path


def load_manifest(manifest_path):

    manifest_path = Path(manifest_path)

    with open(manifest_path, "r") as f:

        manifest = json.load(f)

    return manifest
