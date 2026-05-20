import json


def load_manifest(manifest_path):

    with open(manifest_path, "r") as f:

        manifest = json.load(f)

    return manifest
