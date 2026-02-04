import os
import yaml
import urllib.request


def load_test_config(config_path="examples/register_config_test.yaml"):
    assert os.path.exists(config_path)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Unable to load test config file due to: {e}')
        raise e

    return config


def download_file(path, url):
    if os.path.exists(path):
        print(f"✅ File already exists at {path}")
        return

    print(f"Downloading file from {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"✅ Download complete: {path}")
    except Exception as e:
        print(f"❌ Failed to download file: {e}")
        print(f"Please manually download the file from:\n{url}")
        print(f"and save it to:\n{path}")
