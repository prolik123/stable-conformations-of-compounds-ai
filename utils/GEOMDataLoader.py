import os
import requests
import msgpack
from tqdm import tqdm
from utils.file_untarer import extract_tar_file

folder_path = "../Datasets/GEOM"

crude_url = r"https://dataverse.harvard.edu/api/access/datafile/4327190"
crude_file_name = r"drug_crude.msgpack.tar.gz"
crude_file_name_unpacked = r"qm9_crude.msgpack"

featurized_url = r"https://dataverse.harvard.edu/api/access/datafile/4327191"
featurized_file_name = r"drug_featurized.msgpack.tar.gz"
featurized_file_name_unpacked = r"drug_featurized.msgpack"

def _download_from_url_to_path(url, file_name):
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        print(f"The file {file_name} is already downloaded at {file_path}.")
        return file_path

    print(f"Downloading {file_name} to {file_path}.")
    os.makedirs(folder_path, exist_ok=True)
    response = requests.get(url, stream=True)
    assert response.status_code == 200
    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            ncols=100
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))
    print(f"File downloaded successfully to {file_path}")
    return file_path

def _download_given_part_geom_dataset(download_url, download_file_name, unpacked_file_name):
    downloaded_file_path = _download_from_url_to_path(download_url, download_file_name)
    unpacked_file_path = os.path.join(folder_path, unpacked_file_name)
    extract_tar_file(downloaded_file_path, unpacked_file_path)

def _download_geom_dataset():
    _download_given_part_geom_dataset(crude_url, crude_file_name, crude_file_name_unpacked)
    _download_given_part_geom_dataset(featurized_url, featurized_file_name, featurized_file_name_unpacked)

def _get_unpacker_for_file(unpacked_file_name):
    file = os.path.join(folder_path, unpacked_file_name)
    return msgpack.Unpacker(open(file, "rb"))

def _get_geom_unpackers():
    _download_geom_dataset()
    return _get_unpacker_for_file(crude_file_name_unpacked), _get_unpacker_for_file(featurized_file_name_unpacked)

def load_geom_sample():
    crude_unpacker, featurized_unpacker = _get_geom_unpackers()
    crude_sample = next(iter(crude_unpacker)) # first 1000 entries
    featurized_sample = next(iter(featurized_unpacker)) # first 1000 entries
    return crude_sample, featurized_sample


def load_geom():
    crude_unpacker, featurized_unpacker = _get_geom_unpackers()
    full_crude_data = {}
    for crude_dic in crude_unpacker:
         full_crude_data.update(crude_dic)

    full_features_data = {}
    for features_data in featurized_unpacker:
         full_features_data.update(features_data)

    return full_crude_data, full_features_data


if __name__ == "__main__":
    load_geom_sample()