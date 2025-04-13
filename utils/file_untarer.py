import os
import tarfile

def extract_tar_file(file_path, path_to_save):
    if os.path.exists(path_to_save):
        print(f"File already extracted to {path_to_save}")
        return
    if not os.path.exists(file_path):
        raise AssertionError(f"File {file_path} does not exist.")

    try:
        print(f"Extracting {file_path} to {path_to_save}")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(path_to_save))
            print(f"File successfully extracted to {path_to_save}")
    except tarfile.TarError as e:
        print(f"Error while extracting tar file: {e}")
