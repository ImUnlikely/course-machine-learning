def get_dataset_from_online(url, destination, file_name):
    """Downloads raw github files to given directory relative to this notebook

    Args:
        url (str): the url of the raw data
        destination (str): [description]
        file_name ([type]): [description]
    """
    if not os.path.exists(destination):
        print("Creating directory")
        os.mkdir(destination)
    else:
        print("Directory already exists")

    file_path = os.path.join(destination, file_name)
    if not os.path.exists(file_path):
        print("Downloading file")

        req = requests.get(file_url, stream=True)
        if req.ok:
            print("Saving to given destination")
            with open(file_path, "wb") as f:
                for chunk in req.iter_content(chunk_size=1024*8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
        else:
            print(f"Download failed: status code {req.status_code}\n{req.text}")
    else:
        print("File already exists, cancelling")