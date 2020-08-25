def get_dataset_from_online(url, destination, file_name):
    """Downloads raw github files to given directory relative to this notebook

    Args:
        url (str): the url of the raw data
        destination (str): file path relative to the notebook
        file_name (str): the name of the file you want to create (name.csv)
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

def load_csv_to_dataframe(file_path):
    """Loads a given csv file to dataframe

    Args:
        file_path (str): path to the csv file

    Returns:
        obj: a dataframe containing data from given csv file
    """
    if not os.path.exists(file_path):
        print("File does not exist")
    else:
        dataframe = pd.read_csv(file_path)
        print("Dataframe loaded")
        print(dataframe.info())
        
        return dataframe

