import kagglehub


KAGGLE_HANDLE = "dhruvildave/en-fr-translation-dataset"
LOCAL_DATA_PATH = "data"


def download_data():
    kagglehub.dataset_download(KAGGLE_HANDLE, output_dir=LOCAL_DATA_PATH)


if __name__ == "__main__":
    print("Downloading EN->FR dataset...")
    download_data()
    print("Done.")