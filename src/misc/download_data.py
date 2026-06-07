from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
dataset_name = 'deekshith18/my-something-something-v2-data'
api.dataset_download_files(dataset_name, path='.', unzip=True)
