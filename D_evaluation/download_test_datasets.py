import pandas as pd
import openml
import os
import warnings
warnings.filterwarnings("ignore")


def get_dataset(dataSetId: int):
    try:
        dataset = openml.datasets.get_dataset(
            dataSetId, 
            download_data=True, 
            download_qualities=True, 
            download_features_meta_data=True
        )
        raw_data, *_ = dataset.get_data() 
        data = pd.DataFrame(raw_data)  
        return data
    except Exception as e:
        print(e)




tasks = [3, 15, 29, 31, 37, 43, 49, 219, 9910, 9946, 9957, 9976, 10093, 14952, 3902, 3903, 3917, 3918, 9952, 9971, 9977, 9978, 10101]

filepath = './files_csv/csv_tests'
for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = get_dataset(dataset_id)
    filename = f'dataset_{task_id}.csv'
    full_path = os.path.join(filepath, filename)
    dataset.to_csv(full_path, index=False)