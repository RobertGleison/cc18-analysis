'''
Deprecated:
 
This file is no more utilized. 
The members of the project decided to move on the analysis only with KNN models. To extract data and informations of the datasets, we now utilize the 'knn_exploration.ipynb' 
or 'knn_exploration.py', which provide better logs, more informations about knn and a simple code.

'''



import openml
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import logging
import warnings

# logging.basicConfig(filename='benchmark.log', level=logging.INFO)
warnings.filterwarnings("ignore")



def separate_dataset_characteristics(benchmark: str ="OpenML-CC18", disbalance_threshold: float = 0.3) -> dict:
    benchmark_suite = openml.study.get_suite(benchmark)
    subset_benchmark_suite = benchmark_suite.tasks[0:50]
    disbalanced_binary_tasks = []
    balanced_binary_tasks = []
    multiclass_tasks = []

    for task_id in subset_benchmark_suite:
        task = openml.tasks.get_task(task_id)
        _, targets = task.get_X_and_y()
        num_classes = len(np.unique(targets))
        
        num_instances = len(targets)
        if num_instances > 15000:
            print(f"Dataset {task_id} too big. Discarted")
            continue

        if num_classes == 2:  # Binary classification task
            minority_fraction = pd.Series(targets).value_counts(normalize=True).min()
            if minority_fraction < disbalance_threshold:  disbalanced_binary_tasks.append(task_id)
            else: balanced_binary_tasks.append(task_id)
            continue

        multiclass_tasks.append(task_id)

    return {
        "disbalanced_binary_tasks": disbalanced_binary_tasks,
        "balanced_binary_tasks": balanced_binary_tasks,
        "multiclass_tasks": multiclass_tasks
    }



def filter_columns(results: DataFrame) -> DataFrame:
    keep_columns = ['Model', 'Dataset', 'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall',
                    'mean_test_f1', 'mean_test_roc_auc', 'mean_test_roc_auc_ovr' ,'Data_type']
    keep_columns += [col for col in results.columns if col.startswith('param_')]
    results = results[keep_columns]
    return results



def run_benchmark(model: any, model_name: str, params: dict = None, metrics: list = None, tasks: list = None, tasks_description: str = None) -> DataFrame:
    print(f"\nEvaluating metric for {model_name} model")
    results_list = []

    if tasks is None or tasks == []: return 

    for task_id in tasks:
        print(f"Started task {task_id}")
        task = openml.tasks.get_task(task_id)
        features, targets = task.get_X_and_y()
        
        grid_search = GridSearchCV(model, params, cv=10, scoring=metrics, refit=False, n_jobs=-1)
        grid_search.fit(features, targets)

        results = pd.DataFrame(grid_search.cv_results_)
        results['Dataset'] = task_id
        results['Data_type'] = tasks_description
        results['Model'] = model_name
        results_list.append(results)

    all_results = pd.concat(results_list, ignore_index=True)
    print("Finalized evaluation\n")
    # return all_results
    return filter_columns(all_results)



def concat_list_of_dataframes(list_of_dataframes: list) -> DataFrame:
    if list_of_dataframes: return pd.concat(list_of_dataframes, ignore_index=True)



def create_csv(dataframe, name) -> None:
    # path = os.path.join(os.getcwd(), '../csv_files/')
    # if not dataframe.empty:
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     full_path = os.path.join(path, name)
        dataframe.to_csv(name, index=False)
        print(f"CSV file '{name}' saved successfully.")
    # else:
    #     print("No dataframe provided.")


def main() -> None:
    tasks = separate_dataset_characteristics()
    print("Disbalanced binary tasks: ", tasks['disbalanced_binary_tasks'])
    print("Balanced binary tasks: ", tasks['balanced_binary_tasks'])
    print("Multiclass tasks: ", tasks['multiclass_tasks'])

    dt = make_pipeline(SimpleImputer(strategy='constant'),DecisionTreeClassifier()) # Decision trees are not sensible to non scaling values
    knn = make_pipeline(SimpleImputer(strategy='constant'),StandardScaler(),KNeighborsClassifier())
    svm = make_pipeline(SimpleImputer(strategy='constant'),StandardScaler(),SVC(probability=True))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'roc_auc_ovr']
    models = [knn]
    model_names = ['knn']
    # models = [dt, knn, svm]
    # model_names = ['dt', 'knn', 'svm']

    params = {
        'dt': {
            'decisiontreeclassifier__criterion': ['gini', 'entropy'],  
            'decisiontreeclassifier__max_depth': [5, 7, 9],  
            'decisiontreeclassifier__min_samples_split': [3, 4, 5],  
            'decisiontreeclassifier__min_samples_leaf': [2, 3, 4]
        },
        'knn': {
            'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11],  
            'kneighborsclassifier__weights': ['uniform', 'distance'],  
        },
        'svm': {
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
            'svc__gamma': ['scale', 'auto']  
        }
    }
    extract_metrics(models, model_names, params, metrics, tasks)



def extract_metrics(models, model_names, params, metrics, tasks):
    try:
        for model, model_name in zip(models, model_names):
            metric_results={}
            print(f"\nModel Name: {model_name}")
            print(f"Keys in params: {params.keys()}")
            results = []

            results.append(run_benchmark(model, model_name, params=params[model_name], metrics=metrics, tasks=tasks['disbalanced_binary_tasks'], tasks_description='disbalanced_binary_tasks'))
            results.append(run_benchmark(model, model_name, params=params[model_name], metrics=metrics, tasks=tasks['balanced_binary_tasks'], tasks_description='balanced_binary_tasks'))
            results.append(run_benchmark(model, model_name, params=params[model_name], metrics=metrics, tasks=tasks['multiclass_tasks'], tasks_description='multiclass_tasks'))
            metric_results[model_name] = results

            concatenated_df = concat_list_of_dataframes(metric_results[model_name])
            if not concatenated_df.empty:
                create_csv(concatenated_df, f"metrics_{model_name}.csv")
                print(f"Created csv metrics_{model_name}.csv")
            else:
                print(f"No dataframes to concatenate for {model_name}.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
