from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from modified_knn import KNNModifiedClassifier
from pandas import DataFrame
import pandas as pd
import numpy as np
import warnings
import openml

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
        
        # num_instances = len(targets)
        # if num_instances > 15000:
        #     print(f"Dataset {task_id} too big. Discarted")
        #     continue

        if num_classes == 2:  
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



def run_benchmark(model: any, model_name: str, params=None, metrics: dict = None, tasks: list = None, tasks_description: str = None) -> DataFrame:
    print(f"\nEvaluating metrics for {model_name} model")
    results = pd.DataFrame(columns=["dataset", "model", "neighbours", "weights", "tasks_description"]) 

    if tasks is None or tasks == []:
        return 

    for task_id in tasks:
        print(f"Started task {task_id}")
        task = openml.tasks.get_task(task_id)
        features, targets = task.get_X_and_y()
        
        cv_results = cross_validate(model, features, targets, cv=10, scoring=metrics) 
        scores = {metric: np.mean(cv_results[f'test_{metric}']) if f'test_{metric}' in cv_results.keys() else np.nan for metric in metrics}
        scores.update({"dataset": task_id, "model": model_name, "neighbours": params['kneighborsclassifier__n_neighbors'], "weights": params['kneighborsclassifier__weights'] ,"tasks_description": tasks_description})
        
        results = pd.concat([results, pd.DataFrame(scores, index=[0])], ignore_index=True) 

    print("Finalized evaluation\n")
    return results



def concat_list_of_dataframes(list_of_dataframes: list) -> DataFrame:
    if list_of_dataframes: return pd.concat(list_of_dataframes, ignore_index=True)



def create_csv(dataframe: DataFrame, name: str) -> None:
    dataframe.to_csv(name, index=False)
    print(f"CSV file '{name}' saved successfully.")



def main() -> None:
    tasks = separate_dataset_characteristics()
    print("\nDisbalanced binary tasks: ", tasks['disbalanced_binary_tasks'])
    print("Balanced binary tasks: ", tasks['balanced_binary_tasks'])
    print("Multiclass tasks: ", tasks['multiclass_tasks'])

    models_to_run = {
        'knn': {
            'model': make_pipeline(SimpleImputer(strategy='constant'), StandardScaler(), KNNModifiedClassifier()),
            'params': {
                'knnmodifiedclassifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15],  
                'knnmodifiedclassifier__weights': ['correlations'],  
            },
        },
    }

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'roc_auc_ovr']

    for model_name, model_info in models_to_run.items():
        extract_metrics(model_info['model'], model_name, model_info['params'], metrics, tasks)


def extract_metrics(model: any, model_name: str, params: dict, metrics: list, tasks: dict) -> None:
    try:
        print(f"\nModel Name: {model_name}")
        print(f"Keys in params: {params.keys()}")
        
        all_results = []  

        for param_combination in ParameterGrid(params):
            model_instance = model.set_params(**param_combination)
            print(f"Running benchmark for parameters: {param_combination}")
            
            for task_type, task_list in tasks.items():
                for task in task_list:
                    task_result = run_benchmark(model_instance, model_name, params=param_combination, metrics=metrics, tasks=[task], tasks_description=task_type)
                    all_results.append(task_result)
            

        # Concatenate all results
        concatenated_df = pd.concat(all_results)

        if not concatenated_df.empty:
            create_csv(concatenated_df, f"metrics_{model_name}.csv")
            print(f"Created csv metrics_{model_name}.csv")
        else:
            print(f"No dataframes to concatenate for {model_name}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
