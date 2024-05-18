# from modifications.reference_knn import KNNModifiedClassifier as KNNReferenceClassifier
# from modifications.modified_knn import KNNModifiedClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def k_fold_cross_validation_stratified(X, y, model, n_test: int = 10) -> float:
    skf = StratifiedKFold(n_splits=n_test)
    accuracies = []
    roc_auc = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        roc_auc.append(roc_auc_score(y_test, y_pred))

    return np.mean(accuracies), np.mean(roc_auc)


def process_dataset(dataset_id):
    try:
        df = pd.read_csv(f'cc18-analysis-main/modifications/csv_tests/dataset_{dataset_id}.csv')

        categorical_columns = [column for column in df.columns if df[column].dtype == object]

        for column in categorical_columns:
            dummies = pd.get_dummies(df[column], prefix=column)  
            df = pd.concat([df, dummies], axis=1)  
        df.drop(columns=categorical_columns, inplace=True)

        X = df.iloc[:, :-1]  
        y = df.iloc[:, -1] 

        results_df = pd.DataFrame(columns=['dataset', 'neighbours', 'accuracy_sk', 'accuracy_mod', 'roc_auc_sk', 'roc_auc_mod'])    

        for n in [3,7,11]:
            print(f"Neighbours = {n}")
            sk_knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
            modified_knn = KNNModifiedClassifier(k=n, weight='correlations')
            
            mean_accuracy_sk_knn, mean_roc_auc_sk_knn = k_fold_cross_validation_stratified(X, y, sk_knn, 10)
            mean_accuracy_mod_knn, mean_roc_auc_mod_knn = k_fold_cross_validation_stratified(X, y, modified_knn, 10)

            new_row = pd.DataFrame({'dataset': [dataset_id],
                                'neighbours': [n],
                                'accuracy_sk': [mean_accuracy_sk_knn],
                                'accuracy_mod': [mean_accuracy_mod_knn],
                                'roc_auc_sk': [mean_roc_auc_sk_knn],
                                'roc_auc_mod': [mean_roc_auc_mod_knn]})

            results_df = pd.concat([results_df, new_row], ignore_index=True)

        filename = f'results_{dataset_id}.csv'
        filepath = 'cc18-analysis-main/modifications/csv_results'
        full_path = os.path.join(filepath, filename)
        results_df.to_csv(full_path, index=False)
        print(f"Dataset {dataset_id} processed successfully.")
    except Exception as e:
        print(f"Error processing dataset {dataset_id}: {str(e)}")

dataset_ids = [49, 219, 3021, 3902, 3903, 3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9971, 9976, 9977, 9978, 10093, 10101, 14952]

for dataset_id in dataset_ids:
    # process_dataset(dataset_id)
    csv_results_folder = os.path.join(os.getcwd(), 'csv_results')

    if not os.path.exists(csv_results_folder):
        raise FileNotFoundError(f'The directory {csv_results_folder} does not exist.')

    csv_files = [f for f in os.listdir(csv_results_folder) if f.startswith('result') and f.endswith('.csv')]

    if not csv_files:
        print(f'No CSV files found in {csv_results_folder} for dataset_id {dataset_id}')
        continue

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(csv_results_folder, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_csv_path = os.path.join(csv_results_folder, f'combined_{dataset_id}.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f'All CSV files for dataset_id {dataset_id} have been merged into {combined_csv_path}')
    else:
        print(f'No dataframes to concatenate for dataset_id {dataset_id}')

