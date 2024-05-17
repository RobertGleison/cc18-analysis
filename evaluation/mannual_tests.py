from ..modifications.reference_knn import KNNModifiedClassifier as KNNReferenceClassifier
from ..modifications.modified_knn import KNNModifiedClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def k_fold_cross_validation(X, y, model, n_test: int = 10) -> float:
    kf = KFold(n_splits=n_test)
    accuracies = []
    roc_auc = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        roc_auc.append(roc_auc_score(y_test, y_pred))

    return np.mean(accuracies), np.mean(roc_auc)


def k_fold_cross_validation_y_fix(X, y, model, n_splits: int = 10):
    kf = KFold(n_splits=n_splits)
    accuracies = []
    roc_aucs = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, y_pred))
        else:
            roc_aucs.append(np.nan)  # Append NaN if roc_auc_score cannot be computed

    mean_accuracy = np.mean(accuracies)
    mean_roc_auc = np.nanmean(roc_aucs)  # Compute mean ignoring NaN values
    return mean_accuracy, mean_roc_auc

dataset_id = int(input("Dataset number: "))
df = pd.read_csv(f'csv_tests/dataset_{dataset_id}.csv')
categorical_columns = [column for column in df.columns if df[column].dtype == object]

for column in categorical_columns:
        dummies = pd.get_dummies(df[column], prefix=column)  
        df = pd.concat([df, dummies], axis=1)  
df.drop(columns=categorical_columns, inplace=True)

X = df.iloc[:, :-1]  
y = df.iloc[:, -1] 

results_df = pd.DataFrame(columns=['dataset', 'neighbours', 'accuracy_sk', 'accuracy_ref', 'accuracy_mod', 'roc_auc_sk', 'roc_auc_ref', 'roc_auc_mod'])    

for n in [3,5,7,9,11,13]:
    sk_knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    reference_knn = KNNReferenceClassifier(k=n, weight='distance')
    modified_knn = KNNModifiedClassifier(k=n, weight='correlations')
    
    mean_accuracy_sk_knn, mean_roc_auc_sk_knn = k_fold_cross_validation(X, y, sk_knn, 10)
    mean_accuracy_ref_knn, mean_roc_auc_ref_knn = k_fold_cross_validation(X, y, reference_knn, 10)
    mean_accuracy_mod_knn, mean_roc_auc_mod_knn = k_fold_cross_validation(X, y, modified_knn, 10)

    results_df = results_df.append({
        'dataset': dataset_id,
        'neighbours': n,
        'accuracy_sk': mean_accuracy_sk_knn,
        'accuracy_ref': mean_accuracy_ref_knn,
        'accuracy_mod': mean_accuracy_mod_knn,
        'roc_auc_sk': mean_roc_auc_sk_knn,
        'roc_auc_ref': mean_roc_auc_ref_knn,
        'roc_auc_mod': mean_roc_auc_mod_knn
    }, ignore_index=True)

filename = f'results_{dataset_id}.csv'
filepath = 'csv_results'
full_path = os.path.join(filepath, filename)
results_df.to_csv(full_path, index=False)
