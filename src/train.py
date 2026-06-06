import random
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main(seed):
    set_seed(seed)

    df = pd.read_csv('data/rabies_data.csv')
    X = df.drop(columns=['target']).values
    y = df['target'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=0.25,
        random_state=seed,
        stratify=y
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(
        max_iter=200,
        random_state=seed
    ).fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    acc = accuracy_score(y_te, clf.predict(X_te))

    print(f"seed={seed}  AUC-ROC={auc:.4f}  accuracy={acc:.4f}")
    return clf, scaler, auc, acc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args.seed)