import random
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def sensitivity_at_specificity(y_true, y_scores, target_specificity=0.90):
    """
    Returns the highest sensitivity (recall on the positive class) achievable
    while keeping specificity >= target_specificity. Useful for rare-event
    problems (like wild rabies) where AUC-ROC alone can look optimistic.
    """
    thresholds = np.unique(y_scores)
    best_sensitivity = 0.0
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        if specificity >= target_specificity:
            sensitivity = recall_score(y_true, y_pred)
            best_sensitivity = max(best_sensitivity, sensitivity)
    return best_sensitivity


def run_model(model_name, X, y, groups, seed, n_splits):
    """Runs spatial cross-validation for one model and returns per-fold + summary metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_aucs, fold_accs, fold_praucs, fold_sens = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Rule 1 (kept from original script): split before scaling.
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if model_name == 'logreg':
            # Deterministic convex solver: C and solver fixed explicitly so
            # behavior does not depend on the scikit-learn version or
            # platform (Pineau et al., 2021). NOTE: with 'lbfgs' (or 'sag'),
            # random_state has no effect on the fitted coefficients, since
            # logistic regression has a single global optimum. Results are
            # therefore IDENTICAL across seeds for this model -- by
            # mathematical construction, not a bug.
            clf = LogisticRegression(
                max_iter=1000, C=1.0, solver='lbfgs', random_state=seed
            ).fit(X_tr_s, y_tr)

        elif model_name == 'random_forest':
            # Genuinely stochastic: bootstrap row sampling + random feature
            # subsetting at each split both depend on random_state, so this
            # model DOES exhibit real seed-to-seed variance -- used here to
            # demonstrate the seed-sensitivity check requested for this
            # session, complementing the deterministic logreg baseline.
            clf = RandomForestClassifier(
                n_estimators=100, random_state=seed
            ).fit(X_tr_s, y_tr)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        y_proba = clf.predict_proba(X_te_s)[:, 1]
        y_pred = clf.predict(X_te_s)

        auc = roc_auc_score(y_te, y_proba)
        acc = accuracy_score(y_te, y_pred)
        prauc = average_precision_score(y_te, y_proba)
        sens90 = sensitivity_at_specificity(y_te, y_proba, target_specificity=0.90)

        fold_aucs.append(auc)
        fold_accs.append(acc)
        fold_praucs.append(prauc)
        fold_sens.append(sens90)

        print(f"  [{model_name}] fold={fold_idx}  test_blocks={sorted(set(groups[test_idx]))}  "
              f"AUC-ROC={auc:.4f}  PR-AUC={prauc:.4f}  Sens@Spec90={sens90:.4f}  Acc={acc:.4f}")

    return {
        'auc_roc_mean': float(np.mean(fold_aucs)),
        'auc_roc_std': float(np.std(fold_aucs)),
        'pr_auc_mean': float(np.mean(fold_praucs)),
        'pr_auc_std': float(np.std(fold_praucs)),
        'sensitivity_at_spec90_mean': float(np.mean(fold_sens)),
        'accuracy_mean': float(np.mean(fold_accs)),
    }


def main(seed, n_splits=5):
    set_seed(seed)

    # --- SYNTHETIC PLACEHOLDER DATA -----------------------------------------
    # This dataset validates the reproducibility infrastructure (Git + DVC +
    # MLflow + Docker) per Dr. Gaur's feedback. It is NOT real CDC-MINSA /
    # SENASA epidemiological data and these results must never be read as
    # real model performance. The 'block_id' column simulates spatial blocks
    # (e.g., districts or grid cells); when real georeferenced records are
    # available, block_id should be derived from actual lat/lon coordinates
    # (e.g., via a spatial grid or administrative district -- Meyer et al.,
    # 2019).
    # -------------------------------------------------------------------------
    df = pd.read_csv('data/rabies_data.csv')

    feature_cols = [c for c in df.columns if c not in ['target', 'block_id', 'lat', 'lon']]
    X = df[feature_cols].values
    y = df['target'].values
    groups = df['block_id'].values  # spatial blocking unit

    # Spatial cross-validation: GroupKFold ensures that all rows sharing the
    # same spatial block stay together in either train OR test, never split
    # across both. This prevents spatial leakage between neighboring
    # locations (Meyer et al., 2019), unlike a random row-wise split.
    #
    # NOTE: GroupKFold has no random_state -- fold assignment is deterministic
    # by design, so 'test_blocks' below is identical across all seeds. This
    # is expected and correct for spatial CV.
    print(f"=== Logistic Regression (deterministic baseline) ===")
    logreg_results = run_model('logreg', X, y, groups, seed, n_splits)

    print(f"\n=== Random Forest (seed-sensitive comparison model) ===")
    rf_results = run_model('random_forest', X, y, groups, seed, n_splits)

    print(f"\nseed={seed}")
    print(f"  [logreg]        AUC-ROC={logreg_results['auc_roc_mean']:.4f}±{logreg_results['auc_roc_std']:.4f}  "
          f"PR-AUC={logreg_results['pr_auc_mean']:.4f}±{logreg_results['pr_auc_std']:.4f}  "
          f"Sens@Spec90={logreg_results['sensitivity_at_spec90_mean']:.4f}  "
          f"Accuracy={logreg_results['accuracy_mean']:.4f}")
    print(f"  [random_forest] AUC-ROC={rf_results['auc_roc_mean']:.4f}±{rf_results['auc_roc_std']:.4f}  "
          f"PR-AUC={rf_results['pr_auc_mean']:.4f}±{rf_results['pr_auc_std']:.4f}  "
          f"Sens@Spec90={rf_results['sensitivity_at_spec90_mean']:.4f}  "
          f"Accuracy={rf_results['accuracy_mean']:.4f}")

    return {'logreg': logreg_results, 'random_forest': rf_results}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_splits', type=int, default=5)
    args = ap.parse_args()
    main(args.seed, args.n_splits)