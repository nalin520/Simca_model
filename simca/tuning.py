import numpy as np
from sklearn.model_selection import StratifiedKFold
from simca.model import MultiClassSIMCA
from simca.feature_engineering import build_feature_matrix

def simca_hyper_tune(
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        *,
        candidate_alphas=(0.05, 0.02, 0.01),
        candidate_kmax=(10, 15, 20),
        var_thresholds=(0.90, 0.95)
    ) -> MultiClassSIMCA:
    """
    Very light hyper-parameter sweep ⇒ returns the best-CV SIMCA model.
    """
    best_acc, best_model = -1, None
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)

    for a in candidate_alphas:
        for km in candidate_kmax:
            for vt in var_thresholds:
                cv_scores = []
                for tr_idx, va_idx in skf.split(X_tr, y_tr):
                    m = MultiClassSIMCA(
                        n_components=vt,
                        alpha=a,
                        cv_max_k=km,
                        cv_folds=3,
                        random_state=1
                    )
                    m.fit(X_tr[tr_idx], y_tr[tr_idx])
                    preds, _ = m.predict(X_tr[va_idx])
                    cv_scores.append(np.mean(preds == y_tr[va_idx]))
                score = np.mean(cv_scores)
                if score > best_acc:
                    best_acc, best_model = score, m
    print(f"Best CV accuracy = {best_acc:.3f}")
    return best_model
