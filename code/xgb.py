import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression as LR

def eval_xgb(feat, name, scale_pos_weight=6.47):
    aucs, f1s = [], []
    for tr_idx, va_idx, te_idx in splits:
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', random_state=42, verbosity=0
        )
        clf.fit(feat[tr_idx], labels[tr_idx],
                eval_set=[(feat[va_idx], labels[va_idx])],
                verbose=False)
        probs = clf.predict_proba(feat[te_idx])[:,1]
        aucs.append(roc_auc_score(labels[te_idx], probs))
        preds = (probs >= 0.5).astype(int)
        f1s.append(f1_score(labels[te_idx], preds, average='macro', zero_division=0))
    print(f"  {name:35s}: AUROC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
          f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")
    return np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s)