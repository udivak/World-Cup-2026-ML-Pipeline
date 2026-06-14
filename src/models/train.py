"""Model estimators for the W/D/L profile model (Phase 3).

Two contenders, both wrapped in probability calibration by default:

* **Multinomial Logistic Regression** — the interpretable spine. Median-imputes the (sometimes
  sparse) unit-strength diffs, standard-scales, then fits a regularized multinomial logit. Small-N
  friendly and the source of signed, readable coefficients.
* **HistGradientBoostingClassifier** — handles NaNs natively (no imputation) and captures
  non-linear interactions; a stronger fit when the data supports it.

Both expose the standard scikit-learn ``fit`` / ``predict_proba`` API, so the backtest harness,
the baselines, and live 2026 scoring all drive them identically (no train/serve skew).
"""

from typing import Optional

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.calibrate import calibrate
from src.models.metrics import CLASSES  # noqa: F401  (re-exported for callers)

RANDOM_STATE = 42


def build_logreg(C: float = 0.5, calibrated: bool = True, cv: int = 3):
    """Regularized multinomial logistic regression on imputed+scaled features."""
    pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
        ]
    )
    return calibrate(pipe, cv=cv) if calibrated else pipe


def build_hgb(calibrated: bool = True, cv: int = 3, random_state: int = RANDOM_STATE):
    """Gradient-boosted trees; native NaN handling, regularized for the small tournament set."""
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )
    return calibrate(clf, cv=cv) if calibrated else clf


def build_model(name: str, calibrated: bool = True, **kwargs):
    """Factory by name: ``'logreg'`` or ``'hgb'``."""
    if name == "logreg":
        return build_logreg(calibrated=calibrated, **kwargs)
    if name == "hgb":
        return build_hgb(calibrated=calibrated, **kwargs)
    raise ValueError(f"unknown model '{name}' (expected 'logreg' or 'hgb')")


def fit_predict_proba(estimator, X_train, y_train, X_test, classes: Optional[list] = None):
    """Fit and return test ``predict_proba`` reordered to :data:`CLASSES` column order.

    scikit-learn orders probability columns by ``estimator.classes_`` (sorted); we reindex to the
    ordinal ``[H, D, A]`` so RPS and downstream code see a stable column order even if a training
    fold is missing a class.
    """
    import numpy as np

    classes = classes or CLASSES
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)
    col = {c: i for i, c in enumerate(estimator.classes_)}
    out = np.zeros((proba.shape[0], len(classes)), dtype=float)
    for j, c in enumerate(classes):
        if c in col:
            out[:, j] = proba[:, col[c]]
    # Renormalize in case a class was absent in training (column stays 0).
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums
