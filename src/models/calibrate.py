"""Probability calibration (Phase 3).

The project principle is *calibrate everything*: model probabilities should match observed
frequencies (a "60% team1-win" should win ~60% of the time). We wrap estimators in
scikit-learn's :class:`CalibratedClassifierCV`. With the small tournament set we default to
**Platt/sigmoid** scaling (isotonic overfits on a few hundred rows) and a modest internal CV.
"""

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV


def calibrate(
    estimator: BaseEstimator, method: str = "sigmoid", cv: int = 3
) -> CalibratedClassifierCV:
    """Wrap ``estimator`` in cross-validated probability calibration.

    ``method='sigmoid'`` (Platt) is robust on small data; ``cv`` folds are fit internally on the
    training set only, so calibration introduces no leakage in the across-edition backtest.
    """
    return CalibratedClassifierCV(estimator, method=method, cv=cv)
