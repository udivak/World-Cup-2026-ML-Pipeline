"""Across-edition expanding-window backtest + success gate (Phase 3).

This is the project's success gate. We never use a random split: models are trained on
tournament editions that finished **before** a held-out edition and judged on that future
edition, rolling the cutoff forward one edition at a time (an expanding window). Pooling the
held-out predictions across all editions gives the headline metrics.

Candidates (all calibrated identically):
  * ``Elo-only``        — reference baseline (team identity)
  * ``Squad-overall``   — reference baseline (naive bottom-up: ``diff_overall_xi`` + home adv)
  * ``LogReg profile``  — full profile-difference vector, multinomial logit
  * ``HGB profile``     — full profile-difference vector, gradient-boosted trees
  * ``Ensemble``        — mean of the two profile models' probabilities

GATE (printed PASS/FAIL): the best profile model must beat **both** baselines on **RPS and
log-loss** over the pooled held-out predictions.

Run: ``python -m src.models.evaluate``  (prints the comparison table, per-edition RPS, the gate
verdict, and profile-feature importances).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.features.build_features import FEATURE_COLUMNS, build_feature_frame
from src.models.baselines import (
    ELO_FEATURES,
    SQUAD_OVERALL_FEATURES,
    attach_elo,
    build_elo_baseline,
    build_squad_overall_baseline,
    compute_elo_features,
)
from src.models.metrics import CLASSES, per_class_precision_recall, score_all
from src.models.train import build_hgb, build_logreg, fit_predict_proba

logger = logging.getLogger(__name__)

# An edition only enters the held-out evaluation once enough earlier matches exist to train on.
MIN_TRAIN_MATCHES = 100

BASELINES = ("Elo-only", "Squad-overall")
PROFILE_MODELS = ("LogReg profile", "HGB profile", "Ensemble")


def prepare_dataset() -> pd.DataFrame:
    """Build the feature frame and attach the Elo baseline feature (single code path)."""
    from src.common.io import read_table
    from src.features.labels import load_tournament_editions

    cfg = load_config()
    feats = build_feature_frame()
    feats["edition_year"] = feats["edition_year"].astype(int)

    # Attach each edition's snapshot_date (the across-edition train cutoff).
    editions = load_tournament_editions()
    editions["edition_year"] = editions["edition_year"].astype(int)
    feats = feats.merge(
        editions[["tournament", "edition_year", "snapshot_date"]],
        on=["tournament", "edition_year"], how="left",
    )

    elo = compute_elo_features(
        read_table("matches"),
        k=cfg.elo.k,
        home_advantage=cfg.elo.home_advantage,
        mov=cfg.elo.mov,
    )
    feats = attach_elo(feats, elo)
    feats["date"] = pd.to_datetime(feats["date"])
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    feats["edition"] = feats["tournament"] + " " + feats["edition_year"].astype(str)
    return feats.sort_values("date").reset_index(drop=True)


def _fold_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit every candidate on ``train`` and return ``{model: proba[n_test, 3]}`` for ``test``."""
    y_tr = train["result"].tolist()
    proba = {
        "Elo-only": fit_predict_proba(
            build_elo_baseline(), train[ELO_FEATURES], y_tr, test[ELO_FEATURES]
        ),
        "Squad-overall": fit_predict_proba(
            build_squad_overall_baseline(), train[SQUAD_OVERALL_FEATURES], y_tr,
            test[SQUAD_OVERALL_FEATURES],
        ),
        "LogReg profile": fit_predict_proba(
            build_logreg(calibrated=True), train[FEATURE_COLUMNS], y_tr, test[FEATURE_COLUMNS]
        ),
        "HGB profile": fit_predict_proba(
            build_hgb(calibrated=True), train[FEATURE_COLUMNS], y_tr, test[FEATURE_COLUMNS]
        ),
    }
    proba["Ensemble"] = (proba["LogReg profile"] + proba["HGB profile"]) / 2.0
    return proba


def run_backtest(feats: Optional[pd.DataFrame] = None, min_train: int = MIN_TRAIN_MATCHES) -> dict:
    """Expanding-window across-edition backtest. Returns pooled predictions + metadata."""
    feats = prepare_dataset() if feats is None else feats
    editions = (
        feats[["edition", "snapshot_date"]].drop_duplicates().sort_values("snapshot_date")
    )

    models = list(BASELINES) + list(PROFILE_MODELS)
    proba_acc: dict = {m: [] for m in models}
    y_all: list = []
    edition_all: list = []
    tested_editions: list = []

    for _, ed in editions.iterrows():
        snap = ed["snapshot_date"]
        test = feats[feats["edition"] == ed["edition"]]
        train = feats[feats["date"] < snap]
        if len(train) < min_train or test.empty:
            continue
        fold = _fold_probabilities(train, test)
        for m in models:
            proba_acc[m].append(fold[m])
        y_all.extend(test["result"].tolist())
        edition_all.extend([ed["edition"]] * len(test))
        tested_editions.append({"edition": ed["edition"], "snapshot_date": snap,
                                "n_train": len(train), "n_test": len(test)})

    P = {m: np.vstack(proba_acc[m]) for m in models}
    return {
        "models": models,
        "P": P,
        "y": y_all,
        "edition": edition_all,
        "tested_editions": pd.DataFrame(tested_editions),
        "features": feats,
    }


def summarize(bt: dict) -> pd.DataFrame:
    """Pooled metrics per model, sorted by RPS (primary)."""
    rows = []
    for m in bt["models"]:
        s = score_all(bt["y"], bt["P"][m])
        rows.append({"model": m, **{k: s[k] for k in
                    ("rps", "log_loss", "accuracy", "precision_macro", "recall_macro", "n")}})
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)


def per_edition_rps(bt: dict) -> pd.DataFrame:
    """RPS per held-out edition per model (wide table)."""
    from src.models.metrics import ranked_probability_score

    y = np.array(bt["y"])
    ed = np.array(bt["edition"])
    rows = []
    for e in pd.unique(ed):
        mask = ed == e
        row = {"edition": e, "n": int(mask.sum())}
        for m in bt["models"]:
            row[m] = round(ranked_probability_score(y[mask].tolist(), bt["P"][m][mask]), 4)
        rows.append(row)
    order = {r["edition"]: i for i, r in bt["tested_editions"].reset_index().iterrows()} \
        if not bt["tested_editions"].empty else {}
    out = pd.DataFrame(rows)
    if not bt["tested_editions"].empty:
        merged = out.merge(bt["tested_editions"][["edition", "snapshot_date"]], on="edition")
        out = merged.sort_values("snapshot_date").drop(columns="snapshot_date").reset_index(drop=True)
    return out


def gate_verdict(summary: pd.DataFrame) -> dict:
    """Best profile model must beat BOTH baselines on RPS AND log-loss."""
    base = summary[summary["model"].isin(BASELINES)]
    prof = summary[summary["model"].isin(PROFILE_MODELS)]
    best_baseline_rps = base["rps"].min()
    best_baseline_ll = base["log_loss"].min()
    best = prof.sort_values("rps").iloc[0]
    passed = bool(best["rps"] < best_baseline_rps and best["log_loss"] < best_baseline_ll)
    return {
        "passed": passed,
        "best_model": best["model"],
        "best_rps": float(best["rps"]),
        "best_log_loss": float(best["log_loss"]),
        "baseline_rps": float(best_baseline_rps),
        "baseline_log_loss": float(best_baseline_ll),
        "rps_baseline_model": base.sort_values("rps").iloc[0]["model"],
        "ll_baseline_model": base.sort_values("log_loss").iloc[0]["model"],
    }


def profile_importances(feats: pd.DataFrame, cutoff: str = "2023-01-01", n_repeats: int = 20) -> pd.DataFrame:
    """Permutation importance (ΔRPS) + signed LogReg coefficients on a held-out split."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer

    from src.models.metrics import ranked_probability_score

    cut = pd.Timestamp(cutoff)
    train, test = feats[feats["date"] < cut], feats[feats["date"] >= cut]
    X_tr, y_tr = train[FEATURE_COLUMNS], train["result"]
    X_te, y_te = test[FEATURE_COLUMNS], test["result"]

    model = build_logreg(calibrated=True)
    model.fit(X_tr, y_tr)

    def neg_rps(estimator, X, y):
        proba = estimator.predict_proba(X)
        col = {c: i for i, c in enumerate(estimator.classes_)}
        ordered = np.column_stack([proba[:, col[c]] for c in CLASSES])
        return -ranked_probability_score(list(y), ordered)  # higher is better

    imp = permutation_importance(
        model, X_te, y_te, scoring=neg_rps, n_repeats=n_repeats, random_state=42
    )
    # Signed coefficients from a transparent (uncalibrated) logit on the same training split.
    plain = build_logreg(calibrated=False).fit(X_tr, y_tr)
    coef_home = plain.named_steps["lr"].coef_[list(plain.named_steps["lr"].classes_).index("H")]

    return pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "perm_importance_rps": imp.importances_mean,
        "logit_coef_team1_win": coef_home,
    }).sort_values("perm_importance_rps", ascending=False).reset_index(drop=True)


def _print_report(bt: dict, summary: pd.DataFrame, per_ed: pd.DataFrame,
                  verdict: dict, importances: pd.DataFrame) -> None:
    print("\n=================== Phase 3 backtest (across-edition expanding window) ===================")
    print(f"Held-out editions: {len(bt['tested_editions'])}   pooled held-out matches: {len(bt['y'])}")
    print(f"(min {MIN_TRAIN_MATCHES} training matches required before an edition is scored)\n")

    print(f"{'model':22s} {'RPS':>8} {'logloss':>9} {'acc':>7} {'prec':>7} {'recall':>7}")
    print("-" * 66)
    for _, r in summary.iterrows():
        tag = "  [baseline]" if r["model"] in BASELINES else ""
        print(f"{r['model']:22s} {r['rps']:8.4f} {r['log_loss']:9.4f} {r['accuracy']:7.3f} "
              f"{r['precision_macro']:7.3f} {r['recall_macro']:7.3f}{tag}")

    print("\nPer-edition RPS (lower is better):")
    cols = [c for c in per_ed.columns if c not in ("n",)]
    header = "  " + f"{'edition':28s} {'n':>3} " + " ".join(f"{m[:10]:>10}" for m in bt["models"])
    print(header)
    for _, r in per_ed.iterrows():
        print("  " + f"{r['edition']:28s} {int(r['n']):>3} " +
              " ".join(f"{r[m]:>10.4f}" for m in bt["models"]))

    print("\n--------------------------------- GATE ---------------------------------")
    v = verdict
    print(f"Best profile model        : {v['best_model']}")
    print(f"  RPS      {v['best_rps']:.4f}  vs best baseline {v['baseline_rps']:.4f} "
          f"({v['rps_baseline_model']})  -> {'beats' if v['best_rps']<v['baseline_rps'] else 'DOES NOT beat'}")
    print(f"  log-loss {v['best_log_loss']:.4f}  vs best baseline {v['baseline_log_loss']:.4f} "
          f"({v['ll_baseline_model']})  -> {'beats' if v['best_log_loss']<v['baseline_log_loss'] else 'DOES NOT beat'}")
    print(f"\n  GATE: {'PASS ✅  profile model beats both baselines on RPS and log-loss' if v['passed'] else 'FAIL ❌  does not beat both baselines on both metrics'}")
    print("------------------------------------------------------------------------")

    print("\nProfile-feature importance (permutation ΔRPS on 2023+ hold-out; + signed team1-win logit coef):")
    for _, r in importances.head(10).iterrows():
        print(f"  {r['feature']:24s} ΔRPS={r['perm_importance_rps']:+.5f}  coef={r['logit_coef_team1_win']:+.3f}")
    print("=========================================================================================")


def run_and_report() -> dict:
    feats = prepare_dataset()
    bt = run_backtest(feats)
    summary = summarize(bt)
    per_ed = per_edition_rps(bt)
    verdict = gate_verdict(summary)
    importances = profile_importances(feats)
    _print_report(bt, summary, per_ed, verdict, importances)
    return {"summary": summary, "per_edition": per_ed, "verdict": verdict,
            "importances": importances, "backtest": bt}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    import warnings

    warnings.filterwarnings("ignore")
    run_and_report()
