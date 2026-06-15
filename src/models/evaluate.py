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
from src.features.build_features import MODEL_FEATURES, build_feature_frame
from src.models.baselines import (
    ELO_FEATURES,
    SQUAD_OVERALL_FEATURES,
    attach_elo,
    build_elo_baseline,
    build_squad_overall_baseline,
    compute_elo_features,
)
from src.models.blend import (
    blend_weight_sweep,
    error_correlation,
    nested_blend_eval,
    select_weight_loeo,
)
from src.models.metrics import (
    CLASSES,
    multiclass_log_loss,
    per_class_precision_recall,
    ranked_probability_score,
    score_all,
)
from src.models.train import build_hgb, build_logreg, fit_predict_proba

logger = logging.getLogger(__name__)

# An edition only enters the held-out evaluation once enough earlier matches exist to train on.
MIN_TRAIN_MATCHES = 100

BASELINES = ("Elo-only", "Squad-overall")
PROFILE_MODELS = ("LogReg profile", "HGB profile", "Ensemble")
# Product model — served in Phase 4, NEVER a gate pass (uses Elo). Kept separate from PROFILE_MODELS
# so the gate (which judges bottom-up models only) cannot be satisfied by the blend.
BLEND_MODEL = "Blend (Elo+profile)"
PRODUCT_MODELS = (BLEND_MODEL,)
# RPS-error correlation above this means the two members miss on the same matches — blending can't
# decorrelate, so it would be ~a no-op (the §5.0 precondition gate).
MAX_ERROR_CORRELATION = 0.8
# Adopt the logistic stacker over the global-weight linear pool only if it beats it on nested RPS by
# at least this much. Below it the difference is noise, and the 1-DoF pool is the more robust serve.
STACKER_MIN_RPS_GAIN = 0.001


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
            build_logreg(calibrated=True), train[MODEL_FEATURES], y_tr, test[MODEL_FEATURES]
        ),
        "HGB profile": fit_predict_proba(
            build_hgb(calibrated=True), train[MODEL_FEATURES], y_tr, test[MODEL_FEATURES]
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


def report_models(bt: dict) -> list:
    """All models to show in the comparison: baselines + profile + any attached product model."""
    return list(bt["models"]) + [m for m in PRODUCT_MODELS if m in bt["P"]]


def _build_blend_folds(bt: dict, profile_model: str) -> list:
    """Per-edition held-out predictions for the nested blend — ``profile`` vs ``Elo-only``.

    Each fold is one tested edition: its row positions in the pooled arrays (``idx``), the two
    members' out-of-sample probabilities for those rows, the labels, and ``snapshot_date`` as the
    chronological ``order`` key. Both members are already leakage-free (trained on strictly-prior
    editions in :func:`run_backtest`).
    """
    ed = np.array(bt["edition"])
    y = np.array(bt["y"])
    p_profile, p_elo = bt["P"][profile_model], bt["P"]["Elo-only"]
    order = {r["edition"]: r["snapshot_date"] for _, r in bt["tested_editions"].iterrows()}
    folds = []
    for e in bt["tested_editions"].sort_values("snapshot_date")["edition"]:
        idx = np.where(ed == e)[0]
        if idx.size == 0:
            continue
        folds.append({"edition": e, "order": order[e], "idx": idx,
                      "p_a": p_profile[idx], "p_b": p_elo[idx], "y": y[idx].tolist()})
    return folds


def evaluate_blend(bt: dict, cfg) -> dict:
    """Blend the profile model with Elo (product model) and attach the served prediction to ``bt``.

    Runs the §5.0–§5.4 protocol on the already-pooled, leakage-free fold outputs: the error-corr
    precondition, the LOEO weight sweep + global ``w*``, and the **nested** bake-off between the
    linear pool and the logistic stacker. The combiner with the lower nested RPS wins; its nested
    (leakage-free) pooled prediction is stored in ``bt["P"][BLEND_MODEL]`` so it appears as a
    ``[product]`` row alongside the others. The gate (profile-only) is untouched.
    """
    profile_model = cfg.blend.profile_model if cfg.blend.profile_model in bt["P"] else "Ensemble"
    p_a, p_b, y = bt["P"][profile_model], bt["P"]["Elo-only"], bt["y"]

    corr = error_correlation(p_a, p_b, y)
    sweep = blend_weight_sweep(p_a, p_b, y)
    loeo_w = select_weight_loeo(p_a, p_b, y)
    elo_rps = ranked_probability_score(y, p_b)
    elo_ll = multiclass_log_loss(y, p_b)

    folds = _build_blend_folds(bt, profile_model)
    nested_linear = nested_blend_eval(folds, y, combiner="linear")
    nested_stacker = nested_blend_eval(folds, y, combiner="stacker")

    # Bake-off: default to the simpler, more robust global-weight pool (≈1 DoF). Adopt the stacker
    # only if it beats the linear pool's nested RPS by a *meaningful* margin — the plan favours the
    # scalar pool on small N unless the meta-layer clearly earns its extra parameters.
    chosen = nested_stacker \
        if (nested_linear["rps"] - nested_stacker["rps"]) > STACKER_MIN_RPS_GAIN else nested_linear

    # Robustness band: the contiguous run of weights (excluding the all-Elo endpoint) whose pooled
    # RPS beats Elo — the blend should win across a broad band, not a knife-edge.
    beats = sweep[(sweep["w"] > 0.0) & (sweep["rps"] < elo_rps)]["w"]
    band = (float(beats.min()), float(beats.max())) if not beats.empty else None

    bt["P"][BLEND_MODEL] = chosen["pred"]
    return {
        "profile_model": profile_model,
        "error_correlation": corr,
        "precondition_ok": corr < MAX_ERROR_CORRELATION,
        "sweep": sweep,
        "loeo_weight": loeo_w,
        "beats_elo_band": band,
        "elo_rps": elo_rps,
        "elo_log_loss": elo_ll,
        "nested_linear": nested_linear,
        "nested_stacker": nested_stacker,
        "chosen_combiner": chosen["combiner"],
        "chosen_nested": chosen,
        "beats_elo": bool(chosen["rps"] < elo_rps and chosen["log_loss"] < elo_ll),
    }


def summarize(bt: dict) -> pd.DataFrame:
    """Pooled metrics per model, sorted by RPS (primary). Includes the product blend if attached."""
    rows = []
    for m in report_models(bt):
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
        for m in report_models(bt):
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
    X_tr, y_tr = train[MODEL_FEATURES], train["result"]
    X_te, y_te = test[MODEL_FEATURES], test["result"]

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
        "feature": MODEL_FEATURES,
        "perm_importance_rps": imp.importances_mean,
        "logit_coef_team1_win": coef_home,
    }).sort_values("perm_importance_rps", ascending=False).reset_index(drop=True)


def _print_blend_section(blend: dict) -> None:
    b = blend
    print("\n------------------- PRODUCT MODEL — Blend (Elo + profile) -------------------")
    print("(NOT a gate pass: Elo is a blend input. The bottom-up GATE above is the research result.)")
    print(f"Profile member pooled with Elo : {b['profile_model']}")

    corr = b["error_correlation"]
    if corr < 0.5:
        corr_flag = "low — errors decorrelated, blend should help"
    elif corr < MAX_ERROR_CORRELATION:
        corr_flag = "ELEVATED — limited decorrelation; nested result is the decider"
    else:
        corr_flag = f"HIGH (> {MAX_ERROR_CORRELATION}) — errors move together, blend is ~a no-op"
    print(f"Precondition  per-match RPS-error corr : {corr:+.3f}  [{corr_flag}]")

    band = b["beats_elo_band"]
    band_txt = f"w ∈ [{band[0]:.2f}, {band[1]:.2f}]" if band else "none"
    print(f"LOEO sweep    global w* (profile share) = {b['loeo_weight']:.2f}   "
          f"beats-Elo band: {band_txt}")
    sweep = b["sweep"]
    marks = sweep[np.isclose(sweep["w"] % 0.1, 0.0) | np.isclose(sweep["w"] % 0.1, 0.1)]
    print("    w     " + " ".join(f"{w:>6.1f}" for w in marks["w"]))
    print("    RPS   " + " ".join(f"{r:>6.4f}" for r in marks["rps"]))

    nl, ns = b["nested_linear"], b["nested_stacker"]
    print("Combiner bake-off (nested across-edition; honest out-of-sample):")
    for name, res in (("linear pool", nl), ("logistic stacker", ns)):
        win = "  <- chosen" if res["combiner"] == b["chosen_combiner"] else ""
        print(f"  {name:18s} nested RPS {res['rps']:.4f}   log-loss {res['log_loss']:.4f}{win}")

    cn = b["chosen_nested"]
    print(f"\nServed blend (nested, {b['chosen_combiner']}):")
    print(f"  RPS      {cn['rps']:.4f}  vs Elo {b['elo_rps']:.4f}  -> "
          f"{'beats' if cn['rps'] < b['elo_rps'] else 'DOES NOT beat'}")
    print(f"  log-loss {cn['log_loss']:.4f}  vs Elo {b['elo_log_loss']:.4f}  -> "
          f"{'beats' if cn['log_loss'] < b['elo_log_loss'] else 'DOES NOT beat'}")
    print(f"  PRODUCT: {'beats Elo on RPS and log-loss ✅ (serve the blend in 2026)' if b['beats_elo'] else 'does NOT beat Elo on both metrics'}")
    print("----------------------------------------------------------------------------")


def _print_report(bt: dict, summary: pd.DataFrame, per_ed: pd.DataFrame,
                  verdict: dict, importances: pd.DataFrame, blend: Optional[dict] = None) -> None:
    models = report_models(bt)
    print("\n=================== Phase 3 backtest (across-edition expanding window) ===================")
    print(f"Held-out editions: {len(bt['tested_editions'])}   pooled held-out matches: {len(bt['y'])}")
    print(f"(min {MIN_TRAIN_MATCHES} training matches required before an edition is scored)\n")

    print(f"{'model':22s} {'RPS':>8} {'logloss':>9} {'acc':>7} {'prec':>7} {'recall':>7}")
    print("-" * 66)
    for _, r in summary.iterrows():
        tag = "  [baseline]" if r["model"] in BASELINES else \
            ("  [product]" if r["model"] in PRODUCT_MODELS else "")
        print(f"{r['model']:22s} {r['rps']:8.4f} {r['log_loss']:9.4f} {r['accuracy']:7.3f} "
              f"{r['precision_macro']:7.3f} {r['recall_macro']:7.3f}{tag}")

    print("\nPer-edition RPS (lower is better):")
    header = "  " + f"{'edition':28s} {'n':>3} " + " ".join(f"{m[:10]:>10}" for m in models)
    print(header)
    for _, r in per_ed.iterrows():
        print("  " + f"{r['edition']:28s} {int(r['n']):>3} " +
              " ".join(f"{r[m]:>10.4f}" for m in models))

    print("\n--------------------------------- GATE ---------------------------------")
    v = verdict
    print(f"Best profile model        : {v['best_model']}")
    print(f"  RPS      {v['best_rps']:.4f}  vs best baseline {v['baseline_rps']:.4f} "
          f"({v['rps_baseline_model']})  -> {'beats' if v['best_rps']<v['baseline_rps'] else 'DOES NOT beat'}")
    print(f"  log-loss {v['best_log_loss']:.4f}  vs best baseline {v['baseline_log_loss']:.4f} "
          f"({v['ll_baseline_model']})  -> {'beats' if v['best_log_loss']<v['baseline_log_loss'] else 'DOES NOT beat'}")
    print(f"\n  GATE: {'PASS ✅  profile model beats both baselines on RPS and log-loss' if v['passed'] else 'FAIL ❌  does not beat both baselines on both metrics'}")
    print("------------------------------------------------------------------------")

    if blend is not None:
        _print_blend_section(blend)

    print("\nProfile-feature importance (permutation ΔRPS on 2023+ hold-out; + signed team1-win logit coef):")
    for _, r in importances.head(10).iterrows():
        print(f"  {r['feature']:24s} ΔRPS={r['perm_importance_rps']:+.5f}  coef={r['logit_coef_team1_win']:+.3f}")
    print("=========================================================================================")


def run_and_report() -> dict:
    cfg = load_config()
    feats = prepare_dataset()
    bt = run_backtest(feats)
    # Gate verdict is computed on the PROFILE-ONLY models BEFORE the blend is attached, so the
    # product blend can never leak into the bottom-up gate.
    verdict = gate_verdict(summarize(bt))
    blend = evaluate_blend(bt, cfg)  # attaches BLEND_MODEL to bt["P"]
    summary = summarize(bt)
    per_ed = per_edition_rps(bt)
    importances = profile_importances(feats)
    _print_report(bt, summary, per_ed, verdict, importances, blend)
    return {"summary": summary, "per_edition": per_ed, "verdict": verdict,
            "importances": importances, "backtest": bt, "blend": blend}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    import warnings

    warnings.filterwarnings("ignore")
    run_and_report()
