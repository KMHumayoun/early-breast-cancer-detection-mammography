"""Lock the calibration and cutoff on site 1 before evaluating site 2."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .network import ScreeningModel
from .runtime import device, loader, logits, read_index


def operating_point(negative_scores, specificity=.90):
    values = np.sort(np.asarray(negative_scores))
    if not len(values):
        raise ValueError("Calibration needs negative breasts")
    # Using >= at prediction time, move above the tied boundary to honor the target.
    return float(np.nextafter(values[int(np.ceil(specificity * len(values))) - 1], np.inf))


def measures(y, probability, threshold):
    positive = y == 1
    flagged = probability >= threshold
    return dict(average_precision=float(average_precision_score(y, probability)),
                roc_auc=float(roc_auc_score(y, probability)),
                brier=float(brier_score_loss(y, probability)),
                sensitivity=float(flagged[positive].mean()),
                specificity=float(1-flagged[~positive].mean()),
                fraction_flagged=float(flagged.mean()))


def compare(y, patients, scores, thresholds, draws, seed):
    # Keep both breasts from a sampled patient together. Reuse each draw across models.
    unique, inverse = np.unique(patients, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    clusters = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    point = {name: measures(y, p, thresholds[name]) for name, p in scores.items()}
    differences = []
    draws_by_model = {name: [] for name in scores}
    for _ in range(draws):
        ids = rng.integers(len(unique), size=len(unique))
        idx = np.concatenate([clusters[i] for i in ids])
        if len(np.unique(y[idx])) < 2:
            continue
        for name, p in scores.items():
            draws_by_model[name].append(measures(y[idx], p[idx], thresholds[name]))
        differences.append(draws_by_model["paired"][-1]["average_precision"]
                           - draws_by_model["single"][-1]["average_precision"])
    if not differences:
        raise ValueError("Bootstrap yielded no two-class samples")
    def interval(values):
        return np.percentile(values, [2.5, 97.5]).tolist()
    return dict(test_patients=len(unique), test_breasts=len(y), test_cancers=int(y.sum()),
                test_prevalence=float(y.mean()), metrics=point,
                ci95={n:{k:interval([v[k] for v in ds]) for k in point[n]}
                      for n, ds in draws_by_model.items()},
                paired_ap_difference=point["paired"]["average_precision"]-point["single"]["average_precision"],
                paired_ap_difference_ci95=interval(differences),
                bootstrap_requested=draws, bootstrap_valid=len(differences))


def evaluate(args):
    rows, digest = read_index(args.index)
    cal = [r for r in rows if r["split"] == "calibrate"]
    test = [r for r in rows if r["split"] == "test"]
    cal_y = np.array([r["cancer"] for r in cal])
    y = np.array([r["cancer"] for r in test])
    patients = np.array([r["patient"] for r in test])
    where = device()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Choose a fresh output folder: {out}")
    scores, thresholds, calibration = {}, {}, {}
    heights, recipes = set(), set()
    for name in ("single", "paired"):
        folder = Path(getattr(args, name))
        completion = json.loads((folder/"complete.json").read_text())
        saved = torch.load(folder/"best.pt", map_location="cpu", weights_only=True)
        views = ["CC","MLO"] if name == "paired" else ["MLO"]
        if saved["manifest"] != digest or completion["manifest"] != digest or saved["views"] != views:
            raise ValueError("Models must use this manifest and the specified views")
        heights.add(saved["height"])
        recipes.add((saved["seed"],saved["pretrained"],completion["epochs"]))
        if len(heights) > 1 or len(recipes) > 1:
            raise ValueError("Use matching resolution, seed, initialization and epoch budget")
        args.height = saved["height"]
        model = ScreeningModel(len(views), pretrained=False).to(where)
        model.load_state_dict(saved["state"])
        cal_z = logits(model, loader(cal, args, views), where)
        platt = LogisticRegression(C=1.0).fit(cal_z[:,None], cal_y)
        cal_p = platt.predict_proba(cal_z[:,None])[:,1]
        thresholds[name] = operating_point(cal_p[cal_y == 0])
        calibration[name] = dict(slope=float(platt.coef_[0,0]), intercept=float(platt.intercept_[0]),
                                 cutoff=thresholds[name], selection_epoch=saved["epoch"])
        test_z = logits(model, loader(test, args, views), where)
        scores[name] = platt.predict_proba(test_z[:,None])[:,1]
    report = compare(y, patients, scores, thresholds, args.bootstrap, args.seed)
    operating_counts = {}
    for name, values in scores.items():
        flagged = values >= thresholds[name]
        operating_counts[name] = {
            "true_positive": int(np.sum(flagged & (y == 1))),
            "false_negative": int(np.sum(~flagged & (y == 1))),
            "false_positive": int(np.sum(flagged & (y == 0))),
            "true_negative": int(np.sum(~flagged & (y == 0))),
        }
    report.update(manifest=digest, calibration=calibration, operating_counts=operating_counts)
    out.mkdir(parents=True, exist_ok=False)
    (out/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out/"private_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient","side","outcome","single_score","paired_score"])
        for i, row in enumerate(test):
            writer.writerow([row["patient"],row["side"],int(y[i]),scores["single"][i],scores["paired"][i]])
    from .plots import plot_test_results
    plot_test_results(y, scores, out)
    return report
