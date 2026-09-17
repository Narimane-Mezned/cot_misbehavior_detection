import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support


def youden_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def confusion_at(labels, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def detection_metrics(labels, scores, threshold=None, attack_types=None):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"error": "labels contain only one class; AUC undefined",
                "n_positive": int(labels.sum()), "n_negative": int((labels == 0).sum())}

    auc = float(roc_auc_score(labels, scores))

    if threshold is None:
        threshold = youden_threshold(labels, scores)
        threshold_mode = "automatic (Youden's J)"
    else:
        threshold = float(threshold)
        threshold_mode = "supplied (calibrated)"

    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    cm = confusion_at(labels, scores, threshold)
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)

    out = {
        "auc": auc,
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "false_alarm_rate": cm["fp"] / max(1, cm["fp"] + cm["tn"]),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "roc_curve": {"fpr": fpr_curve.tolist(), "tpr": tpr_curve.tolist()},
    }

    if attack_types is not None:
        attack_types = np.asarray(attack_types, dtype=object)
        per_attack = {}
        for name in sorted({a for a, l in zip(attack_types, labels) if l == 1}):
            mask = (attack_types == name) & (labels == 1)
            caught = int(pred[mask].sum())
            total = int(mask.sum())
            per_attack[name] = {
                "caught": caught,
                "missed": total - caught,
                "recall": caught / total if total else None,
            }
        out["per_attack_recall"] = per_attack

    return out


def compare_models(results_by_model, key="auc"):
    rows = []
    for name, res in results_by_model.items():
        if "error" in res:
            rows.append({"model": name, "error": res["error"]})
            continue
        rows.append({
            "model": name,
            "auc": res["auc"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "false_alarm_rate": res["false_alarm_rate"],
        })
    valid = [r for r in rows if "error" not in r]
    best = max(valid, key=lambda r: r[key]) if valid else None
    return {"rows": rows, "best_by_" + key: best["model"] if best else None}


def format_comparison(comparison):
    lines = []
    header = f"{'model':<34}{'AUC':<10}{'precision':<12}{'recall':<10}{'F1':<10}{'false alarms':<14}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in comparison["rows"]:
        if "error" in r:
            lines.append(f"{r['model']:<34}{r['error']}")
            continue
        lines.append(
            f"{r['model']:<34}{r['auc']:<10.4f}{r['precision']:<12.4f}"
            f"{r['recall']:<10.4f}{r['f1']:<10.4f}{r['false_alarm_rate']:<14.4f}"
        )
    return "\n".join(lines)


def format_per_attack(results_by_model):
    names = sorted({
        a for res in results_by_model.values()
        for a in (res.get("per_attack_recall") or {})
    })
    if not names:
        return "(no per-attack breakdown available)"

    models = list(results_by_model.keys())
    header = f"{'attack':<28}" + "".join(f"{m:<32}" for m in models)
    lines = [header, "-" * len(header)]

    for attack in names:
        row = f"{attack:<28}"
        for model in models:
            per_attack = results_by_model[model].get("per_attack_recall") or {}
            entry = per_attack.get(attack)
            if entry is None:
                cell = "-"
            else:
                total = entry["caught"] + entry["missed"]
                cell = f"{entry['caught']}/{total}"
            row += f"{cell:<32}"
        lines.append(row)

    return "\n".join(lines)