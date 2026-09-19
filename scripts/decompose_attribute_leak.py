import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset

CAMERA_IDX = 6
QUANTISATION_STEPS = [None, 0.1, 0.5, 1.0, 2.0]


class Scorer:
    def __init__(self, ckpt, cfg, mean, std, device, keep=None):
        self.device = device
        self.keep = keep
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.model = PAMPOS(
            input_dim=cfg["model"]["input_dim"] if keep is None else len(keep),
            d_model=cfg["model"]["d_model"], nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"]).to(device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        self.feature_mae = None

    @torch.no_grad()
    def _errors(self, w):
        x = ((torch.from_numpy(w).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        e = per_feature_errors(self.model(x[:, :-1, :]), x[:, 1:, :])
        return normalize_errors(e, self.feature_mae) if self.feature_mae is not None else e

    def calibrate(self, ws):
        self.feature_mae = None
        self.feature_mae = torch.cat([self._errors(w) for w in ws], dim=0).mean(dim=(0, 1))

    def score(self, w, quantise=None):
        s = topk_anomaly_score(self._errors(w), k=3).mean().item()
        return s if quantise is None else round(s / quantise) * quantise


def camera_label(window):
    return float(window[-1, CAMERA_IDX] > 0.5)


def attack_accuracy(X, y, seed=0):
    n = len(y)
    split = int(n * 0.75)
    clf = RandomForestClassifier(n_estimators=150, random_state=seed)
    clf.fit(X[:split], y[:split])
    return float(clf.score(X[split:], y[split:]))


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = cfg["training"]["seq_len"]

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"], seq_len=seq_len)
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, va = random_split(ds, [len(ds) - vs, vs], generator=gen)

    rng = np.random.default_rng(0)
    idx = rng.choice(va.indices, size=min(1500, len(va.indices)), replace=False)
    windows = [ds.sequences[i] for i in idx]
    labels = np.array([camera_label(w) for w in windows])

    prior = max(labels.mean(), 1 - labels.mean())
    print(f"[setup] {len(windows)} held-out windows")
    print(f"[setup] class prior for is_camera_visible: {prior:.3f}\n")

    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    scorer = Scorer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
                    cfg, stats["mean"], stats["std"], device)
    calib = [ds.sequences[i] for i in list(tr.indices)[:300]]
    scorer.calibrate(calib)

    print("=" * 80)
    print("WHERE DOES THE LEAK ACTUALLY COME FROM?")
    print("=" * 80)

    other = [i for i in range(8) if i != CAMERA_IDX]
    X_feat = np.array([w[:, other].flatten() for w in windows])
    acc_feat = attack_accuracy(X_feat, labels)
    print(f"  A. other seven features alone, model never queried : {acc_feat:.3f}")
    print(f"     the adversary already holds these, so this is not model leakage")

    scores = np.array([scorer.score(w) for w in windows]).reshape(-1, 1)
    acc_score = attack_accuracy(scores, labels)
    print(f"  B. the model's score alone                          : {acc_score:.3f}")

    X_both = np.hstack([X_feat, scores])
    acc_both = attack_accuracy(X_both, labels)
    print(f"  C. features plus score (the full attack)            : {acc_both:.3f}")

    print()
    print(f"  class prior                                         : {prior:.3f}")
    print(f"  attributable to natural correlation  (A - prior)    : {acc_feat - prior:+.3f}")
    print(f"  attributable to the model            (C - A)        : {acc_both - acc_feat:+.3f}")

    print()
    print("=" * 80)
    print("DOES QUANTISING THE RELEASED SCORE REDUCE THE MODEL'S CONTRIBUTION?")
    print("=" * 80)
    print(f"{'quantisation':<18}{'attack acc':<14}{'model share':<16}{'flagged windows'}")
    print("-" * 80)

    threshold = None
    calib_path = REPO_ROOT / "data" / "processed" / "calibration.json"
    if calib_path.exists():
        threshold = json.load(open(calib_path)).get("threshold")

    rows = []
    for q in QUANTISATION_STEPS:
        sc = np.array([scorer.score(w, quantise=q) for w in windows]).reshape(-1, 1)
        acc = attack_accuracy(np.hstack([X_feat, sc]), labels)
        flagged = int((sc.flatten() > threshold).sum()) if threshold else -1
        label = "none" if q is None else f"{q}"
        rows.append({"quantisation": q, "attack_accuracy": acc,
                     "model_share": acc - acc_feat, "flagged": flagged})
        print(f"{label:<18}{acc:<14.3f}{acc - acc_feat:+.3f}{'':<11}{flagged}")

    print("-" * 80)
    base_flagged = rows[0]["flagged"]
    best = min(rows, key=lambda r: abs(r["model_share"]))
    print()
    print("=" * 80)
    print("READING")
    print("=" * 80)
    print(f"  Removing the feature from the model does not remove the adversary's ability")
    print(f"  to infer it, because the remaining features carry it naturally ({acc_feat:.3f}).")
    print(f"  What a mitigation can remove is the model's additional contribution, which is")
    print(f"  {acc_both - acc_feat:+.3f} here.")
    if abs(best["model_share"]) < abs(acc_both - acc_feat) - 0.01:
        q = best["quantisation"]
        print(f"  Quantising the released score to {q} reduces that contribution to "
              f"{best['model_share']:+.3f},")
        if base_flagged > 0 and best["flagged"] >= 0:
            delta = abs(best["flagged"] - base_flagged) / base_flagged * 100
            print(f"  changing the flagged-window count by {delta:.1f}%.")
    else:
        print(f"  Quantisation does not meaningfully reduce it.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "attribute_leak_decomposition.json", "w") as f:
        json.dump({"n_windows": len(windows), "class_prior": prior,
                   "features_only": acc_feat, "score_only": acc_score,
                   "features_plus_score": acc_both,
                   "natural_correlation": acc_feat - prior,
                   "model_contribution": acc_both - acc_feat,
                   "quantisation": rows}, f, indent=2)
    print(f"\n[done] saved to outputs/results/attribute_leak_decomposition.json")


if __name__ == "__main__":
    main()