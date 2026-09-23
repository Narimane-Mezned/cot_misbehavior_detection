import json
import math
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
from src.model.dreaming import dream_rollout
from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
from src.data_pipeline.deepaccident_loader import (
    DeepAccidentBenignDataset, parse_label_file, get_frame_number, EGO_TRACK_ID,
)
from src.eval.metrics import detection_metrics

ATTACKS = ["sensor_spoofing", "fake_emergency", "fake_safety",
           "traffic_light_tampering", "universal_perturbation", "sybil"]
SEED_LEN = 10
HORIZON = 3
MIN_EFFECT_MS = 1.0
CAMERA_IDX = 6
KINEMATIC = [0, 1, 2, 3, 4]


def commanded(run):
    bh = ((run.get("attack_record") or {}).get("metadata") or {}).get("enforced_behaviour") or {}
    return [str(t) for t in bh.get("track_ids", []) if str(t) != "-1"]


def find_label_dir(root, scenario):
    for d in sorted(Path(root).glob("*_normal")):
        c = d / "ego_vehicle" / "label" / scenario
        if c.is_dir():
            return c
    return None


def sensors(label_dir):
    out = []
    for f in sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name)):
        p = parse_label_file(f)
        ego = next((o for o in p["objects"] if o["track_id"] == EGO_TRACK_ID), None)
        out.append({"ego": ego, "objects": {str(o["track_id"]): o for o in p["objects"]}})
    return out


def series(run, tid, sens):
    dt = run["fixed_delta_seconds"]
    xs, ys, yaws, idx = [], [], [], []
    for i, fr in enumerate(run["trajectory"]):
        a = fr["agents"].get(tid)
        if a is None:
            continue
        idx.append(i); xs.append(a["x"]); ys.append(a["y"]); yaws.append(math.radians(a["yaw_deg"]))
    if len(xs) < 2:
        return None
    vx, vy = [0.0], [0.0]
    for i in range(1, len(xs)):
        vx.append((xs[i] - xs[i - 1]) / dt); vy.append((ys[i] - ys[i - 1]) / dt)
    pc, cam, dist = [], [], []
    for k, i in enumerate(idx):
        r = sens[i] if i < len(sens) else None
        o = r["objects"].get(tid) if r else None
        e = r["ego"] if r else None
        pc.append(float(o["point_count"]) if o else 0.0)
        cam.append(1.0 if (o and o["is_camera_visible"]) else 0.0)
        dist.append(math.dist((xs[k], ys[k]), (e["x"], e["y"])) if e else 0.0)
    return np.stack([np.array(xs), np.array(ys), np.array(vx), np.array(vy), np.array(yaws),
                     np.array(pc), np.array(cam), np.array(dist)], axis=1).astype(np.float32)


def mean_speed(run, tid, frm, horizon):
    dt = run["fixed_delta_seconds"]
    v, prev, taken = [], None, 0
    for fr in run["trajectory"]:
        cur = fr["agents"].get(tid)
        if prev is not None and cur is not None and fr["frame_idx"] >= frm:
            v.append(math.dist((cur["x"], cur["y"]), (prev["x"], prev["y"])) / dt)
            taken += 1
            if taken >= horizon:
                break
        prev = cur
    return float(np.mean(v)) if v else float("nan")


class Scorer:
    def __init__(self, ckpt, cfg, mean, std, device, input_dim=None):
        self.device = device
        self.mn, self.sd = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.m = torch.as_tensor(mean, dtype=torch.float32)
        self.s = torch.as_tensor(std, dtype=torch.float32)
        self.net = PAMPOS(input_dim=input_dim or cfg["model"]["input_dim"],
                          d_model=cfg["model"]["d_model"], nhead=cfg["model"]["nhead"],
                          num_layers=cfg["model"]["num_layers"],
                          dim_feedforward=cfg["model"]["dim_feedforward"],
                          dropout=cfg["model"]["dropout"]).to(device)
        self.net.load_state_dict(torch.load(ckpt, map_location=device,
                                            weights_only=False)["model_state_dict"])
        self.net.eval()
        self.feature_mae = None

    def _n(self, w):
        return ((torch.from_numpy(w).float() - self.m) / self.s).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def calibrate(self, windows):
        self.feature_mae = None
        e = [per_feature_errors(self.net(self._n(w)[:, :-1, :]), self._n(w)[:, 1:, :])
             for w in windows]
        self.feature_mae = torch.cat(e, dim=0).mean(dim=(0, 1))

    @torch.no_grad()
    def single_step(self, w):
        x = self._n(w)
        e = per_feature_errors(self.net(x[:, :-1, :]), x[:, 1:, :])
        if self.feature_mae is not None:
            e = normalize_errors(e, self.feature_mae)
        return float(topk_anomaly_score(e, k=3).mean().item())

    @torch.no_grad()
    def shortfall(self, seed, actual):
        im = dream_rollout(self.net, self._n(seed), actual.shape[0])
        im = im.squeeze(0).cpu().numpy() * self.sd + self.mn
        return float(np.mean(np.maximum(np.linalg.norm(im[:, 2:4], axis=1)
                                        - np.linalg.norm(actual[:, 2:4], axis=1), 0.0)))


def load_pairs(cfg, sc, traj, root):
    pairs = []
    for s in sorted({f.name.split("__")[0] for f in traj.glob("*__attacked.json")}):
        ld = find_label_dir(root, s)
        fc = traj / f"{s}__clean.json"
        if ld is None or not fc.exists():
            continue
        sens = sensors(ld)
        rc = json.load(open(fc))
        for attack in ATTACKS:
            fa = traj / f"{s}__{attack}__attacked.json"
            if not fa.exists():
                continue
            ra = json.load(open(fa))
            t0 = ra["attack_start_frame"]
            for tid in commanded(ra):
                A, C = series(ra, tid, sens), series(rc, tid, sens)
                if A is None or C is None or t0 < SEED_LEN or t0 + HORIZON > min(len(A), len(C)):
                    continue
                if (mean_speed(rc, tid, t0, HORIZON) - mean_speed(ra, tid, t0, HORIZON)) < MIN_EFFECT_MS:
                    continue
                pairs.append({"scenario": s, "attack": attack, "tid": tid,
                              "A": A, "C": C, "t0": t0})
    return pairs


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckdir = REPO_ROOT / cfg["paths"]["checkpoint_dir"]
    proc = REPO_ROOT / "data" / "processed"
    st = np.load(proc / "feature_stats.npz")
    sc = Scorer(ckdir / "pampos_baseline_best.pt", cfg, st["mean"], st["std"], dev)

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=SEED_LEN + HORIZON)
    g = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, va = random_split(ds, [len(ds) - vs, vs], generator=g)
    sc.calibrate([ds.sequences[i][:SEED_LEN + HORIZON] for i in list(tr.indices)[:300]])

    pairs = load_pairs(cfg, sc, REPO_ROOT / "data" / "attack_trajectories",
                       REPO_ROOT / cfg["data"]["raw_dir"])
    print(f"[data] {len(pairs)} attacked/benign pairs\n")

    print("=" * 96)
    print("ISSUE 1 -- DOES SINGLE-STEP SCORE ATTACKED WINDOWS HIGHER OR LOWER?")
    print("=" * 96)
    print("Two earlier runs disagreed. They differed in model AND scoring unit at")
    print("once. Here the 8-feature model is scored both ways on the same pairs.\n")

    seq_a = [sc.single_step(p["A"][p["t0"] - SEED_LEN:p["t0"] + HORIZON]) for p in pairs]
    seq_b = [sc.single_step(p["C"][p["t0"] - SEED_LEN:p["t0"] + HORIZON]) for p in pairs]

    win_a, win_b = [], []
    for p in pairs:
        A, C, t0 = p["A"], p["C"], p["t0"]
        end = min(len(A), len(C))
        for i in range(t0, end - SEED_LEN + 1):
            win_a.append(sc.single_step(A[i:i + SEED_LEN]))
            win_b.append(sc.single_step(C[i:i + SEED_LEN]))

    print(f"{'granularity':<34}{'n':<10}{'benign med':<14}{'attacked med':<16}{'AUC'}")
    print("-" * 96)
    rows = {}
    for name, a, b in [("sequence (seed + horizon)", seq_a, seq_b),
                       ("window (10 frames, post-attack)", win_a, win_b)]:
        lab = [0] * len(b) + [1] * len(a)
        auc = detection_metrics(lab, list(b) + list(a)).get("auc", float("nan"))
        rows[name] = {"n_attacked": len(a), "auc": auc,
                      "benign_median": float(np.median(b)),
                      "attacked_median": float(np.median(a))}
        print(f"{name:<34}{len(a):<10}{np.median(b):<14.4f}{np.median(a):<16.4f}{auc:.4f}")
    print("-" * 96)
    print()
    sq, wn = rows["sequence (seed + horizon)"]["auc"], rows["window (10 frames, post-attack)"]["auc"]
    if wn < 0.5 < sq:
        print("  RESOLVED: the direction depends on the scoring unit, not the model.")
        print("  Post-attack windows score LOWER (anti-correlated) because a stopped")
        print("  vehicle is easy to predict. A window spanning the transition scores")
        print("  HIGHER because the deceleration itself is unpredictable.")
        print("  The anti-correlation claim holds for post-attack windows only, and")
        print("  must be stated that way.")
    elif wn > 0.5 and sq > 0.5:
        print("  The 8-feature model scores attacked higher at BOTH granularities.")
        print("  The anti-correlation observed with the 5-feature model does not")
        print("  reproduce here. The claim must be dropped or restricted to that model.")
    else:
        print("  Both below 0.5: attacked score lower at both granularities.")
        print("  The anti-correlation claim holds for the 8-feature model.")

    print()
    print("=" * 96)
    print("ISSUE 2 -- ATTRIBUTE INFERENCE OVER REPEATED SPLITS")
    print("=" * 96)

    idxs = list(va.indices)
    feats, labels = [], []
    for i in idxs[:1500]:
        w = ds.sequences[i]
        feats.append(w[:, [c for c in range(8) if c != CAMERA_IDX]].flatten())
        labels.append(float(w[-1, CAMERA_IDX] > 0.5))
    feats, labels = np.array(feats), np.array(labels)
    scores = np.array([[sc.single_step(ds.sequences[i][:SEED_LEN + HORIZON])] for i in idxs[:1500]])

    prior = max(labels.mean(), 1 - labels.mean())
    A_acc, C_acc = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(labels))
        cut = int(len(labels) * 0.75)
        trn, tst = perm[:cut], perm[cut:]
        for X, bucket in [(feats, A_acc), (np.hstack([feats, scores]), C_acc)]:
            clf = RandomForestClassifier(n_estimators=150, random_state=seed)
            clf.fit(X[trn], labels[trn])
            bucket.append(float(clf.score(X[tst], labels[tst])))
    A_acc, C_acc = np.array(A_acc), np.array(C_acc)
    contrib = C_acc - A_acc

    print(f"  class prior                          {prior:.3f}")
    print(f"  A. features only, no model query     {A_acc.mean():.3f} +/- {A_acc.std():.3f}")
    print(f"  C. features + score                  {C_acc.mean():.3f} +/- {C_acc.std():.3f}")
    print(f"  model contribution (C - A)          {contrib.mean():+.3f} +/- {contrib.std():.3f}")
    print(f"  over 5 independent splits")
    print()
    if abs(contrib.mean()) < 2 * contrib.std():
        print("  The contribution is within noise of zero: the detector discloses no")
        print("  additional information about camera visibility.")
    else:
        print("  The contribution is distinguishable from zero and must be reported")
        print("  as a residual leak, not as an absence of leakage.")

    print()
    print("=" * 96)
    print("ISSUE 3 -- WHY DOES REPLAY SCORE LOWER ON SPEED SHORTFALL?")
    print("=" * 96)
    nat_sf, nat_sp = [], []
    for i in idxs[:1500]:
        q = ds.sequences[i]
        nat_sf.append(sc.shortfall(q[:SEED_LEN], q[SEED_LEN:SEED_LEN + HORIZON]))
        nat_sp.append(float(np.linalg.norm(q[SEED_LEN - 1, 2:4])))
    rep_sf = [sc.shortfall(p["C"][p["t0"] - SEED_LEN:p["t0"]],
                           p["C"][p["t0"]:p["t0"] + HORIZON]) for p in pairs]
    rep_sp = [float(np.linalg.norm(p["C"][p["t0"] - 1, 2:4])) for p in pairs]

    print(f"{'set':<28}{'median speed':<16}{'median shortfall':<20}{'frac decelerating'}")
    print("-" * 96)
    for name, sf, sp in [("native DeepAccident", nat_sf, nat_sp),
                         ("clean replay (paired)", rep_sf, rep_sp)]:
        frac = float(np.mean([s > 0.01 for s in sf]))
        print(f"{name:<28}{np.median(sp):<16.2f}{np.median(sf):<20.4f}{frac:.2f}")
    print("-" * 96)
    print()
    print("  Shortfall is only nonzero when a vehicle is slower than predicted.")
    print("  If replayed benign agents decelerate less often than native ones, their")
    print("  shortfall is lower, which explains the 0.13x ratio without invoking")
    print("  replay jitter.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "audit_resolutions.json", "w") as f:
        json.dump({"single_step_direction": rows,
                   "attribute_inference": {"prior": prior,
                                           "features_only_mean": float(A_acc.mean()),
                                           "features_only_std": float(A_acc.std()),
                                           "full_mean": float(C_acc.mean()),
                                           "full_std": float(C_acc.std()),
                                           "contribution_mean": float(contrib.mean()),
                                           "contribution_std": float(contrib.std()),
                                           "n_splits": 5},
                   "shortfall_replay": {
                       "native_median": float(np.median(nat_sf)),
                       "replay_median": float(np.median(rep_sf)),
                       "native_frac_decel": float(np.mean([s > 0.01 for s in nat_sf])),
                       "replay_frac_decel": float(np.mean([s > 0.01 for s in rep_sf]))}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/audit_resolutions.json")


if __name__ == "__main__":
    main()