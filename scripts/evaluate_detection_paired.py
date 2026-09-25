import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
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
SPEED_BANDS = [(0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]

MEASURES = ["random", "heuristic", "single_step", "total", "speed_shortfall"]
MAD_SCALE = 1.4826
LABELS = {"random": "Random",
          "heuristic": "Speed-change heuristic",
          "single_step": "Single-step, 8-feature (as published)",
          "total": "Rollout, total divergence",
          "speed_shortfall": "Rollout, speed shortfall"}


def stable_seed(key):
    return int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16)


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


def heuristic(seq):
    sp = np.linalg.norm(seq[:, 2:4], axis=1)
    return float(np.abs(np.diff(sp)).max()) if len(sp) > 1 else 0.0


class Scorer:
    def __init__(self, ckpt, cfg, mean, std, device):
        self.device = device
        self.mn, self.sd = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.m = torch.as_tensor(mean, dtype=torch.float32)
        self.s = torch.as_tensor(std, dtype=torch.float32)
        self.net = PAMPOS(input_dim=cfg["model"]["input_dim"], d_model=cfg["model"]["d_model"],
                          nhead=cfg["model"]["nhead"], num_layers=cfg["model"]["num_layers"],
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
    def rollout(self, seed, actual):
        s = self._n(seed)
        im_n = dream_rollout(self.net, s, actual.shape[0])
        total = float(torch.abs(im_n - self._n(actual)).mean().item())
        im = im_n.squeeze(0).cpu().numpy() * self.sd + self.mn
        sf = float(np.mean(np.maximum(np.linalg.norm(im[:, 2:4], axis=1)
                                      - np.linalg.norm(actual[:, 2:4], axis=1), 0.0)))
        return total, sf

    def all_measures(self, seed, actual, key):
        full = np.concatenate([seed, actual], axis=0)
        total, sf = self.rollout(seed, actual)
        return {"single_step": self.single_step(full), "total": total,
                "speed_shortfall": sf, "heuristic": heuristic(full),
                "random": float(np.random.default_rng(stable_seed(key)).random())}


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="evaluate a seed-suffixed checkpoint instead of the canonical one")
    args = parser.parse_args()
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    if suffix:
        print(f"[setup] evaluating pampos_baseline{suffix}_best.pt")

    st = np.load(REPO_ROOT / "data" / "processed" / f"feature_stats{suffix}.npz")
    sc = Scorer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / f"pampos_baseline{suffix}_best.pt",
                cfg, st["mean"], st["std"], dev)

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=SEED_LEN + HORIZON)
    g = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, va = random_split(ds, [len(ds) - vs, vs], generator=g)
    sc.calibrate([ds.sequences[i][:SEED_LEN + HORIZON] for i in list(tr.indices)[:300]])

    traj = REPO_ROOT / "data" / "attack_trajectories"
    scen = sorted({f.name.split("__")[0] for f in traj.glob("*__attacked.json")})

    attacked, paired_benign, excluded = [], [], 0
    for s in scen:
        ld = find_label_dir(REPO_ROOT / cfg["data"]["raw_dir"], s)
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
                sa = mean_speed(ra, tid, t0, HORIZON)
                scl = mean_speed(rc, tid, t0, HORIZON)
                if math.isnan(sa) or math.isnan(scl) or (scl - sa) < MIN_EFFECT_MS:
                    excluded += 1
                    continue
                key = (s, attack, tid)
                ma = sc.all_measures(A[t0 - SEED_LEN:t0], A[t0:t0 + HORIZON], key)
                ma["speed_before"] = float(np.linalg.norm(A[t0 - 1, 2:4]))
                attacked.append(ma)
                paired_benign.append(sc.all_measures(C[t0 - SEED_LEN:t0],
                                                     C[t0:t0 + HORIZON], ("clean",) + key))

    native = []
    for i in list(va.indices)[:3000]:
        q = ds.sequences[i]
        native.append(sc.all_measures(q[:SEED_LEN], q[SEED_LEN:SEED_LEN + HORIZON], ("native", int(i))))

    print(f"[data] {len(attacked)} attacked, {len(paired_benign)} paired clean-replay benign")
    print(f"[data] {excluded} commanded agents excluded: the attack did not slow them")
    print(f"[data] {len(native)} native DeepAccident benign sequences\n")

    print("=" * 96)
    print("IS CLEAN REPLAY COMPARABLE TO NATIVE DEEPACCIDENT?")
    print("=" * 96)
    print("If replay itself raises scores, a native benign reference measures the")
    print("simulator rather than the attack. Comparing the two benign sets settles it.\n")
    print(f"{'measure':<40}{'native median':<18}{'replay median':<18}{'ratio'}")
    print("-" * 96)
    shift = {}
    for m in MEASURES:
        n = float(np.median([r[m] for r in native]))
        p = float(np.median([r[m] for r in paired_benign]))
        ratio = p / n if n > 1e-9 else float("inf")
        shift[m] = {"native": n, "replay": p, "ratio": ratio}
        print(f"{LABELS[m]:<40}{n:<18.4f}{p:<18.4f}{ratio:.2f}x")
    print("-" * 96)
    ss = shift["single_step"]["ratio"]
    print()
    if ss > 2.0:
        print(f"  Clean replay scores {ss:.1f}x higher than native data on the published")
        print(f"  measure, with no attack present. A native benign reference is therefore")
        print(f"  not a valid control: the paired clean replay is.")
    elif ss > 1.3:
        print(f"  Clean replay scores {ss:.1f}x higher than native data with no attack")
        print(f"  present. The effect is modest but the paired reference remains safer.")
    else:
        print(f"  Clean replay and native data score comparably ({ss:.2f}x), so the")
        print(f"  native reference is not confounded by the replay mechanism.")

    print()
    print("=" * 96)
    print("CHOOSING AN OPERATING POINT THE SAMPLE SUPPORTS")
    print("=" * 96)
    n = len(paired_benign)
    for q in (99, 95, 90):
        rank = q / 100 * (n - 1)
        print(f"  {q}th percentile of {n} sequences sits at rank "
              f"{rank:.1f}, i.e. the top {n - rank:.0f} observation(s)")
    print()
    print("  The 99th percentile is decided by one or two sequences and is not")
    print("  estimable here. We therefore report the operating point at the 95th")
    print("  percentile, a 5% false-alarm rate, which rank 78 of 83 supports.")

    for ref_name, ref in [("PAIRED CLEAN REPLAY", paired_benign),
                          ("NATIVE DEEPACCIDENT", native)]:
        print()
        print("=" * 96)
        print(f"DETECTION -- benign reference: {ref_name} ({len(ref)} sequences)")
        print("=" * 96)
        print(f"{'scoring procedure':<40}{'AUC':<10}"
              f"{'at 5% FA':<14}{'at 10% FA':<14}{'at 1% FA'}")
        print("-" * 96)
        for m in MEASURES:
            b = np.array([r[m] for r in ref])
            a = np.array([r[m] for r in attacked])
            labels = [0] * len(b) + [1] * len(a)
            thr95 = float(np.percentile(b, 95))
            res = detection_metrics(labels, b.tolist() + a.tolist(), threshold=thr95)
            counts = {}
            for q in (95, 90, 99):
                t = float(np.percentile(b, q))
                counts[q] = f"{int((a > t).sum())}/{len(a)}"
            print(f"{LABELS[m]:<40}{res['auc']:<10.4f}"
                  f"{counts[95]:<14}{counts[90]:<14}{counts[99]}")
        print("-" * 96)
        print("  AUC is threshold-free. The 1% column is shown for completeness but")
        print("  rests on the top one or two benign sequences of the reference set.")

    b = np.array([r["speed_shortfall"] for r in paired_benign])
    thr = float(np.percentile(b, 95))
    print()
    print("=" * 96)
    print("SPEED SHORTFALL BY THE AGENT'S SPEED BEFORE THE ATTACK (paired reference)")
    print("=" * 96)
    print(f"{'speed band':<18}{'n':<8}{'detected':<14}{'rate'}")
    print("-" * 96)
    bands = []
    for lo, hi in SPEED_BANDS:
        sel = [r for r in attacked if lo <= r["speed_before"] < hi]
        if not sel:
            continue
        hit = sum(1 for r in sel if r["speed_shortfall"] > thr)
        name = f"{lo:.0f}-{hi:.0f} m/s" if hi < 1e9 else f"{lo:.0f}+ m/s"
        bands.append({"band": name, "n": len(sel), "detected": hit})
        print(f"{name:<18}{len(sel):<8}{hit}/{len(sel):<9}{hit/len(sel)*100:.1f}%")
    print("-" * 96)

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"detection_paired_comparison{suffix}.json", "w") as f:
        json.dump({"n_attacked": len(attacked), "n_paired": len(paired_benign),
                   "n_native": len(native), "n_excluded": excluded,
                   "domain_shift": shift, "speed_bands": bands}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/detection_paired_comparison.json")


if __name__ == "__main__":
    main()