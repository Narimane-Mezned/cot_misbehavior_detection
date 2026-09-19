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
from src.data_pipeline.deepaccident_loader import (
    DeepAccidentBenignDataset, parse_label_file, get_frame_number, EGO_TRACK_ID,
)

ATTACKS = ["sensor_spoofing", "fake_emergency", "fake_safety",
           "traffic_light_tampering", "universal_perturbation", "sybil"]
SEED_LEN = 10
HORIZON = 3


def commanded_agents(run):
    rec = run.get("attack_record") or {}
    bh = (rec.get("metadata") or {}).get("enforced_behaviour") or {}
    return [str(t) for t in bh.get("track_ids", [])]


def find_label_dir(data_root, scenario):
    for type_dir in sorted(Path(data_root).glob("*_normal")):
        c = type_dir / "ego_vehicle" / "label" / scenario
        if c.is_dir():
            return c
    return None


def load_sensor_records(label_dir):
    out = []
    for f in sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name)):
        parsed = parse_label_file(f)
        ego = next((o for o in parsed["objects"] if o["track_id"] == EGO_TRACK_ID), None)
        out.append({"ego": ego, "objects": {str(o["track_id"]): o for o in parsed["objects"]}})
    return out


def agent_series_full(run, track_id, sensor_frames):
    dt = run["fixed_delta_seconds"]
    xs, ys, yaws, idxs = [], [], [], []
    for i, fr in enumerate(run["trajectory"]):
        a = fr["agents"].get(track_id)
        if a is None:
            continue
        idxs.append(i)
        xs.append(a["x"]); ys.append(a["y"]); yaws.append(math.radians(a["yaw_deg"]))
    if len(xs) < 2:
        return None
    vx, vy = [0.0], [0.0]
    for i in range(1, len(xs)):
        vx.append((xs[i] - xs[i - 1]) / dt)
        vy.append((ys[i] - ys[i - 1]) / dt)
    pc, cam, dist = [], [], []
    for k, i in enumerate(idxs):
        rec = sensor_frames[i] if i < len(sensor_frames) else None
        obj = rec["objects"].get(track_id) if rec else None
        ego = rec["ego"] if rec else None
        pc.append(float(obj["point_count"]) if obj else 0.0)
        cam.append(1.0 if (obj and obj["is_camera_visible"]) else 0.0)
        dist.append(math.dist((xs[k], ys[k]), (ego["x"], ego["y"])) if ego else 0.0)
    return np.stack([np.array(xs), np.array(ys), np.array(vx), np.array(vy),
                     np.array(yaws), np.array(pc), np.array(cam), np.array(dist)],
                    axis=1).astype(np.float32)


class Dreamer:
    def __init__(self, ckpt, cfg, mean, std, device):
        self.device = device
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.model = PAMPOS(
            input_dim=cfg["model"]["input_dim"], d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"], num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"]).to(device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

    def divergence(self, seed, actual):
        s = ((torch.from_numpy(seed).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        a = ((torch.from_numpy(actual).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        return float(torch.abs(dream_rollout(self.model, s, actual.shape[0]) - a).mean().item())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="data/attack_trajectories")
    ap.add_argument("--data_root", default="data/raw")
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    dreamer = Dreamer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
                      cfg, stats["mean"], stats["std"], device)

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / args.data_root,
                                   seq_len=SEED_LEN + HORIZON)
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    _, val_split = random_split(ds, [len(ds) - vs, vs], generator=gen)
    reference = np.array([dreamer.divergence(ds.sequences[i][:SEED_LEN],
                                             ds.sequences[i][SEED_LEN:SEED_LEN + HORIZON])
                          for i in val_split.indices])

    traj_dir = REPO_ROOT / args.traj_dir
    scenarios = sorted({f.name.split("__")[0] for f in traj_dir.glob("*__attacked.json")})

    attacked, benign, kinds = [], [], []
    for scenario in scenarios:
        ld = find_label_dir(REPO_ROOT / args.data_root, scenario)
        if ld is None:
            continue
        sensors = load_sensor_records(ld)
        fclean = traj_dir / f"{scenario}__clean.json"
        if not fclean.exists():
            continue
        run_clean = json.load(open(fclean))
        for attack in ATTACKS:
            fa = traj_dir / f"{scenario}__{attack}__attacked.json"
            if not fa.exists():
                continue
            run_a = json.load(open(fa))
            start = run_a["attack_start_frame"]
            for tid in commanded_agents(run_a):
                A = agent_series_full(run_a, tid, sensors)
                C = agent_series_full(run_clean, tid, sensors)
                if A is None or C is None:
                    continue
                if start < SEED_LEN or start + HORIZON > len(A) or start + HORIZON > len(C):
                    continue
                attacked.append(dreamer.divergence(A[start - SEED_LEN:start],
                                                   A[start:start + HORIZON]))
                benign.append(dreamer.divergence(C[start - SEED_LEN:start],
                                                 C[start:start + HORIZON]))
                kinds.append(attack)

    attacked, benign = np.array(attacked), np.array(benign)

    print("=" * 80)
    print("DIVERGENCE DISTRIBUTIONS")
    print("=" * 80)
    print(f"  benign, held-out DeepAccident (n={len(reference)})")
    print(f"     median {np.median(reference):.4f}   p99 {np.percentile(reference,99):.4f}   "
          f"max {reference.max():.4f}")
    print(f"  benign, clean replay in this experiment (n={len(benign)})")
    print(f"     median {np.median(benign):.4f}   min {benign.min():.4f}   max {benign.max():.4f}")
    print(f"  attacked (n={len(attacked)})")
    print(f"     median {np.median(attacked):.4f}   min {attacked.min():.4f}   max {attacked.max():.4f}")

    print()
    print("=" * 80)
    print("IS THE SEPARATION REAL, OR WITHIN ORDINARY BENIGN VARIATION?")
    print("=" * 80)
    ref_max = reference.max()
    above = (attacked > ref_max).sum()
    gap = attacked.min() - benign.max()
    print(f"  benign maximum over held-out data : {ref_max:.4f}")
    print(f"  attacked sequences above it       : {above}/{len(attacked)}")
    print(f"  gap between lowest attacked and highest benign in this experiment: {gap:+.4f}")
    print()
    if gap > 0 and above >= 0.8 * len(attacked):
        print("  The two classes do not overlap, and attacked divergences exceed anything")
        print("  seen in ordinary benign driving. The separation is driven by the attack.")
    elif gap > 0:
        print("  The classes do not overlap in this experiment, but some attacked")
        print("  divergences fall within the range of ordinary benign driving. The")
        print("  perfect AUC may not survive a larger or more varied sample.")
    else:
        print("  The classes overlap. The reported AUC is not supported by these scores.")

    print()
    print("=" * 80)
    print("PER-ATTACK DIVERGENCE")
    print("=" * 80)
    for a in ATTACKS:
        idx = [i for i, k in enumerate(kinds) if k == a]
        if not idx:
            continue
        va = attacked[idx]; vb = benign[idx]
        print(f"  {a:<26} attacked {va.mean():7.4f}   benign {vb.mean():7.4f}   "
              f"ratio {va.mean()/max(vb.mean(),1e-9):6.1f}x")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "divergence_distributions.json", "w") as f:
        json.dump({"reference_benign": {"median": float(np.median(reference)),
                                        "p99": float(np.percentile(reference, 99)),
                                        "max": float(reference.max()), "n": len(reference)},
                   "experiment_benign": benign.tolist(),
                   "attacked": attacked.tolist(),
                   "kinds": kinds,
                   "gap": float(gap)}, f, indent=2)
    print(f"\n[done] saved to outputs/results/divergence_distributions.json")


if __name__ == "__main__":
    main()