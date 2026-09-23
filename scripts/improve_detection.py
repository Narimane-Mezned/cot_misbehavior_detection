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
from src.eval.metrics import detection_metrics, compare_models, format_comparison

ATTACKS = ["sensor_spoofing", "fake_emergency", "fake_safety",
           "traffic_light_tampering", "universal_perturbation", "sybil"]

UNTRACKED_TRACK_ID = "-1"
MIN_EFFECT_MS = 1.0
SPEED_BANDS = [(0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]
SEED_LEN = 10
HORIZON = 3


def commanded_agents(run):
    rec = run.get("attack_record") or {}
    bh = (rec.get("metadata") or {}).get("enforced_behaviour") or {}
    return [str(t) for t in bh.get("track_ids", []) if str(t) != UNTRACKED_TRACK_ID]


def mean_speed(run, track_id, from_frame, horizon):
    dt = run["fixed_delta_seconds"]
    v, taken = [], 0
    prev = None
    for fr in run["trajectory"]:
        cur = fr["agents"].get(track_id)
        if prev is not None and cur is not None and fr["frame_idx"] >= from_frame:
            v.append(math.dist((cur["x"], cur["y"]), (prev["x"], prev["y"])) / dt)
            taken += 1
            if taken >= horizon:
                break
        prev = cur
    return float(np.mean(v)) if v else float("nan")


def attack_took_effect(run_attacked, run_clean, track_id, start, horizon):
    a = mean_speed(run_attacked, track_id, start, horizon)
    c = mean_speed(run_clean, track_id, start, horizon)
    if math.isnan(a) or math.isnan(c):
        return False, a, c
    return (c - a) >= MIN_EFFECT_MS, a, c


def find_label_dir(data_root, scenario):
    for d in sorted(Path(data_root).glob("*_normal")):
        c = d / "ego_vehicle" / "label" / scenario
        if c.is_dir():
            return c
    return None


def load_sensors(label_dir):
    out = []
    for f in sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name)):
        parsed = parse_label_file(f)
        ego = next((o for o in parsed["objects"] if o["track_id"] == EGO_TRACK_ID), None)
        out.append({"ego": ego, "objects": {str(o["track_id"]): o for o in parsed["objects"]}})
    return out


def series(run, tid, sensors):
    dt = run["fixed_delta_seconds"]
    xs, ys, yaws, idxs = [], [], [], []
    for i, fr in enumerate(run["trajectory"]):
        a = fr["agents"].get(tid)
        if a is None:
            continue
        idxs.append(i); xs.append(a["x"]); ys.append(a["y"]); yaws.append(math.radians(a["yaw_deg"]))
    if len(xs) < 2:
        return None
    vx, vy = [0.0], [0.0]
    for i in range(1, len(xs)):
        vx.append((xs[i] - xs[i-1]) / dt); vy.append((ys[i] - ys[i-1]) / dt)
    pc, cam, dist = [], [], []
    for k, i in enumerate(idxs):
        rec = sensors[i] if i < len(sensors) else None
        o = rec["objects"].get(tid) if rec else None
        e = rec["ego"] if rec else None
        pc.append(float(o["point_count"]) if o else 0.0)
        cam.append(1.0 if (o and o["is_camera_visible"]) else 0.0)
        dist.append(math.dist((xs[k], ys[k]), (e["x"], e["y"])) if e else 0.0)
    return np.stack([np.array(xs), np.array(ys), np.array(vx), np.array(vy),
                     np.array(yaws), np.array(pc), np.array(cam), np.array(dist)],
                    axis=1).astype(np.float32)


class Scorer:
    def __init__(self, ckpt, cfg, mean, std, device):
        self.device = device
        self.mean_np = np.asarray(mean, dtype=np.float32)
        self.std_np = np.asarray(std, dtype=np.float32)
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

    def measures(self, seed, actual):
        s = ((torch.from_numpy(seed).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        a_n = ((torch.from_numpy(actual).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        imagined_n = dream_rollout(self.model, s, actual.shape[0])

        total = float(torch.abs(imagined_n - a_n).mean().item())

        imagined = (imagined_n.squeeze(0).cpu().numpy() * self.std_np) + self.mean_np
        actual_r = actual

        heading = math.atan2(seed[-1, 3], seed[-1, 2]) if abs(seed[-1, 2]) + abs(seed[-1, 3]) > 1e-6 \
            else float(seed[-1, 4])
        ux, uy = math.cos(heading), math.sin(heading)

        longitudinal, lateral = [], []
        for t in range(actual_r.shape[0]):
            dx = imagined[t, 0] - actual_r[t, 0]
            dy = imagined[t, 1] - actual_r[t, 1]
            longitudinal.append(dx * ux + dy * uy)
            lateral.append(abs(-dx * uy + dy * ux))

        speed_pred = np.linalg.norm(imagined[:, 2:4], axis=1)
        speed_true = np.linalg.norm(actual_r[:, 2:4], axis=1)
        shortfall = float(np.mean(np.maximum(speed_pred - speed_true, 0.0)))

        return {
            "total": total,
            "longitudinal": float(np.mean(longitudinal)),
            "lateral": float(np.mean(lateral)),
            "speed_shortfall": shortfall,
        }


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
    sc = Scorer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
                cfg, stats["mean"], stats["std"], device)

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / args.data_root,
                                   seq_len=SEED_LEN + HORIZON)
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    _, val_split = random_split(ds, [len(ds) - vs, vs], generator=gen)

    print(f"[setup] building benign reference from {len(val_split)} held-out sequences")
    ref = [sc.measures(ds.sequences[i][:SEED_LEN], ds.sequences[i][SEED_LEN:SEED_LEN + HORIZON])
           for i in val_split.indices]
    ineffective = []

    traj_dir = REPO_ROOT / args.traj_dir
    scenarios = sorted({f.name.split("__")[0] for f in traj_dir.glob("*__attacked.json")})
    attacked = []
    for scenario in scenarios:
        ld = find_label_dir(REPO_ROOT / args.data_root, scenario)
        fclean = traj_dir / f"{scenario}__clean.json"
        if ld is None or not fclean.exists():
            continue
        sensors = load_sensors(ld)
        run_clean = json.load(open(fclean))
        for attack in ATTACKS:
            fa = traj_dir / f"{scenario}__{attack}__attacked.json"
            if not fa.exists():
                continue
            run_a = json.load(open(fa))
            start = run_a["attack_start_frame"]
            for tid in commanded_agents(run_a):
                A = series(run_a, tid, sensors)
                if A is None or start < SEED_LEN or start + HORIZON > len(A):
                    continue
                took, sp_a, sp_c = attack_took_effect(run_a, run_clean, tid, start, HORIZON)
                m = sc.measures(A[start - SEED_LEN:start], A[start:start + HORIZON])
                if took:
                    m = dict(m)
                    m["speed_before"] = float(np.linalg.norm(A[start - 1, 2:4]))
                    attacked.append(m)
                else:
                    ineffective.append({"scenario": scenario, "attack": attack, "agent": tid,
                                        "attacked_speed": sp_a, "clean_speed": sp_c})

    print(f"[setup] {len(attacked)} attacked sequences where the attack changed behaviour")
    print(f"[setup] {len(ineffective)} commanded agents where it did not, excluded below")
    if ineffective:
        print(f"[setup] exclusion criterion fixed in advance: the attacked agent must be at")
        print(f"[setup] least {MIN_EFFECT_MS} m/s slower than the same agent in the clean replay,")
        print(f"[setup] over the same frames. An attack that does not alter behaviour cannot")
        print(f"[setup] be detected by any behavioural detector.")
        by_reason = {}
        for r in ineffective:
            key = "already stationary" if r["clean_speed"] < 1.0 else "attack had no effect"
            by_reason.setdefault(key, []).append(r)
        for k, v in by_reason.items():
            print(f"[setup]    {k}: {len(v)}")
    print()

    measures = ["total", "longitudinal", "lateral", "speed_shortfall"]
    labels = [0] * len(ref) + [1] * len(attacked)
    results = {}

    print("=" * 84)
    print("EACH MEASURE AGAINST THE HELD-OUT BENIGN REFERENCE")
    print("=" * 84)
    print(f"{'measure':<20}{'benign p99':<14}{'attacked med':<16}{'caught at 1% FA':<18}{'AUC'}")
    print("-" * 84)

    for m in measures:
        b = np.array([r[m] for r in ref])
        a = np.array([r[m] for r in attacked])
        thr = float(np.percentile(b, 99))
        caught = int((a > thr).sum())
        results[m] = detection_metrics(labels, b.tolist() + a.tolist(), threshold=thr)
        auc = results[m].get("auc", float("nan"))
        print(f"{m:<20}{thr:<14.4f}{np.median(a):<16.4f}"
              f"{f'{caught}/{len(a)}':<18}{auc:.4f}")

    print()
    print("=" * 84)
    print("INTERPRETATION")
    print("=" * 84)
    best = max(measures, key=lambda m: results[m].get("auc", 0))
    base_auc = results["total"].get("auc", 0)
    best_auc = results[best].get("auc", 0)
    print(f"  current measure (total divergence) : AUC {base_auc:.4f}")
    print(f"  best measure ({best}){' ' * max(0, 21 - len(best))}: AUC {best_auc:.4f}")
    print()
    if best != "total" and best_auc > base_auc + 0.05:
        print(f"  Scoring by {best} improves on total divergence. An attacked vehicle")
        print(f"  falls behind its predicted position in a consistent direction, whereas")
        print(f"  normal unpredictability is not directional.")
    else:
        print("  No refinement improves meaningfully on total divergence.")

    if attacked and "speed_before" in attacked[0]:
        b = np.array([r["speed_shortfall"] for r in ref])
        thr = float(np.percentile(b, 99))
        print()
        print("=" * 84)
        print("DETECTION BY THE AGENT'S SPEED BEFORE THE ATTACK")
        print("=" * 84)
        print(f"{'speed band':<18}{'n':<8}{'detected':<14}{'rate'}")
        print("-" * 84)
        bands = []
        for lo, hi in SPEED_BANDS:
            sel = [r for r in attacked if lo <= r["speed_before"] < hi]
            if not sel:
                continue
            hit = sum(1 for r in sel if r["speed_shortfall"] > thr)
            name = f"{lo:.0f}-{hi:.0f} m/s" if hi < 1e9 else f"{lo:.0f}+ m/s"
            bands.append({"band": name, "n": len(sel), "detected": hit})
            print(f"{name:<18}{len(sel):<8}{hit}/{len(sel):<9}{hit/len(sel)*100:.1f}%")
        print("-" * 84)
        print(f"  Threshold is {thr:.3f}. A vehicle travelling at v m/s cannot produce a")
        print(f"  shortfall larger than v, so below that speed the signal is bounded under")
        print(f"  the threshold by physics rather than by any property of the detector.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in results.items():
        v = dict(v); v.pop("roc_curve", None)
        clean[k] = v
    with open(out / "divergence_measures.json", "w") as f:
        json.dump({"n_reference": len(ref), "n_attacked": len(attacked),
                   "n_ineffective": len(ineffective), "ineffective": ineffective,
                   "min_effect_ms": MIN_EFFECT_MS,
                   "results": clean}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/divergence_measures.json")


if __name__ == "__main__":
    main()