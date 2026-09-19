import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.dreaming import dream_rollout
from src.eval.metrics import detection_metrics, compare_models, format_comparison, format_per_attack

ATTACKS = ["sensor_spoofing", "fake_emergency", "fake_safety",
           "traffic_light_tampering", "universal_perturbation", "sybil"]
SEED_LEN = 10
HORIZONS = [3, 5, 8]


def commanded_agents(run):
    rec = run.get("attack_record") or {}
    bh = (rec.get("metadata") or {}).get("enforced_behaviour") or {}
    return [str(t) for t in bh.get("track_ids", [])]


def agent_series(run, track_id):
    dt = run["fixed_delta_seconds"]
    frames, xs, ys, yaws = [], [], [], []
    for fr in run["trajectory"]:
        a = fr["agents"].get(track_id)
        if a is None:
            continue
        frames.append(fr["frame_idx"])
        xs.append(a["x"]); ys.append(a["y"]); yaws.append(math.radians(a["yaw_deg"]))
    if len(xs) < 2:
        return None
    vx, vy = [0.0], [0.0]
    for i in range(1, len(xs)):
        vx.append((xs[i] - xs[i - 1]) / dt)
        vy.append((ys[i] - ys[i - 1]) / dt)
    return frames, np.stack([np.array(xs), np.array(ys), np.array(vx),
                             np.array(vy), np.array(yaws)], axis=1).astype(np.float32)


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
        imagined = dream_rollout(self.model, s, actual.shape[0])
        return float(torch.abs(imagined - a).mean().item())


def collect(traj_dir, scenarios, dreamer, horizon):
    scores, labels, kinds = [], [], []
    for scenario in scenarios:
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
                sa = agent_series(run_a, tid)
                sc = agent_series(run_clean, tid)
                if sa is None or sc is None:
                    continue
                fr_a, A = sa
                fr_c, C = sc
                if start < SEED_LEN or start + horizon > len(A) or start + horizon > len(C):
                    continue

                seed = A[start - SEED_LEN:start]
                scores.append(dreamer.divergence(seed, A[start:start + horizon]))
                labels.append(1); kinds.append(attack)

                seed_c = C[start - SEED_LEN:start]
                scores.append(dreamer.divergence(seed_c, C[start:start + horizon]))
                labels.append(0); kinds.append(None)
    return scores, labels, kinds


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="data/attack_trajectories")
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "pampos_ablated.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj_dir = REPO_ROOT / args.traj_dir
    scenarios = sorted({f.name.split("__")[0] for f in traj_dir.glob("*__attacked.json")})
    if not scenarios:
        print(f"[abort] no attacked runs in {traj_dir}")
        return

    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats_ablated.npz")
    dreamer = Dreamer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_ablated_best.pt",
                      cfg, stats["mean"], stats["std"], device)

    print(f"[setup] device {device}")
    print(f"[setup] {len(scenarios)} scenario(s)")
    print(f"[setup] seed length {SEED_LEN} frames, taken BEFORE the attack")
    print(f"[setup] the model imagines how the vehicle would have continued;")
    print(f"[setup] the score is how far reality diverged from that imagination\n")

    results = {}
    for h in HORIZONS:
        scores, labels, kinds = collect(traj_dir, scenarios, dreamer, h)
        if not scores or sum(labels) == 0:
            print(f"[warn] horizon {h}: no usable pairs")
            continue
        results[f"Dreaming, horizon {h}"] = detection_metrics(
            labels, scores, attack_types=kinds)
        print(f"[data] horizon {h}: {sum(labels)} attacked, {len(labels)-sum(labels)} benign")

    if not results:
        print("[abort] nothing to report")
        return

    print()
    print("=" * 88)
    print("DETECTION BY AUTOREGRESSIVE ROLLOUT")
    print("=" * 88)
    print(format_comparison(compare_models(results)))
    print()
    print("=" * 88)
    print("PER-ATTACK RECALL")
    print("=" * 88)
    print(format_per_attack(results))

    best = max(results.items(), key=lambda kv: kv[1].get("auc", 0))
    print()
    print("=" * 88)
    print("COMPARISON WITH SINGLE-STEP SCORING")
    print("=" * 88)
    print(f"  single-step, same model and data : AUC 0.0541")
    print(f"  {best[0]:<33}: AUC {best[1]['auc']:.4f}")
    print()
    if best[1]["auc"] > 0.75:
        print("  Anchoring the prediction before the attack exposes the behavioural change")
        print("  that single-step scoring cannot see. The model expects the vehicle to")
        print("  continue as it was; the attack makes it stop, and the imagined and actual")
        print("  trajectories separate.")
    elif best[1]["auc"] > 0.6:
        print("  Rollout improves on single-step scoring but does not fully separate the")
        print("  two conditions.")
    else:
        print("  Rollout does not resolve the limitation either.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in results.items():
        v = dict(v); v.pop("roc_curve", None)
        clean[k] = v
    with open(out / "dreaming_detection.json", "w") as f:
        json.dump({"scenarios": scenarios, "seed_len": SEED_LEN,
                   "horizons": HORIZONS, "results": clean}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/dreaming_detection.json")


if __name__ == "__main__":
    main()