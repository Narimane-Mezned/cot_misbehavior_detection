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
    build_agent_sequences, list_scenarios, parse_meta,
)
from src.eval.metrics import detection_metrics, compare_models, format_comparison, format_per_attack

SEED_LEN = 10
HORIZON = 3
DT = 0.1
MIN_SPEED_MS = 3.0

ATTACK_TARGET_SPEED = {
    "sensor_spoofing": 0.0,
    "traffic_light_tampering": 0.0,
    "universal_perturbation": 0.0,
    "fake_emergency": 1.0,
    "fake_safety": 0.5,
    "sybil": 0.5,
}


def inject_speed_command(window, target_speed, from_step):
    out = window.copy()
    for t in range(from_step, len(out)):
        vx, vy = out[t - 1, 2], out[t - 1, 3]
        speed = math.hypot(vx, vy)
        if speed < 1e-6:
            ux, uy = math.cos(out[t - 1, 4]), math.sin(out[t - 1, 4])
        else:
            ux, uy = vx / speed, vy / speed
        out[t, 2] = ux * target_speed
        out[t, 3] = uy * target_speed
        out[t, 0] = out[t - 1, 0] + out[t, 2] * DT
        out[t, 1] = out[t - 1, 1] + out[t, 3] * DT
    return out


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

    def shortfall(self, seed, actual):
        s = ((torch.from_numpy(seed).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        imagined_n = dream_rollout(self.model, s, actual.shape[0]).squeeze(0).cpu().numpy()
        imagined = imagined_n * self.std.numpy() + self.mean.numpy()
        pred = np.linalg.norm(imagined[:, 2:4], axis=1)
        true = np.linalg.norm(actual[:, 2:4], axis=1)
        return float(np.mean(np.maximum(pred - true, 0.0)))


def collect_agents(data_root, limit_scenarios=None):
    out = []
    for type_dir in sorted(Path(data_root).glob("*_normal")):
        for name in list_scenarios(type_dir):
            seqs = build_agent_sequences(type_dir, name)
            for tid, arr in seqs.items():
                if len(arr) >= SEED_LEN + HORIZON:
                    out.append((name, tid, np.asarray(arr, dtype=np.float32)))
            if limit_scenarios and len({o[0] for o in out}) >= limit_scenarios:
                return out
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenarios", type=int, default=0)
    ap.add_argument("--max_per_attack", type=int, default=200)
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    dreamer = Dreamer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
                      cfg, stats["mean"], stats["std"], device)

    print(f"[setup] device {device}")
    print(f"[setup] collecting agent trajectories...")
    agents = collect_agents(REPO_ROOT / cfg["data"]["raw_dir"],
                            args.max_scenarios or None)
    scenarios = sorted({a[0] for a in agents})
    print(f"[setup] {len(agents)} agent trajectories from {len(scenarios)} scenarios")

    usable = []
    for name, tid, arr in agents:
        seed = arr[:SEED_LEN]
        speed = np.linalg.norm(seed[-1, 2:4])
        if speed >= MIN_SPEED_MS:
            usable.append((name, tid, arr))
    print(f"[setup] {len(usable)} agents moving faster than {MIN_SPEED_MS} m/s at the seed point\n")

    if len(usable) < 20:
        print("[abort] too few moving agents to evaluate")
        return

    rng = np.random.default_rng(0)
    rng.shuffle(usable)

    benign = []
    for name, tid, arr in usable[:args.max_per_attack * 2]:
        benign.append(dreamer.shortfall(arr[:SEED_LEN], arr[SEED_LEN:SEED_LEN + HORIZON]))
    benign = np.array(benign)
    print(f"[data] {len(benign)} benign sequences")

    results = {}
    all_scores, all_labels, all_kinds = [], [], []

    for attack, target in ATTACK_TARGET_SPEED.items():
        attacked = []
        for name, tid, arr in usable[:args.max_per_attack]:
            actual = inject_speed_command(arr[SEED_LEN:SEED_LEN + HORIZON].copy(), target, 0)
            attacked.append(dreamer.shortfall(arr[:SEED_LEN], actual))
        attacked = np.array(attacked)

        labels = [0] * len(benign) + [1] * len(attacked)
        scores = benign.tolist() + attacked.tolist()
        thr = float(np.percentile(benign, 99))
        results[attack] = detection_metrics(labels, scores, threshold=thr)
        results[attack]["n_attacked"] = len(attacked)

        all_scores += attacked.tolist()
        all_labels += [1] * len(attacked)
        all_kinds += [attack] * len(attacked)

    all_scores = benign.tolist() + all_scores
    all_labels = [0] * len(benign) + all_labels
    all_kinds = [None] * len(benign) + all_kinds
    thr = float(np.percentile(benign, 99))
    overall = detection_metrics(all_labels, all_scores, threshold=thr, attack_types=all_kinds)

    print()
    print("=" * 86)
    print("DETECTION BY SPEED SHORTFALL, ACROSS MANY SCENARIOS")
    print("=" * 86)
    print(f"{'attack':<28}{'target speed':<15}{'n':<8}{'AUC':<10}{'detected at 1% FA'}")
    print("-" * 86)
    for a, r in results.items():
        if "error" in r:
            print(f"{a:<28}{r['error']}")
            continue
        cm = r["confusion_matrix"]
        print(f"{a:<28}{ATTACK_TARGET_SPEED[a]:<15}{r['n_attacked']:<8}"
              f"{r['auc']:<10.4f}{cm['tp']}/{r['n_positive']}")
    print("-" * 86)
    if "error" not in overall:
        cm = overall["confusion_matrix"]
        print(f"{'ALL COMBINED':<28}{'':<15}{overall['n_positive']:<8}"
              f"{overall['auc']:<10.4f}{cm['tp']}/{overall['n_positive']}")
        print(f"\nbenign sequences {len(benign)}, false alarms "
              f"{overall['false_alarm_rate']*100:.2f}%")

    print()
    print("=" * 86)
    print("SCOPE")
    print("=" * 86)
    print(f"  {len(scenarios)} scenarios, {len(usable)} distinct moving agents.")
    print(f"  Attacks are injected by applying the target speed each attack commands in")
    print(f"  CARLA, rather than by re-simulating. This trades simulator fidelity for")
    print(f"  sample size; the CARLA evaluation covers fidelity on a single scenario.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in results.items():
        v = dict(v); v.pop("roc_curve", None)
        clean[k] = v
    ov = dict(overall); ov.pop("roc_curve", None)
    with open(out / "detection_at_scale.json", "w") as f:
        json.dump({"n_scenarios": len(scenarios), "n_agents": len(usable),
                   "n_benign": len(benign), "seed_len": SEED_LEN, "horizon": HORIZON,
                   "per_attack": clean, "overall": ov}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/detection_at_scale.json")


if __name__ == "__main__":
    main()