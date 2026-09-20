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
from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
from src.data_pipeline.deepaccident_loader import build_agent_sequences, list_scenarios
from src.eval.metrics import detection_metrics

SEED_LEN = 10
HORIZON = 3
DT = 0.1
MIN_SPEED_MS = 3.0
SPEED_BANDS = [(3.0, 6.0), (6.0, 10.0), (10.0, 15.0), (15.0, 1e9)]

ATTACK_TARGET_SPEED = {
    "sensor_spoofing": 0.0,
    "traffic_light_tampering": 0.0,
    "universal_perturbation": 0.0,
    "fake_emergency": 1.0,
    "fake_safety": 0.5,
    "sybil": 0.5,
}
KINEMATIC = [0, 1, 2, 3, 4]


def inject_speed_command(window, target_speed):
    out = window.copy()
    for t in range(len(out)):
        prev = out[t - 1] if t > 0 else out[0]
        vx, vy = prev[2], prev[3]
        speed = math.hypot(vx, vy)
        if speed < 1e-6:
            ux, uy = math.cos(prev[4]), math.sin(prev[4])
        else:
            ux, uy = vx / speed, vy / speed
        out[t, 2] = ux * target_speed
        out[t, 3] = uy * target_speed
        if t > 0:
            out[t, 0] = out[t - 1, 0] + out[t, 2] * DT
            out[t, 1] = out[t - 1, 1] + out[t, 3] * DT
    return out


class Model:
    def __init__(self, ckpt, cfg, mean, std, device, input_dim=None):
        self.device = device
        self.mean_np = np.asarray(mean, dtype=np.float32)
        self.std_np = np.asarray(std, dtype=np.float32)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.net = PAMPOS(
            input_dim=input_dim or cfg["model"]["input_dim"],
            d_model=cfg["model"]["d_model"], nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"]).to(device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.eval()
        self.feature_mae = None

    def _norm(self, w):
        return ((torch.from_numpy(w).float() - self.mean) / self.std).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def single_step(self, window):
        x = self._norm(window)
        e = per_feature_errors(self.net(x[:, :-1, :]), x[:, 1:, :])
        if self.feature_mae is not None:
            e = normalize_errors(e, self.feature_mae)
        return topk_anomaly_score(e, k=3).mean().item()

    @torch.no_grad()
    def calibrate(self, windows):
        self.feature_mae = None
        errs = []
        for w in windows:
            x = self._norm(w)
            errs.append(per_feature_errors(self.net(x[:, :-1, :]), x[:, 1:, :]))
        self.feature_mae = torch.cat(errs, dim=0).mean(dim=(0, 1))

    @torch.no_grad()
    def rollout(self, seed, actual):
        s = self._norm(seed)
        a = self._norm(actual)
        imagined_n = dream_rollout(self.net, s, actual.shape[0])
        total = float(torch.abs(imagined_n - a).mean().item())
        imagined = imagined_n.squeeze(0).cpu().numpy() * self.std_np + self.mean_np
        pred = np.linalg.norm(imagined[:, 2:4], axis=1)
        true = np.linalg.norm(actual[:, 2:4], axis=1)
        return total, float(np.mean(np.maximum(pred - true, 0.0)))


def heuristic_speed_change(seq):
    sp = np.linalg.norm(seq[:, 2:4], axis=1)
    return float(np.abs(np.diff(sp)).max()) if len(sp) > 1 else 0.0


def collect(data_root, limit=None):
    out = []
    for type_dir in sorted(Path(data_root).glob("*_normal")):
        for name in list_scenarios(type_dir):
            for tid, arr in build_agent_sequences(type_dir, name).items():
                a = np.asarray(arr, dtype=np.float32)
                if len(a) >= SEED_LEN + HORIZON:
                    out.append((name, tid, a))
            if limit and len({o[0] for o in out}) >= limit:
                return out
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenarios", type=int, default=0)
    ap.add_argument("--n_agents", type=int, default=400)
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckdir = REPO_ROOT / cfg["paths"]["checkpoint_dir"]
    proc = REPO_ROOT / "data" / "processed"

    st8 = np.load(proc / "feature_stats.npz")
    m8 = Model(ckdir / "pampos_baseline_best.pt", cfg, st8["mean"], st8["std"], device)

    m5 = m7 = None
    if (ckdir / "pampos_ablated_best.pt").exists() and (proc / "feature_stats_ablated.npz").exists():
        with open(REPO_ROOT / "configs" / "pampos_ablated.yaml") as f:
            c5 = yaml.safe_load(f)
        s5 = np.load(proc / "feature_stats_ablated.npz")
        m5 = Model(ckdir / "pampos_ablated_best.pt", c5, s5["mean"], s5["std"], device, 5)
    if (ckdir / "pampos_neighbour_best.pt").exists() and (proc / "feature_stats_neighbour.npz").exists():
        with open(REPO_ROOT / "configs" / "pampos_neighbour.yaml") as f:
            c7 = yaml.safe_load(f)
        s7 = np.load(proc / "feature_stats_neighbour.npz")
        m7 = Model(ckdir / "pampos_neighbour_best.pt", c7, s7["mean"], s7["std"], device, 7)

    print(f"[setup] device {device}")
    print(f"[setup] collecting trajectories...")
    agents = collect(REPO_ROOT / cfg["data"]["raw_dir"], args.max_scenarios or None)
    scen = sorted({a[0] for a in agents})
    usable = [(n, t, a) for n, t, a in agents
              if np.linalg.norm(a[SEED_LEN - 1, 2:4]) >= MIN_SPEED_MS]
    rng = np.random.default_rng(0)
    rng.shuffle(usable)
    usable = usable[:args.n_agents]
    print(f"[setup] {len(scen)} scenarios, {len(usable)} moving agents selected\n")

    m8.calibrate([a[:SEED_LEN] for _, _, a in usable[:300]])

    print("[run] scoring benign sequences...")
    rows = []
    for name, tid, arr in usable:
        seed, actual = arr[:SEED_LEN], arr[SEED_LEN:SEED_LEN + HORIZON]
        tot, short = m8.rollout(seed, actual)
        rows.append({"attack": None, "speed": float(np.linalg.norm(seed[-1, 2:4])),
                     "single_step": m8.single_step(arr[:SEED_LEN + HORIZON]),
                     "rollout_total": tot, "shortfall": short,
                     "heuristic": heuristic_speed_change(arr[:SEED_LEN + HORIZON]),
                     "random": float(rng.random())})
        if m5 is not None:
            rows[-1]["single_step_5"] = m5.single_step(arr[:SEED_LEN + HORIZON][:, KINEMATIC])

    print("[run] scoring attacked sequences...")
    for attack, target in ATTACK_TARGET_SPEED.items():
        for name, tid, arr in usable:
            seed = arr[:SEED_LEN]
            actual = inject_speed_command(arr[SEED_LEN:SEED_LEN + HORIZON].copy(), target)
            full = np.concatenate([seed, actual], axis=0)
            tot, short = m8.rollout(seed, actual)
            r = {"attack": attack, "speed": float(np.linalg.norm(seed[-1, 2:4])),
                 "single_step": m8.single_step(full),
                 "rollout_total": tot, "shortfall": short,
                 "heuristic": heuristic_speed_change(full),
                 "random": float(rng.random())}
            if m5 is not None:
                r["single_step_5"] = m5.single_step(full[:, KINEMATIC])
            rows.append(r)

    benign = [r for r in rows if r["attack"] is None]
    attacked = [r for r in rows if r["attack"] is not None]
    labels = [0] * len(benign) + [1] * len(attacked)
    kinds = [None] * len(benign) + [r["attack"] for r in attacked]

    measures = [("Random", "random"),
                ("Speed-change heuristic", "heuristic"),
                ("Single-step, 8-feature (as published)", "single_step")]
    if m5 is not None:
        measures.append(("Single-step, kinematic only", "single_step_5"))
    measures += [("Rollout, total divergence", "rollout_total"),
                 ("Rollout, speed shortfall", "shortfall")]

    print()
    print("=" * 94)
    print("DETECTION -- all scoring procedures, same data, same 1% false-alarm calibration")
    print("=" * 94)
    print(f"{'scoring procedure':<40}{'AUC':<10}{'detected':<16}{'false alarms'}")
    print("-" * 94)

    results = {}
    for label, key in measures:
        b = np.array([r[key] for r in benign])
        a = np.array([r[key] for r in attacked])
        thr = float(np.percentile(b, 99))
        res = detection_metrics(labels, b.tolist() + a.tolist(), threshold=thr, attack_types=kinds)
        results[label] = res
        cm = res["confusion_matrix"]
        print(f"{label:<40}{res['auc']:<10.4f}{cm['tp']}/{len(a):<11}"
              f"{res['false_alarm_rate']*100:.2f}%")
    print("-" * 94)

    print()
    print("=" * 94)
    print("SPEED SHORTFALL -- detection by the agent's speed before the attack")
    print("=" * 94)
    b = np.array([r["shortfall"] for r in benign])
    thr = float(np.percentile(b, 99))
    print(f"{'speed band':<20}{'n':<10}{'detected':<16}{'rate'}")
    print("-" * 94)
    bands = []
    for lo, hi in SPEED_BANDS:
        sel = [r for r in attacked if lo <= r["speed"] < hi]
        if not sel:
            continue
        hit = sum(1 for r in sel if r["shortfall"] > thr)
        name = f"{lo:.0f}-{hi:.0f} m/s" if hi < 1e9 else f"{lo:.0f}+ m/s"
        bands.append({"band": name, "n": len(sel), "detected": hit,
                      "rate": hit / len(sel)})
        print(f"{name:<20}{len(sel):<10}{hit}/{len(sel):<11}{hit/len(sel)*100:.1f}%")
    print("-" * 94)

    print()
    print("=" * 94)
    print("PER ATTACK -- speed shortfall")
    print("=" * 94)
    per_attack = {}
    for attack in ATTACK_TARGET_SPEED:
        sel = [r for r in attacked if r["attack"] == attack]
        hit = sum(1 for r in sel if r["shortfall"] > thr)
        per_attack[attack] = {"n": len(sel), "detected": hit}
        print(f"  {attack:<28}target {ATTACK_TARGET_SPEED[attack]:<6} "
              f"{hit}/{len(sel)}  ({hit/len(sel)*100:.1f}%)")

    print()
    print("=" * 94)
    print("NOTE")
    print("=" * 94)
    print("  Injection reproduces the commanded speed only, so attacks commanding the same")
    print("  speed yield identical sequences: three distinct conditions, not six.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in results.items():
        v = dict(v); v.pop("roc_curve", None)
        clean[k] = v
    with open(out / "detection_full_comparison.json", "w") as f:
        json.dump({"n_scenarios": len(scen), "n_agents": len(usable),
                   "n_benign": len(benign), "n_attacked": len(attacked),
                   "results": clean, "speed_bands": bands,
                   "per_attack": per_attack}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/detection_full_comparison.json")


if __name__ == "__main__":
    main()