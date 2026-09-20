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
DT = 0.1
MIN_SPEED_MS = 3.0
DECEL_MS2 = 5.0
HORIZONS = [10, 20, 30, 50]
SPEED_BANDS = [(3.0, 6.0), (6.0, 10.0), (10.0, 1e9)]

ATTACK_TARGET_SPEED = {
    "sensor_spoofing": 0.0,
    "traffic_light_tampering": 0.0,
    "universal_perturbation": 0.0,
    "fake_emergency": 1.0,
    "fake_safety": 0.5,
    "sybil": 0.5,
}


def inject(window, target_speed, instant=False, decel=DECEL_MS2):
    out = window.copy()
    prev = out[0]
    speed = math.hypot(prev[2], prev[3])
    for t in range(len(out)):
        base = out[t - 1] if t > 0 else out[0]
        vx, vy = base[2], base[3]
        cur = math.hypot(vx, vy)
        ux, uy = (vx / cur, vy / cur) if cur > 1e-6 else (math.cos(base[4]), math.sin(base[4]))

        if instant:
            speed = target_speed
        else:
            speed = max(target_speed, speed - decel * DT) if speed > target_speed \
                else min(target_speed, speed + decel * DT)

        out[t, 2] = ux * speed
        out[t, 3] = uy * speed
        if t > 0:
            out[t, 0] = out[t - 1, 0] + out[t, 2] * DT
            out[t, 1] = out[t - 1, 1] + out[t, 3] * DT
    return out


class Model:
    def __init__(self, ckpt, cfg, mean, std, device):
        self.device = device
        self.mean_np = np.asarray(mean, dtype=np.float32)
        self.std_np = np.asarray(std, dtype=np.float32)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.net = PAMPOS(
            input_dim=cfg["model"]["input_dim"], d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"], num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"]).to(device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.eval()
        self.feature_mae = None

    def _n(self, w):
        return ((torch.from_numpy(w).float() - self.mean) / self.std).unsqueeze(0).to(self.device)

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
        return topk_anomaly_score(e, k=3).mean().item()

    @torch.no_grad()
    def measures(self, seed, actual):
        s = self._n(seed)
        imagined_n = dream_rollout(self.net, s, actual.shape[0])
        total = float(torch.abs(imagined_n - self._n(actual)).mean().item())
        imagined = imagined_n.squeeze(0).cpu().numpy() * self.std_np + self.mean_np
        pred = np.linalg.norm(imagined[:, 2:4], axis=1)
        true = np.linalg.norm(actual[:, 2:4], axis=1)
        shortfall = float(np.mean(np.maximum(pred - true, 0.0)))
        initial = float(np.linalg.norm(seed[-1, 2:4]))
        relative = shortfall / initial if initial > 1e-6 else 0.0
        return total, shortfall, relative


def heuristic(seq):
    sp = np.linalg.norm(seq[:, 2:4], axis=1)
    return float(np.abs(np.diff(sp)).max()) if len(sp) > 1 else 0.0


def collect(root, limit=None):
    out = []
    for td in sorted(Path(root).glob("*_normal")):
        for name in list_scenarios(td):
            for tid, arr in build_agent_sequences(td, name).items():
                a = np.asarray(arr, dtype=np.float32)
                if len(a) >= SEED_LEN + max(HORIZONS):
                    out.append((name, tid, a))
            if limit and len({o[0] for o in out}) >= limit:
                return out
    return out


def evaluate(model, usable, horizon, instant):
    benign, attacked = [], []
    for _, _, arr in usable:
        seed, actual = arr[:SEED_LEN], arr[SEED_LEN:SEED_LEN + horizon]
        tot, sh, rel = model.measures(seed, actual)
        benign.append({"total": tot, "shortfall": sh, "relative": rel,
                       "heuristic": heuristic(arr[:SEED_LEN + horizon]),
                       "speed": float(np.linalg.norm(seed[-1, 2:4]))})
    for attack, target in ATTACK_TARGET_SPEED.items():
        for _, _, arr in usable:
            seed = arr[:SEED_LEN]
            actual = inject(arr[SEED_LEN:SEED_LEN + horizon].copy(), target, instant=instant)
            tot, sh, rel = model.measures(seed, actual)
            attacked.append({"attack": attack, "total": tot, "shortfall": sh, "relative": rel,
                             "heuristic": heuristic(np.concatenate([seed, actual])),
                             "speed": float(np.linalg.norm(seed[-1, 2:4]))})
    return benign, attacked


def score(benign, attacked, key):
    b = np.array([r[key] for r in benign])
    a = np.array([r[key] for r in attacked])
    thr = float(np.percentile(b, 99))
    labels = [0] * len(b) + [1] * len(a)
    res = detection_metrics(labels, b.tolist() + a.tolist(), threshold=thr)
    return res, thr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenarios", type=int, default=0)
    ap.add_argument("--n_agents", type=int, default=400)
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    model = Model(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
                  cfg, st["mean"], st["std"], device)

    print(f"[setup] device {device}")
    agents = collect(REPO_ROOT / cfg["data"]["raw_dir"], args.max_scenarios or None)
    usable = [(n, t, a) for n, t, a in agents
              if np.linalg.norm(a[SEED_LEN - 1, 2:4]) >= MIN_SPEED_MS]
    rng = np.random.default_rng(0)
    rng.shuffle(usable)
    usable = usable[:args.n_agents]
    print(f"[setup] {len({u[0] for u in usable})} scenarios, {len(usable)} moving agents\n")
    model.calibrate([a[:SEED_LEN] for _, _, a in usable[:300]])

    print("=" * 92)
    print("W6 -- DOES REALISTIC DECELERATION CHANGE THE PICTURE?")
    print("=" * 92)
    print(f"Instant injection sets the speed in one frame. Gradual decelerates at")
    print(f"{DECEL_MS2} m/s^2, roughly firm braking, which is what a real vehicle does.\n")
    print(f"{'injection':<14}{'horizon':<10}{'heuristic AUC':<18}{'shortfall AUC':<18}{'shortfall caught'}")
    print("-" * 92)

    rows = []
    for instant in (True, False):
        for h in HORIZONS:
            b, a = evaluate(model, usable, h, instant)
            rh, _ = score(b, a, "heuristic")
            rs, _ = score(b, a, "shortfall")
            cm = rs["confusion_matrix"]
            rows.append({"instant": instant, "horizon": h,
                         "heuristic_auc": rh["auc"], "shortfall_auc": rs["auc"],
                         "shortfall_detected": cm["tp"], "n": len(a)})
            print(f"{'instant' if instant else 'gradual':<14}{h:<10}"
                  f"{rh['auc']:<18.4f}{rs['auc']:<18.4f}{cm['tp']}/{len(a)}")
    print("-" * 92)

    inst = [r for r in rows if r["instant"]]
    grad = [r for r in rows if not r["instant"]]
    print()
    print("  Under instant injection the heuristic looks strong because an 8-to-0 jump")
    print("  in one frame is trivial to spot. Under gradual deceleration that jump")
    print("  disappears and the comparison becomes fair.")

    best_h = max(HORIZONS)
    print()
    print("=" * 92)
    print(f"W5 -- DOES NORMALISING BY INITIAL SPEED HELP SLOW VEHICLES?")
    print("=" * 92)
    b, a = evaluate(model, usable, best_h, instant=False)
    print(f"Gradual injection, horizon {best_h}.\n")
    print(f"{'measure':<26}{'AUC':<12}{'detected':<16}{'3-6 m/s band'}")
    print("-" * 92)

    band_out = {}
    for key, label in [("shortfall", "speed shortfall"),
                       ("relative", "shortfall / initial speed")]:
        res, thr = score(b, a, key)
        cm = res["confusion_matrix"]
        slow = [r for r in a if 3.0 <= r["speed"] < 6.0]
        slow_hit = sum(1 for r in slow if r[key] > thr)
        band_out[key] = {"auc": res["auc"], "detected": cm["tp"], "n": len(a),
                         "slow_detected": slow_hit, "slow_n": len(slow)}
        rate = slow_hit / len(slow) * 100 if slow else 0.0
        print(f"{label:<26}{res['auc']:<12.4f}{cm['tp']}/{len(a):<11}"
              f"{slow_hit}/{len(slow)} ({rate:.1f}%)")
    print("-" * 92)

    print()
    print(f"{'measure':<26}", end="")
    for lo, hi in SPEED_BANDS:
        print(f"{(f'{lo:.0f}-{hi:.0f}' if hi < 1e9 else f'{lo:.0f}+'):<14}", end="")
    print()
    print("-" * 92)
    for key, label in [("shortfall", "speed shortfall"),
                       ("relative", "shortfall / speed")]:
        _, thr = score(b, a, key)
        print(f"{label:<26}", end="")
        for lo, hi in SPEED_BANDS:
            sel = [r for r in a if lo <= r["speed"] < hi]
            if not sel:
                print(f"{'-':<14}", end="")
                continue
            hit = sum(1 for r in sel if r[key] > thr)
            print(f"{f'{hit/len(sel)*100:.1f}%':<14}", end="")
        print()
    print("-" * 92)

    imp = band_out["relative"]["slow_detected"] - band_out["shortfall"]["slow_detected"]
    print()
    if imp > 0:
        print(f"  Normalising improves the slow band by {imp} sequences.")
    elif imp < 0:
        print(f"  Normalising makes the slow band worse by {-imp} sequences.")
    else:
        print(f"  Normalising makes no difference to the slow band.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "injection_and_normalisation.json", "w") as f:
        json.dump({"decel_ms2": DECEL_MS2, "horizons": HORIZONS,
                   "injection_comparison": rows,
                   "normalisation": band_out}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/injection_and_normalisation.json")


if __name__ == "__main__":
    main()