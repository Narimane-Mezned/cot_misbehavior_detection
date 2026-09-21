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

SEED_LEN = 10
HORIZON = 3


def commanded(run):
    bh = ((run.get("attack_record") or {}).get("metadata") or {}).get("enforced_behaviour") or {}
    return [str(t) for t in bh.get("track_ids", [])]


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


def replay_drift(run_clean, sens, tid):
    d = []
    for i, fr in enumerate(run_clean["trajectory"]):
        if i >= len(sens):
            break
        a = fr["agents"].get(tid); b = sens[i]["objects"].get(tid)
        if a and b:
            d.append(math.dist((a["x"], a["y"]), (b["x"], b["y"])))
    return max(d) if d else float("nan")


class M:
    def __init__(self, ck, cfg, mean, std, dev):
        self.dev = dev
        self.mn, self.sd = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.m, self.s = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        self.net = PAMPOS(input_dim=cfg["model"]["input_dim"], d_model=cfg["model"]["d_model"],
                          nhead=cfg["model"]["nhead"], num_layers=cfg["model"]["num_layers"],
                          dim_feedforward=cfg["model"]["dim_feedforward"],
                          dropout=cfg["model"]["dropout"]).to(dev)
        self.net.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model_state_dict"])
        self.net.eval()

    @torch.no_grad()
    def shortfall(self, seed, actual):
        s = ((torch.from_numpy(seed).float() - self.m) / self.s).unsqueeze(0).to(self.dev)
        im = dream_rollout(self.net, s, actual.shape[0]).squeeze(0).cpu().numpy() * self.sd + self.mn
        return float(np.mean(np.maximum(np.linalg.norm(im[:, 2:4], axis=1)
                                        - np.linalg.norm(actual[:, 2:4], axis=1), 0.0)))


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    m = M(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
          cfg, st["mean"], st["std"], dev)

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=SEED_LEN + HORIZON)
    g = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    _, va = random_split(ds, [len(ds) - vs, vs], generator=g)
    ref = np.array([m.shortfall(ds.sequences[i][:SEED_LEN], ds.sequences[i][SEED_LEN:SEED_LEN + HORIZON])
                    for i in list(va.indices)[:3000]])
    thr = float(np.percentile(ref, 99))
    print(f"[setup] 1% false-alarm threshold on held-out benign: {thr:.4f}\n")

    traj = REPO_ROOT / "data" / "attack_trajectories"
    scen = sorted({f.name.split("__")[0] for f in traj.glob("*__attacked.json")})
    summary = []

    for s in scen:
        ld = find_label_dir(REPO_ROOT / cfg["data"]["raw_dir"], s)
        if ld is None:
            print(f"[skip] no labels for {s}")
            continue
        sens = sensors(ld)
        rc = json.load(open(traj / f"{s}__clean.json"))
        print("=" * 104)
        print(s)
        print("=" * 104)
        print(f"{'attack':<24}{'agent':<8}{'drift m':<9}{'before':<8}{'clean':<8}{'att':<8}"
              f"{'benign sf':<11}{'attack sf':<11}{'caught'}")
        print("-" * 104)
        hit = tot = 0
        for f in sorted(traj.glob(f"{s}__*__attacked.json")):
            a = f.name.split("__")[1]
            ra = json.load(open(f))
            t0 = ra["attack_start_frame"]
            for tid in commanded(ra):
                A = series(ra, tid, sens); C = series(rc, tid, sens)
                if A is None or C is None or t0 < SEED_LEN or t0 + HORIZON > min(len(A), len(C)):
                    continue
                sa = m.shortfall(A[t0 - SEED_LEN:t0], A[t0:t0 + HORIZON])
                sb = m.shortfall(C[t0 - SEED_LEN:t0], C[t0:t0 + HORIZON])
                before = float(np.linalg.norm(A[t0 - 1, 2:4]))
                ca = float(np.mean(np.linalg.norm(C[t0:t0 + HORIZON, 2:4], axis=1)))
                aa = float(np.mean(np.linalg.norm(A[t0:t0 + HORIZON, 2:4], axis=1)))
                drift = replay_drift(rc, sens, tid)
                caught = sa > thr
                hit += caught; tot += 1
                flag = "" if caught else "  <-- missed"
                print(f"{a:<24}{tid:<8}{drift:<9.2f}{before:<8.2f}{ca:<8.2f}{aa:<8.2f}"
                      f"{sb:<11.3f}{sa:<11.3f}{'yes' if caught else 'no'}{flag}")
                summary.append({"scenario": s, "attack": a, "agent": tid, "drift": drift,
                                "before": before, "clean_speed": ca, "attacked_speed": aa,
                                "benign_shortfall": sb, "attacked_shortfall": sa, "caught": bool(caught)})
        print("-" * 104)
        print(f"caught {hit}/{tot}\n")

    print("=" * 104)
    print("WHAT DISTINGUISHES THE MISSES?")
    print("=" * 104)
    caught = [r for r in summary if r["caught"]]
    missed = [r for r in summary if not r["caught"]]
    for label, key in [("speed before attack", "before"), ("clean speed after", "clean_speed"),
                       ("attacked speed after", "attacked_speed"), ("replay drift (m)", "drift")]:
        c = np.mean([r[key] for r in caught]) if caught else float("nan")
        mi = np.mean([r[key] for r in missed]) if missed else float("nan")
        print(f"  {label:<24}caught {c:7.2f}    missed {mi:7.2f}")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "detection_per_scenario.json", "w") as f:
        json.dump({"threshold": thr, "rows": summary}, f, indent=2)
    print(f"\n[done] saved to outputs/results/detection_per_scenario.json")


if __name__ == "__main__":
    main()