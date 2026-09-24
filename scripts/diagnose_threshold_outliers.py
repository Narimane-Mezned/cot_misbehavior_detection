import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "paired", REPO_ROOT / "scripts" / "evaluate_detection_paired.py")
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)

import json
import math
import torch
import yaml
from torch.utils.data import random_split


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    sc = paired.Scorer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
                       cfg, st["mean"], st["std"], dev)

    ds = paired.DeepAccidentBenignDataset(
        data_root=REPO_ROOT / cfg["data"]["raw_dir"],
        seq_len=paired.SEED_LEN + paired.HORIZON)
    g = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, _ = random_split(ds, [len(ds) - vs, vs], generator=g)
    sc.calibrate([ds.sequences[i][:paired.SEED_LEN + paired.HORIZON]
                  for i in list(tr.indices)[:300]])

    traj = REPO_ROOT / "data" / "attack_trajectories"
    rows = []
    for s in sorted({f.name.split("__")[0] for f in traj.glob("*__attacked.json")}):
        ld = paired.find_label_dir(REPO_ROOT / cfg["data"]["raw_dir"], s)
        fc = traj / f"{s}__clean.json"
        if ld is None or not fc.exists():
            continue
        sens = paired.sensors(ld)
        rc = json.load(open(fc))
        for attack in paired.ATTACKS:
            fa = traj / f"{s}__{attack}__attacked.json"
            if not fa.exists():
                continue
            ra = json.load(open(fa))
            t0 = ra["attack_start_frame"]
            for tid in paired.commanded(ra):
                A = paired.series(ra, tid, sens)
                C = paired.series(rc, tid, sens)
                if A is None or C is None or t0 < paired.SEED_LEN:
                    continue
                if t0 + paired.HORIZON > min(len(A), len(C)):
                    continue
                sa = paired.mean_speed(ra, tid, t0, paired.HORIZON)
                scl = paired.mean_speed(rc, tid, t0, paired.HORIZON)
                if math.isnan(sa) or math.isnan(scl) or (scl - sa) < paired.MIN_EFFECT_MS:
                    continue
                m = sc.all_measures(C[t0 - paired.SEED_LEN:t0],
                                    C[t0:t0 + paired.HORIZON], ("clean", s, attack, tid))
                m.update({"scenario": s, "attack": attack, "agent": tid,
                          "clean_speed": scl})
                rows.append(m)

    print(f"{len(rows)} paired benign sequences\n")

    for key, label in [("single_step", "single-step"),
                       ("heuristic", "speed-change heuristic"),
                       ("total", "rollout total divergence"),
                       ("speed_shortfall", "rollout speed shortfall")]:
        v = np.array([r[key] for r in rows])
        thr = float(np.percentile(v, 99))
        order = np.argsort(v)[::-1]
        print("=" * 88)
        print(f"{label}   threshold (99th pct) = {thr:.4f}   median = {np.median(v):.4f}")
        print("=" * 88)
        print(f"{'rank':<6}{'score':<12}{'scenario':<45}{'agent':<9}{'clean speed'}")
        print("-" * 88)
        for rank, i in enumerate(order[:6], 1):
            r = rows[i]
            print(f"{rank:<6}{v[i]:<12.4f}{r['scenario']:<45}{r['agent']:<9}"
                  f"{r['clean_speed']:.2f}")
        print(f"{'':<6}{'...':<12}")
        med_i = order[len(order) // 2]
        print(f"{'mid':<6}{v[med_i]:<12.4f}{rows[med_i]['scenario']:<45}"
              f"{rows[med_i]['agent']:<9}{rows[med_i]['clean_speed']:.2f}")
        print()
        top = v[order[0]]
        rest = float(np.percentile(np.delete(v, order[0]), 99))
        print(f"  removing the single highest sequence moves the threshold "
              f"{thr:.4f} -> {rest:.4f}")
        print()

    print("=" * 88)
    print("BENIGN SCORES BY SCENARIO -- single-step")
    print("=" * 88)
    print(f"{'scenario':<45}{'n':<6}{'median':<12}{'max'}")
    print("-" * 88)
    for s in sorted({r["scenario"] for r in rows}):
        v = np.array([r["single_step"] for r in rows if r["scenario"] == s])
        print(f"{s:<45}{len(v):<6}{np.median(v):<12.4f}{v.max():.4f}")


if __name__ == "__main__":
    main()