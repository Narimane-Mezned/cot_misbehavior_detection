import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import PAMPOSTarget

Z_LOCAL = 1.5
CALIBRATION_SIZE = 300
PASSES = 2
N_PROBES = 60
SEEDS = [0, 1, 2]
ROUNDS = [1, 2, 4, 8]
TOTAL_BUDGET = 0.15
K_PARITY_PATH = "outputs/results/calibration_method_comparison.json"


def filter_passes(W, rounds=PASSES, z=Z_LOCAL):
    out = W.copy()
    for _ in range(rounds):
        n, T, _ = out.shape
        flagged = np.zeros((n, T), dtype=bool)
        med = np.zeros_like(out)
        for t in range(T):
            others = np.delete(out, t, axis=1)
            m = np.median(others, axis=1)
            mad = np.median(np.abs(others - m[:, None, :]), axis=1) + 1e-8
            flagged[:, t] = (np.abs(out[:, t, :] - m) / (1.4826 * mad)).max(axis=1) > z
            med[:, t, :] = m
        out = out.copy()
        out[flagged] = med[flagged]
    return out


def est_percentile(s, pct=99.0):
    return float(np.percentile(s, pct))


def est_median_mad(s, k):
    s = np.asarray(s)
    m = np.median(s)
    return float(m + k * 1.4826 * np.median(np.abs(s - m)))


def poison(windows, trigger, fraction, rng):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    k = max(1, int(fraction * len(windows) * T))
    locs = rng.choice(len(windows) * T, size=k, replace=False)
    for loc in locs:
        wi, t = divmod(loc, T)
        out[wi][t] = trigger
    return out


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


def run_campaign(target, train_windows, probes, all_scores, calib_pool,
                 rounds, k_ref, rng, defended):
    """A deployment that recalibrates `rounds` times. The adversary spreads a
    fixed total budget evenly across the rounds, so each individual round
    carries less contamination than the single-round attack does. The
    threshold is whatever the final round produces; the question is whether
    the earlier rounds leave any residue that the final one inherits.

    Two leakage paths are modelled. The calibration set drawn in each round
    retains the windows poisoned in previous rounds that were not filtered
    out, since a deployment that stores its calibration data would carry them
    forward. The per-feature scale is likewise carried forward."""
    per_round = TOTAL_BUDGET / rounds
    carried = [w.copy() for w in calib_pool]
    hidden = 0
    thr = None

    for r in range(rounds):
        trigger = probes[0].mean(axis=0) if len(probes) else None
        poisoned = poison(carried, trigger, per_round, rng)
        used = [x for x in filter_passes(np.stack(poisoned))] if defended else poisoned
        sc = score_all(target, used)
        thr = est_median_mad(sc, k_ref) if defended else est_percentile(sc)
        carried = used          # the next round inherits what survived

    for probe in probes:
        if target.raw_score(probe) <= thr:
            hidden += 1
    fa = 100.0 * float((all_scores > thr).mean())
    return 100.0 * hidden / max(len(probes), 1), fa


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kp = REPO_ROOT / K_PARITY_PATH
    if not kp.exists():
        print(f"[abort] {K_PARITY_PATH} not found")
        return
    k_ref = float(json.load(open(kp))["k_parity"])

    print(f"[setup] device {device}, k = {k_ref:.4f}, {PASSES} filter passes")
    print(f"[setup] A deployment that recalibrates periodically gives the")
    print(f"[setup] adversary several opportunities. We hold the TOTAL budget")
    print(f"[setup] at {int(TOTAL_BUDGET*100)}% and spread it evenly over")
    print(f"[setup] {ROUNDS} rounds, so each round carries less contamination")
    print(f"[setup] than the single-round attack. Each round's surviving")
    print(f"[setup] calibration data is carried into the next.")
    print(f"[setup] {len(SEEDS)} seeds x {N_PROBES} probes\n")

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=cfg["training"]["seq_len"])
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, _ = random_split(ds, [len(ds) - vs, vs], generator=gen)
    train_windows = [ds.sequences[i] for i in tr.indices]
    all_windows = [ds.sequences[i] for i in range(len(ds))]

    target = PAMPOSTarget(
        checkpoint_path=REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
        feature_stats_path=REPO_ROOT / "data" / "processed" / "feature_stats.npz",
        model_config={"input_dim": cfg["model"]["input_dim"],
                      "d_model": cfg["model"]["d_model"],
                      "nhead": cfg["model"]["nhead"],
                      "num_layers": cfg["model"]["num_layers"],
                      "dim_feedforward": cfg["model"]["dim_feedforward"],
                      "dropout": cfg["model"]["dropout"]},
        device=device)

    cache = {}

    def setup(seed):
        if seed not in cache:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train_windows),
                             size=min(CALIBRATION_SIZE, len(train_windows)),
                             replace=False)
            calib = [np.asarray(train_windows[i], dtype=np.float32) for i in idx]
            thr = est_percentile(score_all(target, calib))
            print(f"[cache] scoring all {len(all_windows)} windows for seed {seed}",
                  flush=True)
            sc = np.asarray([target.raw_score(w) for w in all_windows])
            order = np.argsort(-sc)
            probes = [np.asarray(all_windows[i], dtype=np.float32)
                      for i in order[:N_PROBES] if sc[i] > thr]
            cache[seed] = (calib, probes, sc)
        calib, probes, sc = cache[seed]
        return np.random.default_rng(seed + 31337), calib, probes, sc

    results = {}
    for rounds in ROUNDS:
        for defended in (False, True):
            att, fas = [], []
            for seed in SEEDS:
                rng, calib, probes, sc = setup(seed)
                a, fa = run_campaign(target, train_windows, probes, sc, calib,
                                     rounds, k_ref, rng, defended)
                att.append(a); fas.append(fa)
            results[(rounds, defended)] = {
                "attack": float(np.mean(att)), "sd": float(np.std(att)),
                "fa": float(np.mean(fas)),
                "per_round": 100.0 * TOTAL_BUDGET / rounds}
            tag = "defended" if defended else "undefended"
            print(f"[{rounds} round{'s' if rounds > 1 else ' '}] "
                  f"{100.0*TOTAL_BUDGET/rounds:4.1f}% per round  {tag:<10} "
                  f"attack {np.mean(att):5.1f} +/- {np.std(att):4.1f}", flush=True)
        print(flush=True)

    print("=" * 84)
    print("POISONING SPREAD ACROSS SUCCESSIVE CALIBRATION ROUNDS")
    print("=" * 84)
    print("Total budget held at %d%% throughout." % int(TOTAL_BUDGET * 100))
    print()
    print("%-10s%-14s%-22s%-22s%s" % ("rounds", "per round", "undefended",
                                      "defended", "benign FA %"))
    print("-" * 84)
    for rounds in ROUNDS:
        u = results[(rounds, False)]
        d = results[(rounds, True)]
        print("%-10s%-14s%-22s%-22s%.2f" % (
            rounds, "%.1f%%" % u["per_round"],
            "%.1f +/- %.1f" % (u["attack"], u["sd"]),
            "%.1f +/- %.1f" % (d["attack"], d["sd"]), d["fa"]))
    print("-" * 84)

    print()
    print("=" * 84)
    print("READING")
    print("=" * 84)
    one = results[(1, True)]["attack"]
    many = max(results[(r, True)]["attack"] for r in ROUNDS if r > 1)
    print(f"  single round, defended      : {one:.1f}%")
    print(f"  worst multi-round, defended : {many:.1f}%")
    print()
    if many > one + 10:
        print("  Spreading the budget across rounds defeats the defence where a")
        print("  single round does not. Contamination accumulates faster than")
        print("  the filter removes it, and this is a real failure mode that")
        print("  must be reported.")
    elif many > one + 3:
        print("  Spreading the budget gives the adversary a modest advantage.")
        print("  Report the margin.")
    else:
        print("  Spreading the budget across rounds gives the adversary no")
        print("  advantage: each round carries less contamination, and the")
        print("  filter removes it before it can accumulate. The defence is")
        print("  not evaded by patience.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "multiround_poisoning.json", "w") as f:
        json.dump({"rounds": ROUNDS, "total_budget": TOTAL_BUDGET,
                   "passes": PASSES, "seeds": SEEDS, "n_probes": N_PROBES,
                   "k_parity": k_ref,
                   "results": {f"{r}_{d}": v for (r, d), v in results.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/multiround_poisoning.json")


if __name__ == "__main__":
    main()