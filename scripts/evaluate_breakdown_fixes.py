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
Z_GLOBAL = 4.0
CALIBRATION_SIZE = 300
FRACTIONS = [0.15, 0.30]
N_PROBES = 20
SEEDS = [0, 1, 2]
K_PARITY_PATH = "outputs/results/calibration_method_comparison.json"


def _within_window(W, z=Z_LOCAL):
    n, T, _ = W.shape
    flagged = np.zeros((n, T), dtype=bool)
    med = np.zeros_like(W)
    for t in range(T):
        others = np.delete(W, t, axis=1)
        m = np.median(others, axis=1)
        mad = np.median(np.abs(others - m[:, None, :]), axis=1) + 1e-8
        flagged[:, t] = (np.abs(W[:, t, :] - m) / (1.4826 * mad)).max(axis=1) > z
        med[:, t, :] = m
    return flagged, med


def _global_stats(W):
    flat = W.reshape(-1, W.shape[-1])
    gmed = np.median(flat, axis=0)
    gmad = np.median(np.abs(flat - gmed), axis=0) + 1e-8
    return gmed, gmad


def fix_published(W):
    """A: the filter as published. Replace each flagged timestep with the
    median of the other timesteps in its own window."""
    flagged, med = _within_window(W)
    out = W.copy()
    out[flagged] = med[flagged]
    return out, float(flagged.mean())


def fix_global_reference(W):
    """B: flag the same timesteps, but repair toward a global reference drawn
    from the whole calibration set rather than the local window. A minority of
    poisoned windows cannot move a global median, so even a majority-poisoned
    window is repaired toward clean data."""
    flagged, _ = _within_window(W)
    gmed, _ = _global_stats(W)
    out = W.copy()
    out[flagged] = gmed
    return out, float(flagged.mean())


def fix_dual_criterion(W, zl=Z_LOCAL, zg=Z_GLOBAL):
    """C: flag a timestep if it is inconsistent with its own window OR
    improbable under the global distribution. The second test still fires when
    a window is majority-poisoned, because the injected value remains unusual
    relative to the calibration set as a whole."""
    flagged, med = _within_window(W, zl)
    gmed, gmad = _global_stats(W)
    gdev = np.abs(W - gmed) / (1.4826 * gmad)
    far = gdev.max(axis=2) > zg
    both = flagged | far
    out = W.copy()
    out[both] = np.broadcast_to(gmed, W.shape)[both]
    return out, float(both.mean())


def fix_iterated(W, rounds=3):
    """D: apply the published filter repeatedly. Each pass repairs the most
    inconsistent timesteps, which may let the next pass see a cleaner median."""
    out = W.copy()
    rate = 0.0
    for _ in range(rounds):
        flagged, med = _within_window(out)
        out = out.copy()
        out[flagged] = med[flagged]
        rate = float(flagged.mean())
    return out, rate


FIXES = [("A published", fix_published),
         ("B global-reference repair", fix_global_reference),
         ("C dual criterion", fix_dual_criterion),
         ("D iterated x3", fix_iterated)]


def est_percentile(s, pct=99.0):
    return float(np.percentile(s, pct))


def est_median_mad(s, k):
    s = np.asarray(s)
    m = np.median(s)
    return float(m + k * 1.4826 * np.median(np.abs(s - m)))


def poison(windows, trigger, fraction, rng):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    locs = rng.choice(len(windows) * T,
                      size=max(1, int(fraction * len(windows) * T)), replace=False)
    for loc in locs:
        wi, t = divmod(loc, T)
        out[wi][t] = trigger
    return out


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


def run_seed(target, train_windows, all_windows, fraction, seed, k_ref):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows),
                     size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
    calib = [np.asarray(train_windows[i], dtype=np.float32) for i in idx]

    clean_scores = score_all(target, calib)
    clean_thr = est_percentile(clean_scores)
    scores = np.asarray([target.raw_score(w) for w in all_windows])
    order = np.argsort(-scores)
    probes = [np.asarray(all_windows[i], dtype=np.float32)
              for i in order[:N_PROBES] if scores[i] > clean_thr]
    if not probes:
        return None

    out = {}
    for name, fn in FIXES:
        cleaned, _ = fn(np.stack(calib))
        sc_clean = score_all(target, [cleaned[i] for i in range(len(cleaned))])
        thr_clean = est_median_mad(sc_clean, k_ref)
        fa = 100.0 * float((scores > thr_clean).mean())
        utility = 100.0 * float(np.mean([target.raw_score(p) > thr_clean for p in probes]))

        hidden = 0
        for probe in probes:
            poisoned = poison(calib, probe.mean(axis=0), fraction, rng)
            cp, _ = fn(np.stack(poisoned))
            sc = score_all(target, [cp[i] for i in range(len(cp))])
            if target.raw_score(probe) <= est_median_mad(sc, k_ref):
                hidden += 1
        out[name] = {"attack": 100.0 * hidden / len(probes),
                     "fa": fa, "utility": utility}
    return out


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kp = REPO_ROOT / K_PARITY_PATH
    if not kp.exists():
        print(f"[abort] {K_PARITY_PATH} not found")
        return
    k_ref = float(json.load(open(kp))["k_parity"])

    print(f"[setup] device {device}, k = {k_ref:.4f}")
    print(f"[setup] The published filter degrades at high contamination because")
    print(f"[setup] once a majority of a window is poisoned, the within-window")
    print(f"[setup] median is itself the trigger. Three candidate repairs are")
    print(f"[setup] tested against it, all attack-agnostic:")
    print(f"[setup]   B repairs toward a global median instead of a local one")
    print(f"[setup]   C also flags values improbable under the global distribution")
    print(f"[setup]   D applies the published filter three times")
    print(f"[setup] rates {[f'{int(f*100)}%' for f in FRACTIONS]}, "
          f"{len(SEEDS)} seeds x {N_PROBES} probes\n")

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

    results = {}
    for frac in FRACTIONS:
        trials = [r for r in (run_seed(target, train_windows, all_windows,
                                       frac, s, k_ref) for s in SEEDS) if r]
        if not trials:
            continue
        results[frac] = {}
        for name, _ in FIXES:
            a = [t[name]["attack"] for t in trials]
            results[frac][name] = {
                "attack": float(np.mean(a)), "sd": float(np.std(a)),
                "fa": float(np.mean([t[name]["fa"] for t in trials])),
                "utility": float(np.mean([t[name]["utility"] for t in trials]))}
            print(f"[rate {int(frac*100):>3}%] {name:<28}"
                  f"attack {np.mean(a):5.1f}  "
                  f"FA {results[frac][name]['fa']:5.2f}  "
                  f"utility {results[frac][name]['utility']:5.1f}", flush=True)
        print(flush=True)

    if not results:
        print("[abort] nothing completed")
        return

    print("=" * 88)
    print("CANDIDATE REPAIRS FOR THE HIGH-CONTAMINATION BREAKDOWN")
    print("=" * 88)
    for frac in FRACTIONS:
        if frac not in results:
            continue
        print(f"\n  at {int(frac*100)}% contamination")
        print(f"  {'variant':<30}{'attack success':<20}{'benign FA %':<14}{'utility %'}")
        print("  " + "-" * 74)
        for name, _ in FIXES:
            r = results[frac][name]
            txt = f"{r['attack']:.1f} +/- {r['sd']:.1f}"
            print(f"  {name:<30}{txt:<20}{r['fa']:<14.2f}{r['utility']:.1f}")

    print()
    print("=" * 88)
    print("READING")
    print("=" * 88)
    worst = max(f for f in FRACTIONS if f in results)
    base = results[worst]["A published"]
    best = min((n for n, _ in FIXES if n != "A published"),
               key=lambda n: results[worst][n]["attack"])
    b = results[worst][best]
    print(f"  At {int(worst*100)}% the published filter allows {base['attack']:.1f}% "
          f"at {base['fa']:.2f}% false alarms.")
    print(f"  Best candidate: {best}, {b['attack']:.1f}% at {b['fa']:.2f}% "
          f"false alarms, {b['utility']:.1f}% utility.")
    print()
    if b["attack"] < base["attack"] - 10 and b["utility"] > 90:
        print("  The breakdown is fixable. Adopt this variant and report both.")
    elif b["attack"] < base["attack"] - 10:
        print("  Attack success falls but utility suffers; report the trade.")
    else:
        print("  None of the candidates improves materially. The degradation")
        print("  stands as a limitation, now with evidence that three generic")
        print("  repairs were tried.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "breakdown_fixes.json", "w") as f:
        json.dump({"fractions": FRACTIONS, "z_local": Z_LOCAL, "z_global": Z_GLOBAL,
                   "seeds": SEEDS, "n_probes": N_PROBES, "k_parity": k_ref,
                   "results": {str(k): v for k, v in results.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/breakdown_fixes.json")


if __name__ == "__main__":
    main()