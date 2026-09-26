import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HORIZONS = [1, 2, 3, 5, 8]
TOPKS = [1, 3, 5]


def run(h, k):
    cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts" / "evaluate_detection_paired.py"),
           "--horizon", str(h), "--topk", str(k)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  h={h} k={k} FAILED")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-4:]))
        return None
    path = REPO_ROOT / "outputs" / "results" / f"detection_paired_comparison_h{h}k{k}.json"
    if not path.exists():
        print(f"  h={h} k={k}: no output written")
        return None
    return json.load(open(path))


def main():
    print(f"[setup] sweeping horizon {HORIZONS} x top-K {TOPKS}")
    print(f"[setup] {len(HORIZONS) * len(TOPKS)} configurations, everything else fixed")
    print(f"[setup] the paper uses h=3, k=3\n")

    grid = {}
    for h in HORIZONS:
        for k in TOPKS:
            print(f"[run] h={h} k={k}", flush=True)
            r = run(h, k)
            if r is None:
                continue
            res = r.get("results", r)
            grid[(h, k)] = res

    if not grid:
        print("[abort] nothing completed")
        return

    def auc_of(res, label):
        for key, v in res.items():
            if label.lower() in key.lower() and isinstance(v, dict) and "auc" in v:
                return float(v["auc"])
        return float("nan")

    print()
    print("=" * 80)
    print("SHORTFALL AUC BY ROLLOUT HORIZON AND TOP-K")
    print("=" * 80)
    print(f"{'horizon':<12}", end="")
    for k in TOPKS:
        print(f"{f'k={k}':<14}", end="")
    print()
    print("-" * 80)
    for h in HORIZONS:
        print(f"{h:<12}", end="")
        for k in TOPKS:
            v = grid.get((h, k))
            print(f"{auc_of(v, 'shortfall') if v else float('nan'):<14.4f}", end="")
        print()
    print("-" * 80)

    print()
    print("=" * 80)
    print("SINGLE-STEP AUC BY THE SAME GRID (the published rule)")
    print("=" * 80)
    print(f"{'horizon':<12}", end="")
    for k in TOPKS:
        print(f"{f'k={k}':<14}", end="")
    print()
    print("-" * 80)
    for h in HORIZONS:
        print(f"{h:<12}", end="")
        for k in TOPKS:
            v = grid.get((h, k))
            print(f"{auc_of(v, 'single-step') if v else float('nan'):<14.4f}", end="")
        print()
    print("-" * 80)

    sf = [auc_of(v, "shortfall") for v in grid.values()]
    ss = [auc_of(v, "single-step") for v in grid.values()]
    paper = grid.get((3, 3))

    print()
    print("=" * 80)
    print("READING")
    print("=" * 80)
    print(f"  shortfall AUC over the grid  : {min(sf):.4f} to {max(sf):.4f}")
    print(f"  single-step AUC over the grid: {min(ss):.4f} to {max(ss):.4f}")
    if paper:
        print(f"  at the paper's h=3, k=3      : shortfall {auc_of(paper, 'shortfall'):.4f}, "
              f"single-step {auc_of(paper, 'single-step'):.4f}")
    print()
    if min(sf) > max(ss):
        print("  Shortfall exceeds single-step at every configuration tested, so the")
        print("  result does not depend on the particular horizon and top-K chosen.")
    else:
        print("  Shortfall does not exceed single-step everywhere. Report the range and")
        print("  state at which settings the ordering fails.")

    out = REPO_ROOT / "outputs" / "results"
    with open(out / "horizon_topk_sweep.json", "w") as f:
        json.dump({"horizons": HORIZONS, "topks": TOPKS,
                   "shortfall_auc": {f"h{h}k{k}": auc_of(v, "shortfall")
                                     for (h, k), v in grid.items()},
                   "single_step_auc": {f"h{h}k{k}": auc_of(v, "single-step")
                                       for (h, k), v in grid.items()}},
                  f, indent=2)
    print(f"\n[done] saved to outputs/results/horizon_topk_sweep.json")


if __name__ == "__main__":
    main()