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
from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
from src.eval.metrics import detection_metrics, compare_models, format_comparison, format_per_attack

ATTACKS = ["sensor_spoofing", "fake_emergency", "fake_safety",
           "traffic_light_tampering", "universal_perturbation", "sybil"]
FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw"]


def commanded_agents(run):
    record = run.get("attack_record") or {}
    behaviour = (record.get("metadata") or {}).get("enforced_behaviour") or {}
    return [str(t) for t in behaviour.get("track_ids", [])]


NEIGHBOUR_RADIUS = 50.0


def frame_speeds(run, idx):
    dt = run["fixed_delta_seconds"]
    tr = run["trajectory"]
    if idx == 0 or idx >= len(tr):
        return {}
    prev, cur = tr[idx - 1]["agents"], tr[idx]["agents"]
    out = {}
    for tid, c in cur.items():
        p = prev.get(tid)
        if p is not None:
            out[tid] = math.dist((c["x"], c["y"]), (p["x"], p["y"])) / dt
    return out


def neighbour_context_at(run, track_id, idx):
    speeds = frame_speeds(run, idx)
    if track_id not in speeds:
        return 0.0, 0.0
    agents = run["trajectory"][idx]["agents"]
    me = agents[track_id]
    near = [speeds[o] for o, a in agents.items()
            if o != track_id and o in speeds
            and math.dist((a["x"], a["y"]), (me["x"], me["y"])) <= NEIGHBOUR_RADIUS]
    if not near:
        return 0.0, 0.0
    return speeds[track_id] - float(np.mean(near)), float(len(near))


def agent_series(run, track_id, with_neighbour_context=False):
    dt = run["fixed_delta_seconds"]
    frames, xs, ys, yaws, idxs = [], [], [], [], []
    for i, fr in enumerate(run["trajectory"]):
        a = fr["agents"].get(track_id)
        if a is None:
            continue
        frames.append(fr["frame_idx"])
        idxs.append(i)
        xs.append(a["x"])
        ys.append(a["y"])
        yaws.append(math.radians(a["yaw_deg"]))

    if len(xs) < 2:
        return None

    vx, vy = [0.0], [0.0]
    for i in range(1, len(xs)):
        vx.append((xs[i] - xs[i - 1]) / dt)
        vy.append((ys[i] - ys[i - 1]) / dt)

    cols = [np.array(xs), np.array(ys), np.array(vx), np.array(vy), np.array(yaws)]

    if with_neighbour_context:
        rel, cnt = [], []
        for i in idxs:
            r, c = neighbour_context_at(run, track_id, i)
            rel.append(r)
            cnt.append(c)
        cols += [np.array(rel), np.array(cnt)]

    return {"frames": frames, "features": np.stack(cols, axis=1).astype(np.float32)}


def windows_from_run(run, seq_len, only_after_frame=None, with_neighbour_context=False):
    out = []
    for track_id in commanded_agents(run):
        series = agent_series(run, track_id, with_neighbour_context)
        if series is None:
            continue
        feats = series["features"]
        frames = series["frames"]
        for start in range(0, len(feats) - seq_len + 1, seq_len):
            end = start + seq_len
            if only_after_frame is not None and frames[start] < only_after_frame:
                continue
            out.append({
                "window": feats[start:end],
                "track_id": track_id,
                "start_frame": frames[start],
            })
    return out


class Scorer:
    def __init__(self, checkpoint, config, mean, std, device):
        self.device = device
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.model = PAMPOS(
            input_dim=config["model"]["input_dim"],
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
        ).to(device)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.feature_mae = None

    @torch.no_grad()
    def _errors(self, window):
        x = (torch.from_numpy(window).float() - self.mean) / self.std
        x = x.unsqueeze(0).to(self.device)
        e = per_feature_errors(self.model(x[:, :-1, :]), x[:, 1:, :])
        if self.feature_mae is not None:
            e = normalize_errors(e, self.feature_mae)
        return e

    def calibrate(self, windows, percentile=99.0):
        self.feature_mae = None
        errs = [self._errors(w) for w in windows]
        self.feature_mae = torch.cat(errs, dim=0).mean(dim=(0, 1))
        scores = [self.score(w) for w in windows]
        return float(np.percentile(scores, percentile))

    def score(self, window):
        return topk_anomaly_score(self._errors(window), k=3).mean().item()


def speed_change_baseline(window):
    speed = np.linalg.norm(window[:, 2:4], axis=1)
    return float(np.abs(np.diff(speed)).max()) if len(speed) > 1 else 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="data/attack_trajectories")
    ap.add_argument("--seq_len", type=int, default=10)
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "pampos_ablated.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj_dir = REPO_ROOT / args.traj_dir
    files = sorted(traj_dir.glob("*__attacked.json"))
    if not files:
        print(f"[abort] no attacked runs found in {traj_dir}")
        return

    scenarios = sorted({f.name.split("__")[0] for f in files})
    print(f"[setup] device {device}")
    print(f"[setup] {len(scenarios)} scenario(s): {', '.join(scenarios)}")
    print(f"[setup] evaluating the five-feature model on {FEATURE_NAMES}")
    print(f"[setup] the eight-feature model cannot be evaluated here: point_count and")
    print(f"[setup] is_camera_visible require sensors, which were not re-simulated\n")

    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats_ablated.npz")
    scorer = Scorer(
        REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_ablated_best.pt",
        cfg, stats["mean"], stats["std"], device)

    nb_cfg_path = REPO_ROOT / "configs" / "pampos_neighbour.yaml"
    nb_ckpt = REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_neighbour_best.pt"
    nb_stats_path = REPO_ROOT / "data" / "processed" / "feature_stats_neighbour.npz"
    nb_scorer = None
    if nb_cfg_path.exists() and nb_ckpt.exists() and nb_stats_path.exists():
        with open(nb_cfg_path) as f:
            nb_cfg = yaml.safe_load(f)
        nb_stats = np.load(nb_stats_path)
        nb_scorer = Scorer(nb_ckpt, nb_cfg, nb_stats["mean"], nb_stats["std"], device)
        print("[setup] neighbour-context model loaded; both will be compared")
    else:
        print("[setup] neighbour-context model not found; run train_neighbour.py to add it")

    attacked_windows, benign_windows = [], []
    for scenario in scenarios:
        fclean = traj_dir / f"{scenario}__clean.json"
        if not fclean.exists():
            print(f"[warn] no clean replay for {scenario}; skipping")
            continue
        run_clean = json.load(open(fclean))

        for attack in ATTACKS:
            fa = traj_dir / f"{scenario}__{attack}__attacked.json"
            if not fa.exists():
                continue
            run_a = json.load(open(fa))
            start = run_a["attack_start_frame"]

            nb_a = {(x["track_id"], x["start_frame"]): x["window"]
                    for x in windows_from_run(run_a, args.seq_len,
                                              only_after_frame=start,
                                              with_neighbour_context=True)}
            for w in windows_from_run(run_a, args.seq_len, only_after_frame=start):
                w["attack"] = attack
                w["nb_window"] = nb_a.get((w["track_id"], w["start_frame"]))
                attacked_windows.append(w)

            run_clean["attack_record"] = run_a["attack_record"]
            nb_c = {(x["track_id"], x["start_frame"]): x["window"]
                    for x in windows_from_run(run_clean, args.seq_len,
                                              only_after_frame=start,
                                              with_neighbour_context=True)}
            for w in windows_from_run(run_clean, args.seq_len, only_after_frame=start):
                w["attack"] = None
                w["nb_window"] = nb_c.get((w["track_id"], w["start_frame"]))
                benign_windows.append(w)

    control_windows = benign_windows
    print(f"[data] {len(attacked_windows)} attacked windows, "
          f"{len(benign_windows)} benign windows")
    print(f"[data] negatives are the SAME agents in the clean replay, over the same")
    print(f"[data] frames, so the only difference is the attack")
    if not attacked_windows or not control_windows:
        print("[abort] need both attacked and control windows")
        return

    clean_files = sorted(traj_dir.glob("*__clean.json"))
    calib = []
    for f in clean_files:
        run = json.load(open(f))
        for track_id in run["trajectory"][0]["agents"]:
            series = agent_series(run, track_id)
            if series is None:
                continue
            feats = series["features"]
            for s in range(0, len(feats) - args.seq_len + 1, args.seq_len):
                calib.append(feats[s:s + args.seq_len])
    print(f"[data] {len(calib)} calibration windows from clean replays")

    threshold = scorer.calibrate(calib) if calib else None
    print(f"[data] calibrated threshold {threshold:.4f}\n" if threshold else "")

    windows = attacked_windows + control_windows
    labels = [1] * len(attacked_windows) + [0] * len(control_windows)
    attack_types = [w["attack"] for w in windows]

    detector_scores = [scorer.score(w["window"]) for w in windows]
    baseline_scores = [speed_change_baseline(w["window"]) for w in windows]
    rng = np.random.default_rng(0)
    random_scores = rng.random(len(windows)).tolist()

    inverted_scores = [-s for s in detector_scores]

    pre_windows = []
    for scenario in scenarios:
        fclean = traj_dir / f"{scenario}__clean.json"
        if not fclean.exists():
            continue
        run_clean = json.load(open(fclean))
        start = None
        for attack in ATTACKS:
            fa = traj_dir / f"{scenario}__{attack}__attacked.json"
            if fa.exists():
                start = json.load(open(fa))["attack_start_frame"]
                break
        if start is None:
            continue
        for tid in run_clean["trajectory"][0]["agents"]:
            series = agent_series(run_clean, tid)
            if series is None:
                continue
            frames, feats = series["frames"], series["features"]
            for i in range(0, len(feats) - args.seq_len + 1):
                if frames[i] + args.seq_len <= start:
                    pre_windows.append(feats[i:i + args.seq_len])

    print(f"[data] {len(pre_windows)} pre-attack reference windows "
          f"(clean replay, before injection)")

    if pre_windows:
        ref = np.array([scorer.score(w) for w in pre_windows])
        lo, hi = float(np.percentile(ref, 1)), float(np.percentile(ref, 99))
        span = max(hi - lo, 1e-9)
        two_sided = [max((lo - sc) / span, (sc - hi) / span) for sc in detector_scores]
    else:
        two_sided = [0.0] * len(detector_scores)

    self_ref = {}
    for scenario in scenarios:
        for attack in ATTACKS + [None]:
            name = f"{scenario}__{attack}__attacked.json" if attack else f"{scenario}__clean.json"
            f = traj_dir / name
            if not f.exists():
                continue
            run = json.load(open(f))
            start = run.get("attack_start_frame", 10)
            for tid in run["trajectory"][0]["agents"]:
                series = agent_series(run, tid)
                if series is None:
                    continue
                frames, feats = series["frames"], series["features"]
                pre = [scorer.score(feats[i:i + args.seq_len])
                       for i in range(0, len(feats) - args.seq_len + 1)
                       if frames[i] + args.seq_len <= start]
                if pre:
                    self_ref[(name, tid)] = float(np.median(pre))

    def own_baseline(w, scenario, attack):
        name = f"{scenario}__{attack}__attacked.json" if attack else f"{scenario}__clean.json"
        return self_ref.get((name, w["track_id"]))

    relative_drop = []
    for w, lab in zip(windows, labels):
        sc_w = scorer.score(w["window"])
        base = own_baseline(w, scenarios[0], w["attack"])
        if base is None or base < 1e-9:
            relative_drop.append(0.0)
        else:
            relative_drop.append(max(0.0, (base - sc_w) / base))

    results = {
        "PAMPOS (5-feature)": detection_metrics(
            labels, detector_scores, threshold=threshold, attack_types=attack_types),
        "PAMPOS, inverted score": detection_metrics(
            labels, inverted_scores, attack_types=attack_types),
        "PAMPOS, two-sided": detection_metrics(
            labels, two_sided, attack_types=attack_types),
        "Relative surprise drop": detection_metrics(
            labels, relative_drop, attack_types=attack_types),
        "Speed-change heuristic": detection_metrics(
            labels, baseline_scores, attack_types=attack_types),
        "Random": detection_metrics(
            labels, random_scores, attack_types=attack_types),
    }

    if nb_scorer is not None:
        nb_calib = []
        for f in sorted(traj_dir.glob("*__clean.json")):
            run = json.load(open(f))
            for tid in run["trajectory"][0]["agents"]:
                series = agent_series(run, tid, with_neighbour_context=True)
                if series is None:
                    continue
                feats = series["features"]
                for i in range(0, len(feats) - args.seq_len + 1, args.seq_len):
                    nb_calib.append(feats[i:i + args.seq_len])
        nb_threshold = nb_scorer.calibrate(nb_calib) if nb_calib else None
        nb_scores = []
        for w in windows:
            nw = w.get("nb_window")
            nb_scores.append(nb_scorer.score(nw) if nw is not None else 0.0)
        results["PAMPOS + neighbour context"] = detection_metrics(
            labels, nb_scores, threshold=nb_threshold, attack_types=attack_types)
        results["PAMPOS + neighbour, inverted"] = detection_metrics(
            labels, [-x for x in nb_scores], attack_types=attack_types)

    print("=" * 88)
    print("DETECTION PERFORMANCE -- attacked agents vs the same agents driving normally")
    print("=" * 88)
    print(format_comparison(compare_models(results)))
    print()
    print("=" * 88)
    print("PER-ATTACK RECALL")
    print("=" * 88)
    print(format_per_attack(results))

    fwd = results["PAMPOS (5-feature)"]
    inv = results["PAMPOS, inverted score"]
    if "error" not in fwd and "error" not in inv:
        print()
        print("=" * 88)
        print("DIRECTION OF THE SIGNAL")
        print("=" * 88)
        print(f"  score as-is : AUC {fwd['auc']:.4f}")
        print(f"  score negated: AUC {inv['auc']:.4f}")
        ts = results["PAMPOS, two-sided"]
        if "error" not in ts:
            print(f"  two-sided    : AUC {ts['auc']:.4f}")
        if inv["auc"] > 0.6:
            print("  Attacked windows score LOWER than control windows. The attacks suppress")
            print("  motion, and a next-step predictor finds a stationary vehicle easier to")
            print("  predict than a moving one, so surprise falls rather than rises.")
        elif max(fwd["auc"], inv["auc"]) < 0.6:
            print("  Neither direction separates the two classes. The detector carries no")
            print("  usable signal about these attacks.")

    main_res = results["PAMPOS (5-feature)"]
    if "error" not in main_res:
        cm = main_res["confusion_matrix"]
        print()
        print(f"confusion at the calibrated threshold: "
              f"tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}")
        print(f"false-alarm rate on benign windows    : {main_res['false_alarm_rate']:.4f}")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    serialisable = {}
    for k, v in results.items():
        v = dict(v)
        v.pop("roc_curve", None)
        serialisable[k] = v
    with open(out / "detection_evaluation.json", "w") as f:
        json.dump({
            "scenarios": scenarios,
            "features": FEATURE_NAMES,
            "n_attacked": len(attacked_windows),
            "n_benign": len(benign_windows),
            "n_calibration": len(calib),
            "threshold": threshold,
            "results": serialisable,
        }, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/detection_evaluation.json")


if __name__ == "__main__":
    main()