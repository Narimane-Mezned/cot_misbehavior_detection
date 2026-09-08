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
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target_with_canonical_calibration
from src.attacks.adversarial_ml_attacks.knockoff_nets import KnockoffNetsAttack, pampos_target_query_func
from src.attacks.adversarial_ml_attacks.attribute_inference_black_box import (
    AttributeInferenceBlackBoxAttack, flatten_window_query_func,
)

KNOCKOFF_BUDGETS = [500, 200, 100, 50, 25, 10]
ATTRIBUTE_AUX_SIZES = [300, 150, 75, 30, 15, 5]


def sample_smooth_windows(n, seq_len, feature_mean, feature_std, step_scale=0.15):
    base = np.random.normal(feature_mean, feature_std, size=(n, 1, len(feature_mean)))
    steps = np.random.normal(0.0, feature_std * step_scale, size=(n, seq_len, len(feature_mean)))
    trajectories = base + np.cumsum(steps, axis=1)
    return trajectories.reshape(n, -1).astype(np.float32)


def sweep_knockoff(target, seq_len, n_features, val_windows):
    print("\n" + "=" * 60)
    print("KNOCKOFF NETS -- fidelity vs. query budget")
    print("=" * 60)

    input_dim = seq_len * n_features
    feature_mean = target.norm_mean.cpu().numpy()
    feature_std = target.norm_std.cpu().numpy()
    query_func = pampos_target_query_func(target, seq_len, n_features)
    x_eval = np.array([w.flatten() for w in val_windows[:100]])

    real_labels = np.array([1 if query_func(x) >= 0.5 else 0 for x in x_eval])
    n_anomalous = int(real_labels.sum())
    print(f"  [diagnostic] Real eval set: {n_anomalous}/{len(real_labels)} genuinely anomalous, "
          f"{len(real_labels) - n_anomalous}/{len(real_labels)} genuinely benign")
    if n_anomalous == 0:
        print(f"  [warning] Zero anomalous examples in eval set -- fidelity will be trivially inflated "
              f"by class imbalance. Interpreting fidelity alone here would be misleading.")

    results = []
    for budget in KNOCKOFF_BUDGETS:
        attack = KnockoffNetsAttack(input_dim=input_dim, output_dim=1, mode="classification", query_budget=budget, strategy="adaptive")
        attack._sample_candidates = lambda n: sample_smooth_windows(n, seq_len, feature_mean, feature_std)
        attack.extract(query_func, x_eval=x_eval, target_eval_func=query_func)

        surrogate_preds = attack.predict(x_eval)
        surrogate_labels = (surrogate_preds[:, 0] >= 0.5).astype(int)

        if n_anomalous > 0:
            anomalous_mask = real_labels == 1
            recall_on_anomalous = float(np.mean(surrogate_labels[anomalous_mask] == 1))
        else:
            recall_on_anomalous = None

        benign_mask = real_labels == 0
        recall_on_benign = float(np.mean(surrogate_labels[benign_mask] == 0)) if benign_mask.sum() > 0 else None

        always_benign_fidelity = (len(real_labels) - n_anomalous) / len(real_labels)

        print(f"  budget={budget:>4} -> fidelity={attack.final_fidelity:.3f} | "
              f"recall_on_anomalous={recall_on_anomalous if recall_on_anomalous is not None else 'N/A'} | "
              f"recall_on_benign={recall_on_benign if recall_on_benign is not None else 'N/A'} | "
              f"(a naive always-benign guesser would score {always_benign_fidelity:.3f} fidelity for free)")

        results.append({
            "query_budget": budget, "fidelity": attack.final_fidelity,
            "recall_on_anomalous": recall_on_anomalous, "recall_on_benign": recall_on_benign,
            "always_benign_baseline_fidelity": always_benign_fidelity,
        })

    return results


def sweep_attribute_inference(target, seq_len, n_features, train_windows, val_windows):
    print("\n" + "=" * 60)
    print("ATTRIBUTE INFERENCE -- accuracy vs. auxiliary query budget")
    print("=" * 60)

    input_dim = seq_len * n_features
    attribute_index = (seq_len - 1) * n_features + 6
    query_func = flatten_window_query_func(target, seq_len, n_features)
    eval_states = np.array([w.flatten() for w in val_windows[:100]])

    results = []
    for aux_size in ATTRIBUTE_AUX_SIZES:
        aux_states = np.array([w.flatten() for w in train_windows[:aux_size]])
        attack = AttributeInferenceBlackBoxAttack(input_dim=input_dim, attribute_index=attribute_index, attribute_values=[0.0, 1.0])
        attack.fit(aux_states, query_func)
        eval_results = attack.evaluate_accuracy(eval_states, query_func)
        print(f"  aux_queries={aux_size:>4} -> accuracy={eval_results['accuracy']:.3f}, advantage={eval_results['advantage_over_baseline']:+.3f}")
        results.append({"aux_queries": aux_size, "accuracy": eval_results["accuracy"], "advantage": eval_results["advantage_over_baseline"]})

    return results


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    n_features = config["model"]["input_dim"]
    model_config = {
        "input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    print("[setup] Loading trained PAMPOS checkpoint with canonical calibration...")
    target = load_pampos_target_with_canonical_calibration(REPO_ROOT, model_config)
    print(f"[setup] Canonical threshold: {target.threshold:.4f}")

    print("[setup] Rebuilding real train/val split (for train/val window access elsewhere in this script)...")
    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]
    val_windows = [full_dataset.sequences[i] for i in val_subset.indices]

    knockoff_results = sweep_knockoff(target, seq_len, n_features, val_windows)
    attribute_results = sweep_attribute_inference(target, seq_len, n_features, train_windows, val_windows)

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    safe_knockoff = [r for r in knockoff_results if r["fidelity"] < 0.7]
    safe_attribute = [r for r in attribute_results if r["advantage"] < 0.1]
    if safe_knockoff:
        print(f"Knockoff fidelity drops below 0.7 at query_budget <= {max(r['query_budget'] for r in safe_knockoff)}")
    else:
        print("Knockoff fidelity never dropped below 0.7 in this sweep -- even 10 queries may be enough. Consider this a real limitation.")
    if safe_attribute:
        print(f"Attribute inference advantage drops below 0.1 at aux_queries <= {max(r['aux_queries'] for r in safe_attribute)}")
    else:
        print("Attribute inference advantage never dropped below 0.1 in this sweep -- this attack may be hard to mitigate via query limiting alone.")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "query_budget_sweep.json"
    with open(output_path, "w") as f:
        json.dump({"knockoff_nets": knockoff_results, "attribute_inference": attribute_results}, f, indent=2, default=str)

    print(f"\n[done] Saved results to {output_path}")


if __name__ == "__main__":
    main()