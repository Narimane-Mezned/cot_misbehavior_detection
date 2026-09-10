import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.losses import huber_loss, per_feature_errors, normalize_errors, topk_anomaly_score
from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset
from src.attacks.adversarial_ml_attacks.attribute_inference_black_box import (
    AttributeInferenceBlackBoxAttack,
)

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]
CAMERA_IDX = 6
EPOCHS = 60


class WindowList(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i])


def normalise(windows):
    stacked = np.concatenate(windows, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return [(w - mean) / std for w in windows], mean, std


def train_model(train_windows, n_features, device, seed=42):
    torch.manual_seed(seed)
    model = PAMPOS(input_dim=n_features, d_model=128, nhead=8, num_layers=3,
                   dim_feedforward=256, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loader = DataLoader(WindowList(train_windows), batch_size=32, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            batch = batch.to(device).float()
            preds = model(batch[:, :-1, :])
            loss = huber_loss(preds, batch[:, 1:, :], delta=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"      epoch {epoch+1}/{EPOCHS}  loss {total/len(loader):.5f}")
    model.eval()
    return model


class SimpleTarget:
    """Queryable wrapper. Accepts RAW windows and normalises internally, matching
    the real pipeline -- passing pre-normalised windows would destroy the 0/1
    values of binary features that the attribute attack needs."""

    def __init__(self, model, device, mean, std):
        self.model = model
        self.device = device
        self.mean = mean
        self.std = std
        self.feature_mae = None
        self.threshold = None

    @torch.no_grad()
    def _errors(self, window):
        w = (np.asarray(window, dtype=np.float32) - self.mean) / self.std
        x = torch.from_numpy(w.astype(np.float32)).to(self.device).unsqueeze(0)
        preds = self.model(x[:, :-1, :])
        e = per_feature_errors(preds, x[:, 1:, :])
        if self.feature_mae is not None:
            e = normalize_errors(e, self.feature_mae)
        return e

    def raw_score(self, window):
        return topk_anomaly_score(self._errors(window), k=3).mean().item()

    def calibrate(self, windows, percentile=99.0):
        errs = [self._errors(w) for w in windows]
        self.feature_mae = torch.cat(errs, dim=0).mean(dim=(0, 1))
        scores = [self.raw_score(w) for w in windows]
        self.threshold = float(np.percentile(scores, percentile))

    def predict_proba(self, window, scale=2.0):
        z = (self.raw_score(window) - self.threshold) / (scale + 1e-8)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.array([1.0 - p, p])


def run_attribute_inference(target, train_raw, val_raw, raw_train, raw_val,
                            n_features, seq_len, camera_present):
    """
    The sensitive attribute is always is_camera_visible taken from the RAW data.
    When camera_present is False the model never saw it, so any residual
    inference must come from natural correlation with the remaining features.
    """
    def flat(ws):
        return np.array([w.flatten() for w in ws])

    aux = flat(train_raw[:300])
    ev = flat(val_raw[:100])

    def query(flat_state):
        window = flat_state.reshape(seq_len, n_features).astype(np.float32)
        return target.predict_proba(window)[1]

    if camera_present:
        attr_idx = (seq_len - 1) * n_features + CAMERA_IDX
        attack = AttributeInferenceBlackBoxAttack(
            input_dim=seq_len * n_features, attribute_index=attr_idx,
            attribute_values=[0.0, 1.0])
        attack.fit(aux, query)
        return attack.evaluate_accuracy(ev, query)

    # camera feature absent from the model input: append the raw label as an
    # extra column so the attack has a target to predict, and query using only
    # the real 7-feature window.
    aux_lab = np.array([raw_train[i][-1, CAMERA_IDX] for i in range(min(300, len(raw_train)))])
    ev_lab = np.array([raw_val[i][-1, CAMERA_IDX] for i in range(min(100, len(raw_val)))])

    aux_aug = np.hstack([aux, aux_lab.reshape(-1, 1)])
    ev_aug = np.hstack([ev, ev_lab.reshape(-1, 1)])

    def query_aug(flat_state):
        window = flat_state[:-1].reshape(seq_len, n_features).astype(np.float32)
        return target.predict_proba(window)[1]

    attack = AttributeInferenceBlackBoxAttack(
        input_dim=seq_len * n_features + 1,
        attribute_index=seq_len * n_features,
        attribute_values=[0.0, 1.0])
    attack.fit(aux_aug, query_aug)
    return attack.evaluate_accuracy(ev_aug, query_aug)


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device: {device}")

    data_root = REPO_ROOT / config["data"]["raw_dir"]
    ds = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    gen = torch.Generator().manual_seed(config["training"]["seed"])
    val_size = max(1, int(len(ds) * config["training"]["val_fraction"]))
    tr_sub, va_sub = random_split(ds, [len(ds) - val_size, val_size], generator=gen)
    raw_train = [ds.sequences[i] for i in tr_sub.indices]
    raw_val = [ds.sequences[i] for i in va_sub.indices]
    print(f"[setup] {len(raw_train)} train / {len(raw_val)} val windows\n")

    results = {}

    for label, keep_camera in [("8-feature (baseline)", True), ("7-feature (camera removed)", False)]:
        print("=" * 74)
        print(label)
        print("=" * 74)

        if keep_camera:
            tr = [w.copy() for w in raw_train]
            va = [w.copy() for w in raw_val]
        else:
            tr = [np.delete(w, CAMERA_IDX, axis=1) for w in raw_train]
            va = [np.delete(w, CAMERA_IDX, axis=1) for w in raw_val]

        n_features = tr[0].shape[1]
        tr_norm, mean, std = normalise(tr)

        print(f"   training {n_features}-feature model...")
        model = train_model(tr_norm, n_features, device)

        target = SimpleTarget(model, device, mean, std)
        target.calibrate(tr[:300])
        scores = np.array([target.raw_score(w) for w in va])
        flagged = int((scores > target.threshold).sum())
        print(f"   threshold {target.threshold:.4f} | flagged {flagged}/{len(va)} val windows")

        print(f"   running attribute inference...")
        ai = run_attribute_inference(target, tr, va, raw_train, raw_val,
                                     n_features, seq_len, keep_camera)
        print(f"   accuracy {ai['accuracy']:.3f} | baseline {ai['baseline_accuracy']:.3f} "
              f"| advantage {ai['advantage_over_baseline']:+.3f}\n")

        results[label] = {
            "n_features": n_features,
            "threshold": target.threshold,
            "flagged_val": flagged,
            "n_val": len(va),
            "attribute_inference": ai,
        }

    print("=" * 74)
    print("COMPARISON")
    print("=" * 74)
    a = results["8-feature (baseline)"]
    b = results["7-feature (camera removed)"]
    print(f"{'':<34}{'8-feature':>18}{'7-feature':>18}")
    print(f"{'attribute inference accuracy':<34}{a['attribute_inference']['accuracy']:>18.3f}"
          f"{b['attribute_inference']['accuracy']:>18.3f}")
    print(f"{'advantage over baseline':<34}{a['attribute_inference']['advantage_over_baseline']:>+18.3f}"
          f"{b['attribute_inference']['advantage_over_baseline']:>+18.3f}")
    print(f"{'calibrated threshold':<34}{a['threshold']:>18.4f}{b['threshold']:>18.4f}")
    print(f"{'flagged val windows':<34}{a['flagged_val']:>18}{b['flagged_val']:>18}")

    print("\nInterpretation:")
    print("  A large drop in advantage means removing the feature closes the leak.")
    print("  A similar flagged count means the detector still separates the same windows,")
    print("  though true detection cost cannot be measured without attack-labelled data.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "attribute_inference_mitigation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/attribute_inference_mitigation.json")


if __name__ == "__main__":
    main()