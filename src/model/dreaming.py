import torch


@torch.no_grad()
def dream_rollout(model, seed_seq: torch.Tensor, horizon: int) -> torch.Tensor:
    model.eval()
    current_seq = seed_seq.clone()
    imagined_steps = []

    for _ in range(horizon):
        preds = model(current_seq)
        next_step = preds[:, -1:, :]
        imagined_steps.append(next_step)
        current_seq = torch.cat([current_seq, next_step], dim=1)

    return torch.cat(imagined_steps, dim=1)


@torch.no_grad()
def measure_error_accumulation(
    model,
    seed_seq: torch.Tensor,
    ground_truth_continuation: torch.Tensor,
    max_horizon: int,
) -> torch.Tensor:
    imagined = dream_rollout(model, seed_seq, max_horizon)
    per_step_error = torch.abs(imagined - ground_truth_continuation).mean(dim=(0, 2))
    return per_step_error


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pampos import PAMPOS

    torch.manual_seed(42)

    model = PAMPOS()
    model.eval()

    batch_size = 8
    seed_len = 10
    max_horizon = 5
    input_dim = 8

    full_seq = torch.randn(batch_size, seed_len + max_horizon, input_dim)
    seed_seq = full_seq[:, :seed_len, :]
    ground_truth_continuation = full_seq[:, seed_len:, :]

    imagined = dream_rollout(model, seed_seq, max_horizon)
    print(f"Seed shape: {tuple(seed_seq.shape)}")
    print(f"Imagined shape: {tuple(imagined.shape)}")

    per_step_error = measure_error_accumulation(model, seed_seq, ground_truth_continuation, max_horizon)
    print("\nError accumulation over horizon (untrained model, expect roughly flat/random):")
    for step, err in enumerate(per_step_error, start=1):
        print(f"  step {step}: mean abs error = {err.item():.4f}")