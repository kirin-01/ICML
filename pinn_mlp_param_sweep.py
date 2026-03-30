import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_ground_truth(x, y):
    return 1.0 / (1.0 + x**2 + y**2)


def get_forcing_function(x, y):
    denom = 1.0 + x**2 + y**2
    return 4.0 * (1.0 - x**2 - y**2) / (denom ** 3)


class FlexibleMLP(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        layers = []
        in_dim = 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_training_points(n_interior=2000, n_boundary=400):
    x_in = (torch.rand(n_interior, 1, device=device) * 2 - 1).requires_grad_(True)
    y_in = (torch.rand(n_interior, 1, device=device) * 2 - 1).requires_grad_(True)

    x_bc = torch.rand(n_boundary, 1, device=device) * 2 - 1
    y_bc = torch.rand(n_boundary, 1, device=device) * 2 - 1
    mask = torch.rand(n_boundary, 1, device=device) > 0.5
    x_bc[mask] = torch.sign(x_bc[mask])
    y_bc[~mask] = torch.sign(y_bc[~mask])
    return x_in, y_in, x_bc, y_bc


def compute_loss(model, x_in, y_in, x_bc, y_bc):
    u_pred = model(x_in, y_in)
    grads = torch.autograd.grad(
        u_pred, [x_in, y_in], torch.ones_like(u_pred), create_graph=True
    )
    u_xx = torch.autograd.grad(
        grads[0], x_in, torch.ones_like(grads[0]), create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        grads[1], y_in, torch.ones_like(grads[1]), create_graph=True
    )[0]

    loss_pde = torch.mean((-(u_xx + u_yy) - get_forcing_function(x_in, y_in)) ** 2)
    loss_bc = torch.mean((model(x_bc, y_bc) - get_ground_truth(x_bc, y_bc)) ** 2)
    return loss_pde + loss_bc


def train_mlp(hidden_dims, total_epochs=2000, switch_epoch=500):
    model = FlexibleMLP(hidden_dims).to(device)
    param_count = count_params(model)
    print(f"\nTraining MLP {hidden_dims} | Parameters: {param_count}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        history_size=20,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    best_loss = float("inf")
    best_model_weights = None
    loss_history = []
    start_time = time.time()

    for epoch in range(total_epochs):
        x_in, y_in, x_bc, y_bc = sample_training_points()

        if epoch < switch_epoch:
            optimizer_adam.zero_grad()
            loss = compute_loss(model, x_in, y_in, x_bc, y_bc)
            loss.backward()
            optimizer_adam.step()
            current_loss = loss.item()
        else:
            def closure():
                optimizer_lbfgs.zero_grad()
                loss = compute_loss(model, x_in, y_in, x_bc, y_bc)
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)
            current_loss = compute_loss(model, x_in, y_in, x_bc, y_bc).item()

        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: {current_loss:.6f} (Best: {best_loss:.6f})")

    model.load_state_dict(best_model_weights)
    duration = time.time() - start_time
    print(f"Finished MLP {hidden_dims} in {duration:.2f}s | Best loss: {best_loss:.8f}")
    return {
        "hidden_dims": hidden_dims,
        "param_count": param_count,
        "loss_history": loss_history,
        "best_loss": best_loss,
        "duration": duration,
    }


if __name__ == "__main__":
    total_epochs = 2000
    switch_epoch = 500

    # Exact parameter counts for a 2D-input, 1D-output two-hidden-layer MLP.
    configs = {
        "20 params": [5],
        "70 params": [20],
        "100 params": [50,50],
        "1000 params": [13, 64],
    }

    results = {}
    for name, hidden_dims in configs.items():
        results[name] = train_mlp(
            hidden_dims,
            total_epochs=total_epochs,
            switch_epoch=switch_epoch,
        )

    print("\nSummary")
    print("-" * 72)
    print(f"{'Config':<12} {'Hidden Dims':<12} {'Params':<10} {'Best Loss':<14} {'Time (s)':<10}")
    print("-" * 72)
    for name, result in results.items():
        print(
            f"{name:<12} {str(result['hidden_dims']):<12} "
            f"{result['param_count']:<10} {result['best_loss']:<14.6e} {result['duration']:<10.2f}"
        )

    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.semilogy(result["loss_history"], label=f"{name} ({result['hidden_dims']})")

    plt.axvline(x=switch_epoch, color="green", linestyle=":", label="Switch to LBFGS")
    plt.title("MLP Parameter Sweep Under Unified Adam -> LBFGS Protocol")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig("mlp_param_sweep_adam_lbfgs.png", dpi=300, bbox_inches="tight")
    plt.show()
