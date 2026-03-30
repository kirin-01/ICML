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


def get_activation(name):
    if name.lower() == "tanh":
        return nn.Tanh
    if name.lower() == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation: {name}")


class FlexibleMLP(nn.Module):
    def __init__(self, layer_dims, activation_name):
        super().__init__()
        activation_cls = get_activation(activation_name)
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-2], layer_dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_cls())
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
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


def compute_pinn_loss(model, x_in, y_in, x_bc, y_bc):
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


@torch.no_grad()
def evaluate_solution_mse(model, grid_size=201):
    xs = torch.linspace(-1.0, 1.0, grid_size, device=device)
    ys = torch.linspace(-1.0, 1.0, grid_size, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.reshape(-1, 1)
    y_flat = yy.reshape(-1, 1)
    pred = model(x_flat, y_flat)
    target = get_ground_truth(x_flat, y_flat)
    return torch.mean((pred - target) ** 2).item()


def train_model(config, optimizer_mode="adam", total_epochs=2000, switch_epoch=500, lr=1e-2):
    model = FlexibleMLP(config["layer_dims"], config["activation"]).to(device)
    param_count = count_params(model)
    active_adam_epochs = total_epochs if optimizer_mode == "adam" else switch_epoch

    print(
        f"\nTraining {config['name']} | Activation: {config['activation']} | "
        f"Params: {param_count} | Optimizer: {optimizer_mode}"
    )

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=active_adam_epochs, eta_min=lr * 0.01
    )
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
        use_lbfgs = optimizer_mode == "hybrid" and epoch >= switch_epoch

        if use_lbfgs:
            def closure():
                optimizer_lbfgs.zero_grad()
                loss = compute_pinn_loss(model, x_in, y_in, x_bc, y_bc)
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)
            current_loss = compute_pinn_loss(model, x_in, y_in, x_bc, y_bc).item()
        else:
            optimizer_adam.zero_grad()
            loss = compute_pinn_loss(model, x_in, y_in, x_bc, y_bc)
            loss.backward()
            optimizer_adam.step()
            scheduler.step()
            current_loss = loss.item()

        loss_history.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if epoch % 100 == 0:
            current_lr = optimizer_adam.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: loss={current_loss:.6e} | "
                f"best={best_loss:.6e} | adam_lr={current_lr:.3e}"
            )

    model.load_state_dict(best_model_weights)
    duration = time.time() - start_time
    solution_mse = evaluate_solution_mse(model)

    print(
        f"Finished {config['name']} in {duration:.2f}s | "
        f"Best train loss: {best_loss:.6e} | Solution MSE: {solution_mse:.6e}"
    )

    return {
        "name": config["name"],
        "layer_dims": config["layer_dims"],
        "activation": config["activation"],
        "param_count": param_count,
        "best_train_loss": best_loss,
        "solution_mse": solution_mse,
        "duration": duration,
        "loss_history": loss_history,
    }


def plot_results(results, protocol_name, switch_epoch):
    plt.figure(figsize=(10, 6))
    for result in results:
        label = (
            f"{result['name']} | {result['activation']} | "
            f"{result['param_count']} params"
        )
        plt.semilogy(result["loss_history"], label=label)

    if protocol_name == "hybrid":
        plt.axvline(x=switch_epoch, color="green", linestyle=":", label="Switch to LBFGS")

    plt.title(f"Reviewer-style MLP Comparison ({protocol_name})")
    plt.xlabel("Epochs")
    plt.ylabel("PINN Loss (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"reviewer_mlp_protocol_{protocol_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    total_epochs = 2000
    switch_epoch = 500
    lr = 1e-2

    reviewer_configs = [
        {"name": "MLP-[2,5,1]", "layer_dims": [2, 5, 1], "activation": "tanh"},
        {"name": "MLP-[2,20,1]", "layer_dims": [2, 20, 1], "activation": "tanh"},
        {"name": "MLP-[2,50,50,1]-Tanh", "layer_dims": [2, 50, 50, 1], "activation": "tanh"},
        {"name": "MLP-[2,50,50,1]-GELU", "layer_dims": [2, 50, 50, 1], "activation": "gelu"},
    ]

    # To reproduce the numbers the reviewer quoted, start with Adam + cosine.
    # Change to ["adam", "hybrid"] if you also want the switched-optimizer variant.
    protocols_to_run = ["adam"]

    for protocol_name in protocols_to_run:
        results = []
        for config in reviewer_configs:
            results.append(
                train_model(
                    config,
                    optimizer_mode=protocol_name,
                    total_epochs=total_epochs,
                    switch_epoch=switch_epoch,
                    lr=lr,
                )
            )

        print("\nSummary")
        print("-" * 108)
        print(
            f"{'Model':<24} {'Activation':<10} {'Params':<8} "
            f"{'Best Train Loss':<18} {'Solution MSE':<18} {'Time (s)':<10}"
        )
        print("-" * 108)
        for result in results:
            print(
                f"{result['name']:<24} {result['activation']:<10} {result['param_count']:<8} "
                f"{result['best_train_loss']:<18.6e} {result['solution_mse']:<18.6e} "
                f"{result['duration']:<10.2f}"
            )

        plot_results(results, protocol_name, switch_epoch)
