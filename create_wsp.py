
import json
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# Recreate the exact MLP architecture
class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


# Load your trained model
model = MyMLP()
model.load_state_dict(
    torch.load("original.pt", map_location="cpu", weights_only=True)
)
model.eval()

# Generate new validation examples
X, y = make_moons(
    n_samples=400,
    noise=0.15,
    random_state=123
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


def evaluate(model):
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()

    return accuracy, probabilities, predictions, logits


# Evaluate original model
original_accuracy, _, _, _ = evaluate(model)

print(f"Original accuracy: {original_accuracy:.2%}")


# Intentionally corrupt ONE parameter
with torch.no_grad():
    output_layer = model.network[2]

    # Force the output logit to be positive
    # for all possible hidden activations.
    corrupted_bias = output_layer.weight.abs().sum() + 2.0
    output_layer.bias.copy_(corrupted_bias.reshape_as(output_layer.bias))


# Save corrupted model
torch.save(model.state_dict(), "corrupted.pt")


# Evaluate corrupted model
corrupted_accuracy, probabilities, predictions, logits = evaluate(model)

print(f"Corrupted accuracy: {corrupted_accuracy:.2%}")


# Export all corrupted weights
weights = {}

for name, parameter in model.named_parameters():
    weights[name] = parameter.detach().tolist()


# Include 10 examples from each class
indices_0 = (y[:, 0] == 0).nonzero(as_tuple=True)[0][:10]
indices_1 = (y[:, 0] == 1).nonzero(as_tuple=True)[0][:10]

examples = []

for idx in torch.cat([indices_0, indices_1]).tolist():
    examples.append({
        "input": X[idx].tolist(),
        "expected_label": int(y[idx].item()),
        "output_logit": float(logits[idx].item()),
        "predicted_probability": float(probabilities[idx].item()),
        "predicted_label": int(predictions[idx].item())
    })


export = {
    "architecture": "Linear(2,8) -> Tanh -> Linear(8,1)",
    "parameter_count": sum(p.numel() for p in model.parameters()),
    "output": "Raw logit; sigmoid threshold 0.5",
    "validation_accuracy": corrupted_accuracy,
    "validation_sample_count": len(X),
    "weights": weights,
    "examples": examples
}

with open("weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("Saved corrupted.pt and weights.json")
