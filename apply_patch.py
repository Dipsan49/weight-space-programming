
import json
import math
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# Recreate the original MLP architecture


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


#Load a saved model

def load_model(filename):
    model = MyMLP()

    state_dict = torch.load(
        filename,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


original_model = load_model("original.pt")
corrupted_model = load_model("corrupted.pt")


#Read ChatGPT's JSON repair patch

with open("repair.patch.json", "r") as f:
    patch = json.load(f)

# Convert JSON Patch operations into our internal format
edits = []

for operation in patch:
    if operation["op"] != "replace":
        raise ValueError("Only 'replace' operations are supported.")

    # Example: /weights/network.2.bias/0
    parts = operation["path"].strip("/").split("/")

    if len(parts) < 3 or parts[0] != "weights":
        raise ValueError(f"Invalid patch path: {operation['path']}")

    tensor_name = parts[1]
    indices = [int(i) for i in parts[2:]]

    edits.append({
        "tensor": tensor_name,
        "index": indices,
        "value": operation["value"]
    })

print("Loaded patch:", edits)


if not isinstance(edits, list) or not edits:
    raise ValueError("Patch must contain a non-empty 'edits' list.")


#Apply proposed weight modifications

# Start from the corrupted model
repaired_model = load_model("corrupted.pt")

parameters = dict(repaired_model.named_parameters())

with torch.no_grad():
    for edit in edits:
        tensor_name = edit["tensor"]
        index = edit["index"]
        new_value = edit["value"]

        if tensor_name not in parameters:
            raise ValueError(f"Unknown parameter: {tensor_name}")

        parameter = parameters[tensor_name]

        if not isinstance(index, list):
            raise ValueError("Index must be a list.")

        if len(index) != parameter.ndim:
            raise ValueError(
                f"Expected {parameter.ndim} indices "
                f"for {tensor_name}, got {len(index)}"
            )

        for i, size in zip(index, parameter.shape):
            if type(i) is not int or not 0 <= i < size:
                raise ValueError(
                    f"Invalid index {index} for {tensor_name}"
                )

        if (
            isinstance(new_value, bool)
            or not isinstance(new_value, (int, float))
            or not math.isfinite(new_value)
        ):
            raise ValueError("Replacement value must be finite.")

        tensor_index = tuple(index)

        old_value = parameter[tensor_index].item()

        parameter[tensor_index] = float(new_value)

        print(f"\nModified: {tensor_name}")
        print(f"Index: {index}")
        print(f"Old value: {old_value:.6f}")
        print(f"New value: {parameter[tensor_index].item():.6f}")


# recreate the validation dataset

X, y = make_moons(
    n_samples=400,
    noise=0.15,
    random_state=123
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


# Evaluate classification accuracy

def evaluate(model):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        accuracy = (
            (predictions == y).float().mean().item()
        )

    return accuracy


original_accuracy = evaluate(original_model)
corrupted_accuracy = evaluate(corrupted_model)
repaired_accuracy = evaluate(repaired_model)


# Compare all three models

print("\n========== WSP RESULTS ==========")

print(f"Original Accuracy:  {original_accuracy:.2%}")
print(f"Corrupted Accuracy: {corrupted_accuracy:.2%}")
print(f"Repaired Accuracy:  {repaired_accuracy:.2%}")

improvement = (
    repaired_accuracy - corrupted_accuracy
) * 100

print(f"\nAccuracy improvement: {improvement:+.2f} percentage points")

# save repaired model

torch.save(
    repaired_model.state_dict(),
    "repaired.pt"
)

print("\nRepaired model saved as repaired.pt")
