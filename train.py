
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

#  generate a small classification dataset
torch.manual_seed(42)

X, y = make_moons(
    n_samples=1000,
    noise=0.15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


# define our neural network
class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),  # Input -> Hidden
            nn.Tanh(),        # Activation
            nn.Linear(8, 1)   # Hidden -> Output
        )

    def forward(self, x):
        return self.network(x)


# initialize the model on CPU
model = MyMLP()

# define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train the network
for epoch in range(1000):
    model.train()

    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# evaluate the model
model.eval()

with torch.no_grad():
    logits = model(X_test)
    predicted = (torch.sigmoid(logits) >= 0.5).float()
    accuracy = (predicted == y_test).float().mean().item()

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Save our model
torch.save(model.state_dict(), "original.pt")

print("Model saved successfully!")


# Inspect its parameters
for name, param in model.named_parameters():
    print(f"\n{name}")
    print(param.data)
