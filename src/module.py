import combined_sequence as cs
import dummy_data_generation as ddg
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchinfo

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device} device")


class CustomRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden_size = 2
        self.combined_sequence = cs.CombinedSequence(input_size, output_size)
        
    def forward(self, x):
        x = self.combined_sequence(x)
        
        return x


class FullyConnectedRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )
        
    def forward(self, x):
        return self.linear_relu_stack(x)


def train_test_split(x_tensor, y_tensor, test_size=0.2):
    indices = torch.randperm(len(x_tensor))
    split_idx = int(len(x_tensor) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return x_tensor[train_indices], y_tensor[train_indices], x_tensor[test_indices], y_tensor[test_indices]


def train_model(model, x_train, y_train, epochs=1000, learning_rate=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    x_train, y_train, x_vali, y_vali = train_test_split(x_train, y_train)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % (epochs / 20) == 0 or epoch == epochs - 1:
            y_vali_pred = model(x_vali)
            loss_test = loss_fn(y_vali_pred, y_vali)
            print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}, Validation Loss: {loss_test.item():.4f}")


def plot_results(model, x_tensor, y_tensor, title):
    model.eval()
    with torch.no_grad():
        # Sort the data by x values for proper line plotting
        sorted_indices = torch.argsort(x_tensor.squeeze())
        x_sorted = x_tensor[sorted_indices]
        y_sorted = y_tensor[sorted_indices]
        
        y_pred = model(x_sorted)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_sorted, y_sorted, label='True data', color='blue', alpha=0.6, s=20)
        plt.plot(x_sorted, y_pred, label='Predicted data', color='red', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)



# Generate the data
x, y, y_true = ddg.generate_dummy_data(ddg.quadratic, n_samples=500, noise_std=0.2)

x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

model_fullyconnected = FullyConnectedRegression(input_size=1, output_size=1)
model_custom = CustomRegression(input_size=1, output_size=1)

x_train, y_train, x_test, y_test = train_test_split(x_tensor, y_tensor)

train_model(model_fullyconnected, x_train, y_train, epochs=5000, learning_rate=0.01)
train_model(model_custom, x_train, y_train, epochs=5000, learning_rate=0.001)

plot_results(model_fullyconnected, x_test, y_test, 'Fully Connected NN Results')
plot_results(model_custom, x_test, y_test, 'Custom Activation NN Results')

torchinfo.summary(model_fullyconnected, input_data=x_test, device='cpu')
torchinfo.summary(model_custom, input_data=x_test, device='cpu')

for name, param in model_custom.named_parameters():
    if param.requires_grad:
        print(name, param.data)

input_weights = model_custom.combined_sequence.linear1.weight.data.reshape(3, 1)        
output_weights = model_custom.combined_sequence.linear2.weight.data.reshape(3, 1)
output_weights[0] = input_weights[0]**2 * output_weights[0]
function_logits = output_weights
function_probs = torch.nn.functional.softmax(function_logits, dim=0)
 
print(function_logits)
print(function_probs)
          
plt.show()