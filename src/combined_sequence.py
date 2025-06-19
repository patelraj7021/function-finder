import torch

class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x**2
    
class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)
    
class Exponential(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.exp(x)

class CombinedSequence(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.activation_functions = [Quadratic(), Sin(), Exponential()]
        hidden_size = len(self.activation_functions)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        activation_outputs = []
        for i, activation_function in enumerate(self.activation_functions):
            activation_outputs.append(activation_function(x[:, i:i+1]))
        x = torch.cat(activation_outputs, dim=1)
        x = self.linear2(x)
        return x