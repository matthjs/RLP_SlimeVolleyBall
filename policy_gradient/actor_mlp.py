import torch
import torch.nn as nn

class PolicyNetworkMLP(nn.Module):
    
    def __init__(self):
        super(PolicyNetworkMLP, self).__init__()
        """
        Input: N (12,) vectors where
            N is the batch size
        """
        # sequential groups together layers in one function call
        # group together fully connected layers
        self.linear_layers = nn.Sequential(
            torch.nn.Linear(12, 128),   # was 3200, 256
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 6),
            torch.nn.Softmax(dim = 1)
        )
    
    # defines how the input is passed through the network
    def forward(self, x):
        x = self.linear_layers(x)
        return x
    
    def save(self, file_name = './params/modelACTOR_MLP.pth'):
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name = './params/modelACTOR_MLP.pth'):
        self.load_state_dict(torch.load(file_name))