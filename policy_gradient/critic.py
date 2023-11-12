import torch
import torch.nn as nn

"""
nn.Conv2d: 
    in_channel: RGB -> 3, grayscale -> 1
    out_channel: depth
    kernel_size: in 2d case a square matrix
    stride: how far does the kernel move over img
    padding: implicitly zero padding input
    Pooling: replace output of net at certain location 
        with summary statistic of the nearby outputs. 
        Helps make representation become approximately 
        invariant to small translations of the input.
"""
class StateValueNetwork(nn.Module):
    
    def __init__(self):
        super(StateValueNetwork, self).__init__()
        """
        Input: (N, 4, 84, 84) tensor where
            N is the batch size
            4 is the channel depth (4 frames of the game are provided)
            84 x 84 is the width and height of the actual grayscale image
        """
        self.cnn_layers = nn.Sequential(

            # input = 4 x 84 x 84, Output = 32 x 20 x 20
            torch.nn.Conv2d(4, 32, kernel_size = 8, stride = 4),
            # input = 32 x 20 x 20, Output = 64 x 9 x 9
            torch.nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            # input = 64 x 9 x 9, Output = 64 x 7 x 7
            torch.nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
        )

        # group together fully connected layers
        # two fully connected layers for regression
        # difference with CNN -> final layer is single value
        self.linear_layers = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )
    
    # defines how the input is passed through the network
    # returns value for given state
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x
    
    def save(self, file_name = './params/modelCRITIC.pth'):
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name = './params/modelCRITIC.pth'):
        self.load_state_dict(torch.load(file_name))