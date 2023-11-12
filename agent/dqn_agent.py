import torch
import random
import numpy as np
from collections import deque
from agent.agent import Agent
from dqn.mlp import MLP
from dqn.trainerDQN import TrainerDQN
from dqn.dqnCNN import CNN
from loops.welford import finalizeAggregate


class DQNAgent(Agent):

    # the last three parameters are for the trainer
    def __init__(self, epsilon, param_copying, schedule, batch_size, learning_rate, gamma, usesImageData = True):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        """
        If set to true then use CNN to process image data, otherwise MLP is used
        """
        self.usesImageData = usesImageData
        if (self.usesImageData):
            self.model = CNN().to(self.device)
            self.model_TDtarget = CNN().to(self.device)
        else:
            self.model = MLP().to(self.device)
            self.model_TDtarget = MLP().to(self.device)

        # copy over parameters
        self.model_TDtarget.load_state_dict(self.model.state_dict())

        self.param_copying = param_copying
        self.train_count = 0

        target, end = schedule
        self.epsilon = epsilon
        self.epsilon_target = target
        self.epsilon_decrease = (1 - target) / end
        
        self.trainer = TrainerDQN(self.model, batch_size, learning_rate, gamma)

        self.avg_loss = []
        self.var_loss = []

        self.memory_buffer = deque([], maxlen = 10000)

    # converts to tensors
    def addTrajectory(self, trajectory):
        state, action, reward, nextState = trajectory

        state_t = self.__process_state__(state)
        action_t = torch.as_tensor(action, device = self.device, dtype = torch.int64)
        reward_t = torch.as_tensor(reward, device = self.device, dtype = torch.float32)
        nextState_t = self.__process_state__(nextState)
        
        self.memory_buffer.append((state_t, action_t, reward_t, nextState_t))

    def sampleTrajectory(self, batch_size):
        return random.sample(self.memory_buffer, batch_size)

    """
    change (1 x 84 x 84 x 4) tensor to (1 x 4 x 84 x 84) tensor
    so that the "4" is interpreted as channel depth in the
    network.
    """
    def __process_state__(self, state):
        tensor = torch.as_tensor(state, device = self.device, dtype = torch.float32)
        if (self.usesImageData):
            tensor = tensor.view(1, 4, 84, 84)
        else:
            tensor = tensor.view(1, 12)

        return tensor
    
    # runs optimizer on model
    def train(self):
        # copy weights over to target network
        if (self.train_count == self.param_copying):
            self.train_count = 0
            self.model_TDtarget.load_state_dict(self.model.state_dict())

        # linearly decrease epsilon up to target epsilon
        if (self.getGameInfo().done and self.epsilon > self.epsilon_target):
            self.epsilon = self.epsilon - self.epsilon_decrease
            if (self.epsilon < self.epsilon_target):
                self.epsilon = 0.1

        self.trainer.train(self)
        self.train_count += 1

    # neural network approximates Q function
    # agent follows epsilon greedy policy
    # assume batch input
    def policy(self, state):

        action = None
        model_output = None

        if (np.random.uniform(0, 100) >= self.epsilon * 100):
            # choose best action according to Q
            model_output = self.model(self.__process_state__(state))
            action = torch.argmax(model_output).item()

        else:
            action = random.randint(0, 5)

        return action

    """
    save current estimate of average reward and variance reward
    """
    def recordLoss(self):
        meanLoss, _, varianceLoss = finalizeAggregate(self.trainer.lossAggregate)
        self.avg_loss.append(meanLoss)
        self.var_loss.append(varianceLoss)

    def getLoss(self):
        return [self.avg_loss], [self.var_loss], ["MSE loss"]

    def load_parameters(self):
        self.model.load()

    def save_parameters(self):
        self.model.save()



