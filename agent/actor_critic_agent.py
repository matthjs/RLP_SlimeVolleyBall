import torch
from collections import deque
from agent.agent import Agent
from loops.welford import finalizeAggregate
from policy_gradient.actor import PolicyNetwork
from policy_gradient.actor_mlp import PolicyNetworkMLP
from policy_gradient.critic import StateValueNetwork
from policy_gradient.critic_mlp import StateValueNetworkMLP
from policy_gradient.trainerAC import TrainerAC
from torch.distributions import Categorical

"""
Agent encapsulates one-step actor critic method
"""
class ActorCriticAgent(Agent):

    def __init__(self, learning_rate_actor, learning_rate_critic, gamma, usesImageData = True):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        """
        If set to true then use CNN to process image data, otherwise MLP is used
        """
        self.usesImageData = usesImageData
        if (self.usesImageData):
            self.actor = PolicyNetwork().to(self.device)
            self.critic = StateValueNetwork().to(self.device)
        else:
            self.actor = PolicyNetworkMLP().to(self.device)
            self.critic = StateValueNetworkMLP().to(self.device)

        self.trainer = TrainerAC(self.actor, self.critic, learning_rate_actor, learning_rate_critic, gamma)
        
        # loss information is accessible via the agent
        self.actor_avg_loss = []
        self.actor_var_loss = []
        self.critic_avg_loss = []
        self.critic_var_loss = []

        # memory of 1
        self.memory_buffer = deque([], maxlen = 1)

    def addTrajectory(self, trajectory):
        state, action, reward, nextState = trajectory

        state_t = self.__process_state__(state)
        action_t = torch.as_tensor(action, device = self.device, dtype = torch.int64)
        reward_t = torch.as_tensor(reward, device = self.device, dtype = torch.float32)
        nextState_t = self.__process_state__(nextState)
        
        self.memory_buffer.append((state_t, action_t, reward_t, nextState_t))
    
    def sampleTrajectory(self, batch_size = 1):
        return self.memory_buffer.pop()

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
    
    # assumes state is processed (!)
    def critic_value(self, state):
        return self.critic(state)

    # runs optimizer on model
    def train(self):
        self.trainer.train(self)
    
    # select action from probability vector (network approx policy)
    def policy(self, state):

        # initialize probability distribution to sample actions from
        probVec = self.actor(self.__process_state__(state))
        distribution = Categorical(probVec)
        action = distribution.sample()

        # return action
        return action.item()

    # assumes state is processed (!)
    def policy_log_prob(self, action, state):

        # initialize probability distribution to sample actions from
        probVec = self.actor(state)
        distribution = Categorical(probVec)

        return distribution.log_prob(action)

    # online update of (sample) variance and mean of the loss
    def recordLoss(self):
        meanLoss_actor, _, varianceLoss_actor = finalizeAggregate(self.trainer.policy_lossAggregate)
        meanLoss_critic, _, varianeLoss_critic = finalizeAggregate(self.trainer.stateval_lossAggregate)
        self.actor_avg_loss.append(meanLoss_actor)
        self.actor_var_loss.append(varianceLoss_actor)
        self.critic_avg_loss.append(meanLoss_critic)
        self.critic_var_loss.append(varianeLoss_critic)

    def getLoss(self):
        return [self.actor_avg_loss, self.critic_avg_loss], [self.actor_var_loss, self.critic_var_loss], ["Actor (Policy Function) Loss", "Critic (Value Function) MSE Loss"]

    def load_parameters(self):
        self.actor.load()
        self.critic.load()

    def save_parameters(self):
        self.actor.save()
        self.critic.save()