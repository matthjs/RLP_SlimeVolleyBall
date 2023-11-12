import torch
from collections import deque
from agent.agent import Agent
from loops.welford import finalizeAggregate
from policy_gradient.actor import PolicyNetwork
from policy_gradient.actor_mlp import PolicyNetworkMLP
from policy_gradient.critic import StateValueNetwork
from policy_gradient.critic_mlp import StateValueNetworkMLP
from torch.distributions import Categorical

from policy_gradient.trainerREINFORCE import TrainerREINFORCE

"""
Agent encapsulates REINFORCE and REINFORCE with baseline
method. We want to take advantage of polymorphism in main
to prevent code duplication. This requires reinforce to have
a similar interface to the Temporal Difference agents. To allow
this the train method is ajusted to not let the agent train
(i.e. update the parameters of the models) until the episode is
finished.
"""
class REINFORCEAgent(Agent):

    """
    A bit problematic, normally the trainers have access to the gamma
    hyperparameter, but in this case the agent needs to know about it.
    gamma in the agent class is discount_reward
    """
    def __init__(self, BASELINE, discount_reward, learning_rate_actor, learning_rate_critic, gamma, usesImageData = True):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.usesImageData = usesImageData
        if (self.usesImageData):
            self.actor = PolicyNetwork().to(self.device)
        else:
            self.actor = PolicyNetworkMLP().to(self.device)

        self.BASELINE = BASELINE
        if (BASELINE):
            if (self.usesImageData):
                self.critic = StateValueNetwork().to(self.device)
            else:
                self.critic = StateValueNetworkMLP().to(self.device)
        else:
            self.critic = None

        self.trainer = TrainerREINFORCE(self.actor, self.critic, learning_rate_actor, learning_rate_critic, gamma)
        
        self.actor_avg_loss = []
        self.actor_var_loss = []

        if (BASELINE):
            self.critic_avg_loss = []
            self.critic_var_loss = []
        else:
            self.critic_avg_loss = None
            self.critic_var_loss = None

        self.discount_reward = discount_reward
        self.discounted_return = 0.0

        # memory of 1
        self.memory_buffer = deque([], maxlen = 1)
    
    def recordReward(self, reward):
        self.discounted_return += self.discount_reward * reward

    """
    change (N x 84 x 84 x 4) tensor to (N x 4 x 84 x 84) tensor
    so that the "4" is interpreted as channel depth in the
    network.
    Also converts scalar action and reward to tensor object
    """
    def addTrajectory(self, trajectory):
        state, action, reward, nextState = trajectory

        state_t = self.__process_state__(state)
        action_t = torch.as_tensor(action, device = self.device, dtype = torch.int64)
        reward_t = torch.as_tensor(reward, device = self.device, dtype = torch.float32)
        nextState_t = self.__process_state__(nextState)
        
        self.memory_buffer.append((state_t, action_t, reward_t, nextState_t))
    
    def sampleTrajectory(self, batch_size = 1):
        return self.memory_buffer.pop()

    def __process_state__(self, state):
        tensor = torch.as_tensor(state, device = self.device, dtype = torch.float32)
        if (self.usesImageData):
            tensor = tensor.view(1, 4, 84, 84)
        else:
            tensor = tensor.view(1, 12)

        return tensor
    
    # assumes state is processed (!)
    def critic_value(self, state):
        if (self.BASELINE):
            return self.critic(state)
        else:
            raise Exception("No baseline so no critic: cannot get output of critic network")

    # runs optimizer on model
    def train(self):
        # updates consumes return
        if (self.getGameInfo().done):
            self.trainer.train(self)
            self.discounted_return = 0
    
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

    def recordLoss(self):
        if (self.BASELINE):
            meanLoss_actor, _, varianceLoss_actor = finalizeAggregate(self.trainer.policy_lossAggregate)
            meanLoss_critic, _, varianeLoss_critic = finalizeAggregate(self.trainer.stateval_lossAggregate)
            self.actor_avg_loss.append(meanLoss_actor)
            self.actor_var_loss.append(varianceLoss_actor)
            self.critic_avg_loss.append(meanLoss_critic)
            self.critic_var_loss.append(varianeLoss_critic)            
        else:
            meanLoss_actor, _, varianceLoss_actor = finalizeAggregate(self.trainer.policy_lossAggregate)
            self.actor_avg_loss.append(meanLoss_actor)
            self.actor_var_loss.append(varianceLoss_actor)

    def getLoss(self):

        if (self.BASELINE):
            return [self.actor_avg_loss, self.critic_avg_loss], [self.actor_var_loss, self.critic_var_loss], ["Actor (Policy Function) Loss", "Critic (Value Function) MSE Loss"]
        else:
            return [self.actor_avg_loss], [self.actor_var_loss], ["Actor (Policy Function) Loss"]

    def load_parameters(self):

        file_name = "./params/model_REINFORCE_"
        if (not self.usesImageData):
            file_name += "MLP"

        if (self.BASELINE):
            self.actor.load(file_name = file_name + "_actor__BASELINE.pth")
            self.critic.load(file_name = file_name + "_critic_BASELINE.pth")
        else:
            self.actor.load(file_name = file_name + "_actor.pth")

    def save_parameters(self):

        file_name = "./params/model_REINFORCE_"
        if (not self.usesImageData):
            file_name += "MLP"

        if (self.BASELINE):
            self.actor.save(file_name = file_name + "_actor__BASELINE.pth")
            self.critic.save(file_name = file_name + "_critic_BASELINE.pth")
        else:
            self.actor.save(file_name = file_name + "_actor.pth")
    
    def hasBaseline(self):
        return self.BASELINE
    
    # note: returns pytorch tensor
    def getReturn(self):
        return torch.as_tensor(self.discounted_return, device = self.device, dtype = torch.float32).view([1, 1])
