import torch
import torch.optim as optim
import torch.nn.functional as F
from dqn.trainer import Trainer
from loops.welford import updateAggregate

# performs the optimization process / learning parameters
# given model and input
class TrainerAC(Trainer):
    
    def __init__(self, actor, critic, learning_rate_actor, learning_rate_critic, gamma):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.learning_rate = learning_rate
        self.gamma = gamma
        self.actor = actor
        self.critic = critic
        self.I = 0

        self.policy_optimizer = optim.Adam(actor.parameters(), lr = learning_rate_actor)
        self.stateval_optimizer = optim.Adam(critic.parameters(), lr = learning_rate_critic)

        self.policy_runningLoss = 0.0
        self.stateval_runningLoss = 0.0
        self.policy_lossAggregate = (0.0, 0.0, 0.0)
        self.stateval_lossAggregate = (0.0, 0.0, 0.0)

    """
    S: (N, 4, 84, 84) tensor where
        N is the batch size
        4 is the channel depth (4 frames of the game are provided)
        84 x 84 is the width and height of the actual grayscale image
        N of (S, A, R, S)
        based on:
            https://github.com/chengxi600/RLStuff/blob/master/Actor-Critic/Actor-Critic_TD_0.ipynb
        # input is batch of (S, A, R, S)
        # with one-step actor critic BATCH_SIZE is always 1
    """
    def train(self, agent):

        # Actor Critic Agent only stores the last trajectory
        state, action, reward, next_state = agent.sampleTrajectory()

        # get the action and log probability
        logProb = agent.policy_log_prob(action, state)

        # get output from critic (value function for state)
        # encapsulated in value method of agent
        # next_state is None iff terminal (?)
        state_val = agent.critic_value(state)
        next_state_val = None

        if (next_state != None): 
            next_state_val = agent.critic_value(next_state)
        else:
            next_state_val = torch.tensor([0]).float().unsqueeze(0).to(self.device)
        
        # calculate MSE loss for value network
        value_loss = F.mse_loss(reward + self.gamma * next_state_val, state_val)
        value_loss *= self.I

        # calculate loss of policy
        advantage = reward + self.gamma * next_state_val.item() - state_val.item()
        policy_loss = -logProb * advantage
        policy_loss *= self.I    # added!

        # gradient descent step policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph = True)    # calculate gradient
        self.policy_optimizer.step()

        # gradient descent step value function
        self.stateval_optimizer.zero_grad()
        value_loss.backward()
        self.stateval_optimizer.step()

        # update discount factor
        self.I *= self.gamma

        # save information about losses
        self.policy_runningLoss += policy_loss.item()
        self.stateval_runningLoss += value_loss.item()
        self.policy_lossAggregate = updateAggregate(self.policy_lossAggregate, policy_loss.item())
        self.stateval_lossAggregate = updateAggregate(self.stateval_lossAggregate, value_loss.item())


