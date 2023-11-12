from collections import namedtuple
import torch
import torch.nn as nn
from dqn.trainer import Trainer
from loops.welford import updateAggregate

# performs the optimization process / learning parameters
# given model and input
class TrainerDQN(Trainer):
    
    def __init__(self, model, batch_size, learning_rate, gamma):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model    # make sure model is on GPU!
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        self.runningLoss = 0.0
        self.lossAggregate = (0.0, 0.0, 0.0)

    """
    S: (N, 4, 84, 84) tensor where
        N is the batch size
        4 is the channel depth (4 frames of the game are provided)
        84 x 84 is the width and height of the actual grayscale image
        N of (S, A, R, S)
        based on:
            https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # input is batch of (S, A, R, S)
    """
    def train(self, agent):
    
        Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

        if len(agent.memory_buffer) < self.batch_size:
            return
        
        transitions = agent.sampleTrajectory(self.batch_size)

        # converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                            device = self.device, dtype = torch.bool)

        # shape: (BATCH_SIZE, 4, 84, 84)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # shape: (BATCH_SIZE, 4, 84, 84)
        state_batch = torch.cat(batch.state)
        # shape: (BATCH_SIZE)
        action_batch = torch.as_tensor(batch.action, device = self.device, dtype = torch.int64)
        # shape: (BATCH_SIZE)
        reward_batch = torch.as_tensor(batch.reward, device = self.device, dtype = torch.float32)

        # compute q values of policy network and get
        # q values for the specifically chosen actions
        # model output is BATCH_SIZE x 5 tensor
        model_output = agent.model(state_batch)
        qValues = []
        for idx, vec in enumerate(model_output):
            qValues.append(vec[action_batch[idx]])
        
        state_action_values = torch.as_tensor(qValues, device = self.device, dtype = torch.float32)

        next_state_values = torch.zeros(self.batch_size, device = self.device)
        
        # compute max q values for next state in batch from target network
        with torch.no_grad():
            next_state_values[non_final_mask] = agent.model_TDtarget(non_final_next_states).max(1)[0]

        # compute the full TD target
        td_target = (next_state_values * self.gamma) + reward_batch

        # compute (mean square error) loss
        loss = self.loss(state_action_values, td_target)

        # optimize the model
        self.optimizer.zero_grad()    # clear gradients
        loss.requires_grad = True
        loss.backward()    # compute gradient

        torch.nn.utils.clip_grad_value_(agent.model.parameters(), 100)
        self.optimizer.step()

        # keep track of running average and variance of loss
        self.runningLoss += loss.item()
        self.lossAggregate = updateAggregate(self.lossAggregate, loss.item())