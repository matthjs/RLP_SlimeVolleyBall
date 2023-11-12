This repository contains Pytorch implementations of various reinforcement learning
algorithms.

These algorithms are applied to the SlimeVolleyGym environment.

Running the code (also requirements.txt):

Use
    python main.py MULTITRAIN <number of episodes> <agentType(s)>

To run a multithreaded program that trains all
agents (DQN, AC, REINFORCE, REINFORCE-BL) (+ gets results for random policy)
for N number of episodes at once.

Use
    python main.py <setting> <number of episodes> <agentType>
For individual agents

Use
    python main.py PLOT folder <csv files>
    to plot the data from csv files. The csv will need to be in a
    very specific format for everything not to break down. However,
    the csv files that are saved after training an agent work.

<settings>:
    TRAIN: train an agent from scratch. So NO loading parameters and NO
            visualization of the game.
    PLAY: tries to load the parameters of the agent you specified. Visualizes
            the game.
    EVAL: parameter loading, no training, no game visualization.

By default, the agent plays against a baseline policy (see BaselinePolicy
    class in slimevolley.py)

<agentTypes>:
The agent factory class (agentFactory.py) uses the following str representations
to create the appropriate RL agent:
DQN = Deep Q-Network
QN = Q-network (MLP)
AC = One-Step Actor-Critic (using CNN)
AC-MLP = One-Step Actor-Critic (using MLP)
REINFORCE = REINFORCE (Monte-Carlo Policy Gradient algorithm) (using CNN)
REINFORCE-MLP = REINFORCE (using MLP)
REINFORCE-BL = REINFORCE with Baseline (using CNN)
REINFORCE-BL-MLP = REINFORCE with Baseline (using MLP)

