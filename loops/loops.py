import gym
from test_atari import *
from agent.agentFactory import AgentFactory
from loops.slimeEnv import getSlimeEnv
from loops.welford import finalizeAggregate, updateAggregate

"""
Records results from a random policy
"""
def random_agent_loop(episodes: int = 1000):
    viewer = rendering.SimpleImageViewer(maxwidth=2160)

    env = gym.make("SlimeVolleySurvivalNoFrameskip-v0")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, 4)

    totalReward = 0
    avgScores = []
    scores = []    # score <=> undiscounted return
    varianceScores = []
    rewardAggregate = (0, 0, 0)

    for episode in range(0, episodes):
        env.seed(689)
        env.reset()
        done = False

        episode_reward = 0
        while not done:
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)

            episode_reward += reward

        totalReward += episode_reward
        scores.append(episode_reward)
        # run online algorithm for tracking mean, variance scores = return
        rewardAggregate = updateAggregate(rewardAggregate, episode_reward)
        meanReward, _, varianceReward = finalizeAggregate(rewardAggregate)
        avgScores.append(meanReward)
        varianceScores.append(varianceReward)

    return (totalReward, avgScores, varianceScores, scores, "RANDOM")

def checkSetting(setting: str) -> bool:

    if (setting == "TRAIN" or setting == "PLAY" or setting == "EVAL"):
        return True
    
    raise Exception("Non-existent loop setting: " + setting)

"""
Does agent loop. The opponent (left) is a pretrained Baseline Policy
using an RNN (see BaselinePolicy class in SlimeVolley.py).
Input:
    agentType: used as input for AgentFactory class. Returning
            classes inheriting from the Abstract Base Class Agent
            Note that the string names differ from the class names.
            This is done because (for example) REINFORCE and REINFORCE with Baseline
            are implemented in the same class.
    setting:
        TRAIN: train an agent from scratch. So NO loading parameters and NO
                visualization of the game.
        PLAY: tries to load the parameters of the agent you specified. Visualizes
                the game.
        EVAL: parameter loading, no training, no game visualization.
    episodes:
        How many episodes (terminal state = time out or someone loses) should
        the environment play itself out?
    saveParams:
        For if you want to train but do not want to save the parameters.

Returns:
    A tuple of (what the agent has accomplished):
        1. totalReward: The total reward the agent has received
        2. avgScores: The average undiscounted return per episode (list)
        3. Scores: The exact score (undiscounted return / sum of rewards) received within each specific episode
        3. avgLosses: Average loss (the loss function that the optimizer of the agent's model uses)
            per episode (FOR EACH USED LOSS FUNCTION) (list of list)
        4. loss_labels: If multiple loss functions were used (ex. actor critic methods) then
                        the names for each. If one loss function than it is a list with a single element.
        5. agentType: The agent type (str representation that the agent factory uses)
"""
def loop(agentType: str, setting: str, episodes: int, saveParams: bool = False):
    print("[" + agentType + "]: " + "in train loop!")

    viewer = rendering.SimpleImageViewer(maxwidth=2160)

    agent = AgentFactory().createAgent(agentType)    # factory class for agents
    env = getSlimeEnv(agent)                         # factory function for environment
    checkSetting(setting)

    if (setting != "TRAIN"):
        print("loading parameters!")
        agent.load_parameters()

    # recording relevant information
    totalReward = 0
    avgScores = []    # average score
    scores = []    # score <=> undiscounted return
    varianceScores = []
    rewardAggregate = (0, 0, 0)

    for episode in range(0, episodes):

        if (episode == episodes / 2):
            print("[" + agentType + "]: " + "Half way done!")

        env.seed(689)
        obs = env.reset()
        done = False

        episode_reward = 0

        while not done:

            old_obs = obs
            action = agent.policy(obs)
            obs, reward, done, info = env.step(action)

            agent.recordGameInfo(info, done)    # monte carlo agents need to be aware of done
            agent.addTrajectory((old_obs, action, reward, obs))
            
            if (setting == "TRAIN"):
                agent.train()
            elif (setting == "PLAY"):
                env.render()
                sleep(0.04)

            episode_reward += reward

        # update performance metrics
        totalReward += episode_reward
        scores.append(episode_reward)
        # run online algorithm for tracking mean, variance rewards
        rewardAggregate = updateAggregate(rewardAggregate, episode_reward)
        meanReward, _, varianceReward = finalizeAggregate(rewardAggregate)
        avgScores.append(meanReward)
        varianceScores.append(varianceReward)

        agent.recordLoss()    # the agent is responsible for recording its loss

    if (setting == "TRAIN" or saveParams):
        agent.save_parameters()

    avgLosses, varianceLosses, loss_labels = agent.getLoss()

    return (totalReward, avgScores, varianceScores, scores, avgLosses, varianceLosses, loss_labels, agentType)