from agent.REINFORCE_agent import REINFORCEAgent
from agent.actor_critic_agent import ActorCriticAgent
from agent.dqn_agent import DQNAgent


class AgentFactory:

    def createAgent(self, type):
        if (type == "DQN"):
            return DQNAgent(epsilon = 1, param_copying = 20, schedule = (0.1, 400), batch_size = 64, learning_rate = 0.001, gamma = 0.9)
        elif (type == "QN"):
            return DQNAgent(epsilon = 1, param_copying = 20, schedule = (0.1, 400), batch_size = 64, learning_rate = 0.001, gamma = 0.9, usesImageData = False)
        elif (type == "AC"):
            return ActorCriticAgent(learning_rate_actor = 0.001, learning_rate_critic = 0.005, gamma = 0.9)
        elif (type == "AC-MLP"):
            return ActorCriticAgent(learning_rate_actor = 0.001, learning_rate_critic = 0.005, gamma = 0.9, usesImageData = False)
        elif (type == "REINFORCE"):
            return REINFORCEAgent(BASELINE = False, discount_reward = 0.9, learning_rate_actor = 0.001, learning_rate_critic = 0.005, gamma = 0.9)
        elif (type == "REINFORCE-MLP"):
            return REINFORCEAgent(BASELINE = False, discount_reward = 0.9, learning_rate_actor = 0.001, learning_rate_critic = 0.005, gamma = 0.9, usesImageData = False)
        elif (type == "REINFORCE-BL"):
            return REINFORCEAgent(BASELINE = True, discount_reward = 0.9, learning_rate_actor = 0.001, learning_rate_critic = 0.005, gamma = 0.9)
        elif (type == "REINFORCE-BL-MLP"):
            return REINFORCEAgent(BASELINE = True, discount_reward = 0.9, learning_rate_actor = 0.001, learning_rate_critic = 0.005, gamma = 0.9, usesImageData = False)
        else:
            raise Exception("Non-existent agent type")