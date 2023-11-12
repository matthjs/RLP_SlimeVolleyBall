from abc import ABC, abstractmethod

"""
generic RL agent
"""
class Agent(ABC):

    # inner class for recording game information
    class GameInfo:

        def __init__(self):
            self.done = False
            self.info = None
        
        def setDone(self, done):
            self.done = done

        def setInfo(self, info):
            self.info = info

    def __init__(self):
        self.gameInfo = self.GameInfo()
        self.usesImageData = True

    @abstractmethod
    def addTrajectory(self, trajectory):
        pass

    @abstractmethod
    def addTrajectory(self, trajectory):
        pass

    @abstractmethod
    def sampleTrajectory(self, batch_size = 1):
        pass

    @abstractmethod
    def __process_state__(self, state):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def policy(self, state):
        pass

    @abstractmethod
    def load_parameters(self):
        pass

    @abstractmethod
    def save_parameters(self):
        pass

    """
    Necessary for monte carlo agents
    to use the same interface as TD agents.
    Recording "done" can be used to prevent
    the agent from training until an episode
    is finished.
    """
    def recordGameInfo(self, info, done = False) -> None:
        self.gameInfo.setDone(done)
        self.gameInfo.setInfo(info)

    def getGameInfo(self) -> GameInfo:
        return self.gameInfo

