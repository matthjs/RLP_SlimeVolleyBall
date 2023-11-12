from abc import ABC, abstractmethod

# performs the optimization process / learning parameters
# given model and input
class Trainer(ABC):

    @abstractmethod
    def train(self, agent):
        pass