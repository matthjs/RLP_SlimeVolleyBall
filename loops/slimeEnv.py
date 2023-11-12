import gym
from slimevolleygym.slimevolley import FrameStack, SlimeVolleyEnv
from test_atari import MaxAndSkipEnv, NoopResetEnv, WarpFrame
from agent.agent import Agent

"""
same as SlimeVolleySurvivalNoFrameSkip-v0 but with
feature vectors for states
"""
class SlimeVolleyEnv_WF(SlimeVolleyEnv):
    from_pixels = False     # use feature vectors not images
    atari_mode = True       # represent actions by integers
    survival_bonus = True   # also used in SlimeVolleySurvivalNoFrameSkip-v0

"""
Factory function for slimevolley environment.
It decides based on an Agent object, which
version of slimevolley it should return
(i.e. one that provides images for states
or feature vectors for states).
"""
def getSlimeEnv(agent: Agent):
    
    if (not agent.usesImageData):
        return SlimeVolleyEnv_WF()

    env = gym.make("SlimeVolleySurvivalNoFrameskip-v0")
    # atari game preprocessing
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    return env