from test_atari import *
import pandas as pd
import sys
import concurrent.futures
from loops.loops import loop, random_agent_loop
from plotting.plotting import superplot, superplot2

# record results
def recordResults(results, baseline_result, folder = "./results/"):

    totalRewardRandom, avgScoresRandom, varianceScoresRandom, scoresRandom, labelRandom = baseline_result
    dataFrame = pd.DataFrame()
    dataFrame["sample_mean_score"] = avgScoresRandom
    dataFrame["sample_variance_score"] = varianceScoresRandom
    dataFrame["scores"] = scoresRandom
    dataFrame.to_csv(folder + "RANDOM.csv")

    for result in results:
        totalReward, avgScores, varianceScores, scores, avgLosses, varianceLosses, lossLabels, agentType = result
        dataFrame = pd.DataFrame()

        dataFrame["sample_mean_score"] = avgScores
        dataFrame["sample_variance_score"] = varianceScores
        dataFrame["scores"] = scores
        for avgLoss, lossLabel in zip(avgLosses, lossLabels):
            dataFrame[lossLabel + "_mean"] = avgLoss
        
        for varianceLoss, lossLabel in zip(varianceLosses, lossLabels):
            dataFrame[lossLabel + "_variance"] = varianceLoss

        dataFrame.to_csv(folder + agentType + ".csv")

# works but is very brittle
def plotFromCSV(paths, folder = "./results/", baselinePath = "RANDOM.csv"):

    df = pd.read_csv(folder + baselinePath)
    baseline_result = (0, df["sample_mean_score"].to_numpy(), df["sample_variance_score"], df["scores"], baselinePath[0:-4])

    results = []
    for path in paths:
        dataFrame = pd.read_csv(folder + path)
        
        result = []
        loss_mean = []
        loss_var = []
        lossLabels = []

        # first: sample mean reward
        # second: sample variance reward
        for col in dataFrame.columns:

            if (col[-4:] == "mean"):
                loss_mean.append(dataFrame[col].to_numpy())
                lossLabels.append(col[:-5])
            elif (col[-8:] == "variance"):    # assume variance idk
                loss_var.append(dataFrame[col].to_numpy())
            else:
                result.append(dataFrame[col].to_numpy())
        
        result.append(loss_mean)
        
        result.append(loss_var)

        result.append(lossLabels)
        result.append(path[0:-4])    # get name
    
        results.append(tuple(result))
    
    superplot2(results, baseline_result)


# train agents in parallel
def multiAgentTrain(agentList, episodes):

    executor = concurrent.futures.ThreadPoolExecutor(max_workers = len(agentList) + 1)

    futures = []
    for agentType in agentList:
            futures.append(executor.submit(loop, agentType, "TRAIN", episodes, True))
    
    baseline = executor.submit(random_agent_loop, episodes)

    results = []
    for future in futures:
        results.append(future.result())

    baseline_result = baseline.result()

    superplot(results, baseline_result)
    recordResults(results, baseline_result)

def singleAgentSetup(agentType, setting, episodes):
    
    # get results from random policy from a seperate thread
    executor = concurrent.futures.ThreadPoolExecutor(max_workers = 2)
    baseline = executor.submit(random_agent_loop, episodes)

    result = loop(agentType, setting, episodes)
    baseline_result = baseline.result()

    if (setting == "TRAIN"):
        superplot([result], baseline_result)
        recordResults([result], baseline_result)

"""
make sure that linker is able to link
with libcuda.so (!)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/pkgs/cuda-driver-dev-11.7.99-0/lib/stubs
"""
if __name__ == "__main__":

    if (sys.argv[1] == "PLOT"):
        plotFromCSV(sys.argv[3:], sys.argv[2])
    elif (len(sys.argv) >= 3):
        if (sys.argv[1] == "MULTITRAIN"):
            multiAgentTrain(sys.argv[3:], int(sys.argv[2]))
        else:
            singleAgentSetup(sys.argv[3], sys.argv[1], int(sys.argv[2]))

