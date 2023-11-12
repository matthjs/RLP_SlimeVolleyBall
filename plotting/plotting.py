import numpy as np
import matplotlib.pyplot as plt

def unzip(listOfTuples):
    return list(zip(*listOfTuples))

def colorSelector(agentType):

    if (agentType == "RANDOM"):
        return "crimson"
    elif (agentType == "DQN"):
        return "limegreen"
    elif (agentType == "QN"):
        return "green"
    elif (agentType == "AC"):
        return "darkorange"
    elif (agentType == "REINFORCE-BL"):
        return "royalblue"
    elif (agentType == "REINFORCE"):
        return "aqua"
    elif (agentType == "AC-MLP"):
        return "orange"
    elif (agentType == "REINFORCE-MLP"):
        return "deepskyblue"
    elif (agentType == "REINFORCE-BL-MLP"):
        return "blue"

# spaghetti code that displays the results nicely
# average reward should be interpreted as average score
def superplot(results, rewardBaseline, folder = "./results/"):
    
    totalRewardRandom, avgRewardsRandom, varianceRewardsRandom, scoresRandom, labelRandom = rewardBaseline
    
    unzipped = unzip(results)
    listTotalReward = unzipped[0]
    listAvgRewards = unzipped[1]
    listVarRewards = unzipped[2]
    listScores = unzipped[3]
    listAvgLosses = unzipped[4]
    listLossLabels = unzipped[5]
    agentTypes = unzipped[6]

    MIN_REWARD = np.min(listAvgRewards) - np.min(listAvgRewards) * 0.1 #(np.min(listVarRewards) * 0.1)
    MIN_REWARD = min(MIN_REWARD, np.min(avgRewardsRandom) - np.min(avgRewardsRandom) * 0.1) #(np.min(varianceRewardsRandom) * 0.1))
    MAX_REWARD = np.max(listAvgRewards) + np.max(listAvgRewards) * 0.1 #(np.max(listVarRewards) * 0.1)
    MAX_REWARD = max(MAX_REWARD, np.max(avgRewardsRandom) + np.max(avgRewardsRandom) * 0.1) #(np.max(varianceRewardsRandom) * 0.1))
    print("MAXIMUM RECEIVED AVG REWARD:", np.max(listAvgRewards))
    print("MINIMUM RECEIVED REWARD:", np.min(listAvgRewards))

    figTR, axTR = plt.subplots(figsize=(8, 6))
    figAvgR, axAvgR = plt.subplots(figsize=(8, 6))

    axTR.set_title("Total Reward Obtained")
    axTR.grid()
    axAvgR.set_title("Average Score over Episodes")
    axAvgR.grid()
    axAvgR.set_ylim(MIN_REWARD, MAX_REWARD)
    axAvgR.set_ylabel("Average Score")
    axAvgR.set_xlabel("Number of Episodes")

    numEpisodes = len(avgRewardsRandom)
    x = np.linspace(0, numEpisodes, numEpisodes, endpoint = False)
    # plot rewards random policy
    axAvgR.plot(x, avgRewardsRandom, label = labelRandom, color = colorSelector(labelRandom))
    axAvgR.fill_between(
        x,
        avgRewardsRandom - (np.sqrt(varianceRewardsRandom) * 0.1),
        avgRewardsRandom + (np.sqrt(varianceRewardsRandom) * 0.1),
        alpha = 0.2,
        color = colorSelector(labelRandom)
    )

    for result in results:
        totalReward, avgRewards, varianceRewards, scores, avgLosses, varianceLosses, lossLabels, agentType = result

        axAvgR.plot(x, avgRewards, label = agentType, color = colorSelector(agentType))
        axAvgR.fill_between(
            x,
            avgRewards - (np.sqrt(varianceRewards) * 0.1),
            avgRewards + (np.sqrt(varianceRewards) * 0.1),
            alpha = 0.2,
            color = colorSelector(agentType)
        )

        figAvgR_singular, axAvgR_singular = plt.subplots(figsize=(8, 6))
        axAvgR_singular.set_title("Average Score Over Episodes" + " (RANDOM vs " + agentType + ")")
        axAvgR_singular.grid()
        axAvgR_singular.set_ylabel("Average Score")
        axAvgR_singular.set_xlabel("Number of Episodes")
        axAvgR_singular.plot(x, avgRewards, label = agentType, color = colorSelector(agentType))
        axAvgR_singular.fill_between(
            x,
            avgRewards - (np.sqrt(varianceRewards) * 0.1),
            avgRewards + (np.sqrt(varianceRewards) * 0.1),
            alpha = 0.2,
            color = colorSelector(agentType)
        )
        axAvgR_singular.plot(x, avgRewardsRandom, label = labelRandom, color = colorSelector("RANDOM"))
        axAvgR_singular.fill_between(
            x,
            avgRewardsRandom - (np.sqrt(varianceRewardsRandom) * 0.1),
            avgRewardsRandom + (np.sqrt(varianceRewardsRandom) * 0.1),
            alpha = 0.2,
            color = colorSelector(labelRandom)
        )
        axAvgR_singular.legend()
        axAvgR_singular.set_ylim(MIN_REWARD, MAX_REWARD)
        figAvgR_singular.savefig(folder + "img_avgReward_" + agentType + ".png", dpi = 300)

        figLoss, axLoss = plt.subplots(figsize=(8, 6))
        axLoss.set_title("Average Loss over Episodes " + "(" + agentType + ")")
        axLoss.grid()
        axLoss.set_ylabel("Average Loss")
        axLoss.set_xlabel("Number of Episodes")
        
        for avgLoss, varianceLoss, lossLabel in zip(avgLosses, varianceLosses, lossLabels):
            axLoss.plot(x, avgLoss, label = lossLabel)
            axLoss.fill_between(
                x,
                avgLoss - (np.sqrt(varianceLoss) * 0.1),
                avgLoss + (np.sqrt(varianceLoss) * 0.1),
                alpha = 0.2,
            )
    
        axLoss.legend()
        figLoss.savefig(folder + "img_avgLoss_" + agentType + ".png", dpi = 300)

    axAvgR.legend()
    figAvgR.savefig(folder + "img_avgReward_.png", dpi = 300)

# spaghetti code that displays the results nicely
def superplot2(results, rewardBaseline, folder = "./results/"):

    def isMLP(type):
        if (type == "QN"):
            return True
        
        return len(type) >= 3 and type[-3:] == "MLP"
    
    totalRewardRandom, avgRewardsRandom, varianceRewardsRandom, scoresRandom, labelRandom = rewardBaseline
    
    unzipped = unzip(results)
    listTotalReward = unzipped[0]
    listAvgRewards = unzipped[1]
    listVarRewards = unzipped[2]
    listScores = unzipped[3]
    listAvgLosses = unzipped[4]
    listLossLabels = unzipped[5]
    agentTypes = unzipped[6]

    MIN_REWARD = np.min(listAvgRewards) - np.min(listAvgRewards) * 0.1
    MIN_REWARD = min(MIN_REWARD, np.min(avgRewardsRandom) - np.min(avgRewardsRandom) * 0.1)
    MAX_REWARD = np.max(listAvgRewards) + np.max(listAvgRewards) * 0.1
    MAX_REWARD = max(MAX_REWARD, np.max(avgRewardsRandom) + np.max(avgRewardsRandom) * 0.1)
    print("MAXIMUM RECEIVED AVG REWARD:", np.max(listAvgRewards))
    print("MINIMUM RECEIVED REWARD:", np.min(listAvgRewards))

    figTR, axTR = plt.subplots(figsize=(8, 6))
    figAvgR, (axAvgR, axAvgR_MLP) = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(9, 6))

    axTR.set_title("Total Reward Obtained")
    axTR.grid()
    axAvgR.set_title("Average Score over Episodes")
    axAvgR.grid()
    axAvgR_MLP.grid()
    axAvgR.set_ylim(MIN_REWARD, MAX_REWARD)
    axAvgR.set_ylabel("Average Score")
    axAvgR_MLP.set_ylabel("Average Score")
    axAvgR_MLP.set_xlabel("Number of Episodes")

    numEpisodes = len(avgRewardsRandom)
    x = np.linspace(0, numEpisodes, numEpisodes, endpoint = False)
    # plot rewards random policy
    axAvgR.plot(x, avgRewardsRandom, color = colorSelector(labelRandom))
    axAvgR.fill_between(
        x,
        avgRewardsRandom - (np.sqrt(varianceRewardsRandom) * 0.1),
        avgRewardsRandom + (np.sqrt(varianceRewardsRandom) * 0.1),
        alpha = 0.2,
        color = colorSelector(labelRandom)
    )
    axAvgR_MLP.plot(x, avgRewardsRandom, label = labelRandom, color = colorSelector(labelRandom))
    axAvgR_MLP.fill_between(
        x,
        avgRewardsRandom - (np.sqrt(varianceRewardsRandom) * 0.1),
        avgRewardsRandom + (np.sqrt(varianceRewardsRandom) * 0.1),
        alpha = 0.2,
        color = colorSelector(labelRandom)
    )

    for result in results:
        totalReward, avgRewards, varianceRewards, scores, avgLosses, varianceLosses, lossLabels, agentType = result

        if (isMLP(agentType)):
            axAvgR_MLP.plot(x, avgRewards, label = agentType, color = colorSelector(agentType))
            axAvgR_MLP.fill_between(
                x,
                avgRewards - (np.sqrt(varianceRewards) * 0.1),
                avgRewards + (np.sqrt(varianceRewards) * 0.1),
                alpha = 0.2,
                color = colorSelector(agentType)
            )
        else:
            axAvgR.plot(x, avgRewards, label = agentType, color = colorSelector(agentType))
            axAvgR.fill_between(
                x,
                avgRewards - (np.sqrt(varianceRewards) * 0.1),
                avgRewards + (np.sqrt(varianceRewards) * 0.1),
                alpha = 0.2,
                color = colorSelector(agentType)
            )

        figAvgR_singular, axAvgR_singular = plt.subplots(figsize=(8, 6))
        axAvgR_singular.set_title("Average Score Over Episodes" + " (RANDOM vs " + agentType + ")")
        axAvgR_singular.grid()
        axAvgR_singular.set_ylabel("Average Score")
        axAvgR_singular.set_xlabel("Number of Episodes")
        axAvgR_singular.plot(x, avgRewards, label = agentType, color = colorSelector(agentType))
        axAvgR_singular.fill_between(
            x,
            avgRewards - (np.sqrt(varianceRewards) * 0.1),
            avgRewards + (np.sqrt(varianceRewards) * 0.1),
            alpha = 0.2,
            color = colorSelector(agentType)
        )
        axAvgR_singular.plot(x, avgRewardsRandom, label = labelRandom, color = colorSelector("RANDOM"))
        axAvgR_singular.fill_between(
            x,
            avgRewardsRandom - (np.sqrt(varianceRewardsRandom) * 0.1),
            avgRewardsRandom + (np.sqrt(varianceRewardsRandom) * 0.1),
            alpha = 0.2,
            color = colorSelector(labelRandom)
        )
        axAvgR_singular.legend()
        axAvgR_singular.set_ylim(MIN_REWARD, MAX_REWARD)
        figAvgR_singular.savefig(folder + "img_avgReward_" + agentType + ".png", dpi = 300)

        figLoss, axLoss = plt.subplots(figsize=(8, 6))
        axLoss.set_title("Average Loss over Episodes " + "(" + agentType + ")")
        axLoss.grid()
        axLoss.set_ylabel("Average Loss")
        axLoss.set_xlabel("Number of Episodes")
        
        for avgLoss, varianceLoss, lossLabel in zip(avgLosses, varianceLosses, lossLabels):
            axLoss.plot(x, avgLoss, label = lossLabel)
            axLoss.fill_between(
                x,
                avgLoss - (np.sqrt(varianceLoss) * 0.1),
                avgLoss + (np.sqrt(varianceLoss) * 0.1),
                alpha = 0.2,
            )
    
        axLoss.legend()
        figLoss.savefig(folder + "img_avgLoss_" + agentType + ".png", dpi = 300)

    axAvgR.legend(fontsize = 7.5)
    axAvgR_MLP.legend(fontsize = 7.5)
    figAvgR.savefig(folder + "img_avgReward_.png", dpi = 300)