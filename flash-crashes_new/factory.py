from tensorforce.agents import TRPOAgent, VPGAgent, DQNAgent
# import tensorflow as tf
import numpy as np
from arena import ClearingHouse
import pickle
from tqdm import tqdm
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--population", help="int of how many agents you want to test")
parser.add_argument("--resources", help="int of how many agents you want to test")
parser.add_argument("--agent", nargs='+', help="select agent type [ppo, dqn, vpg] ")
parser.add_argument("--ceiling", help="upper threshold for the bar")
parser.add_argument("--floor", help="lower threshold for the bar")

args = parser.parse_args()

#training phase

Market = ClearingHouse(int(args.population), int(args.resources), float(args.ceiling), float(args.floor))

def get_agent(agentType): 

    if agentType == "dqn":
        agent = DQNAgent(
            states={"type":'float', "shape": (int(args.population), 1, int(args.resources),)},
            actions={"type":'int', "shape": (int(args.resources),), "num_values":3},
            memory=1000,
            network="auto",
        )
    elif agentType == "vpg":
        agent = VPGAgent(
            states={"type":'float', "shape": (int(args.population), 1, int(args.resources),)},
            actions={"type":'int', "shape": (int(args.resources),), "num_values":3},
            network="auto",
            memory=1000,
        )
    elif agentType == "trpo":
        agent = TRPOAgent(
            states={"type":'float', "shape": (int(args.population), 1, int(args.resources),)},
            actions={"type":'int', "shape": (int(args.resources),), "num_values":3},
            network="auto",
            memory=1000,
        )

    return agent

playerList = []

for agentType in args.agent:
    agent_batch = int(12 / len(args.agent))
    for i in range(agent_batch):
        agent = get_agent(agentType)
        agent.initialize()
        print("agent", i, "ready")
        playerList.append(agent)

print(playerList)

 

training_size = 100000
# training_size = 10
for i in tqdm(range(training_size)):

    state = Market.get_state()
    if i < 1:
        state = state
    else:
        state = [state]


    actionVector = []
    for player in playerList:
        action = player.act(state)
        actionVector.append(action)

    rewards = Market.computeRewards(actionVector)

    x = 0
    for player in playerList:
        player.observe(reward=rewards[x], terminal=False)
        x += 1

# #playing phase 
Market = ClearingHouse(int(args.population), int(args.resources), float(args.ceiling), float(args.floor))


play_size = 100000
# play_size = 10
for i in tqdm(range(play_size)):

    state = Market.get_state()

    if i < 1:
        state = state
    else:
        state = [state]

    actionVector = []
    for player in playerList:
        action = player.act(state)
        actionVector.append(action)


    rewards = Market.computeRewards(actionVector)

    x = 0
    for player in playerList:
        player.observe(reward=rewards[x], terminal=False)
        x += 1

print(Market.score)

f = open("market_pop" + args.population + "_res" + args.resources + "_" + "-".join(args.agent) + ".pkl", "wb") 

pickle.dump(Market.market, f)

