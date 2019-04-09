from tensorforce.agents import TRPOAgent
# import tensorflow as tf
import numpy as np
from arena import ClearingHouse
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--population", help="int of how many agents you want to test")
parser.add_argument("--agent", nargs='+', help="select agent type [ppo, dqn, vpg] ")
parser.add_argument("--ceiling", help="upper threshold for the bar")
parser.add_argument("--floor", help="lower threshold for the bar")

args = parser.parse_args()

numberOfPlayers = 10

#training phase

Market = ClearingHouse(int(args.population), 4, float(args.ceiling), float(args.floor))

initialState = Market.get_state()

agent = TRPOAgent(
    states={"type":'float', "shape": (int(args.population), 1, 4,)},
    actions={"type":'int', "shape": (Market.numOfResources,), "num_values":3},
    network="auto",
    memory=10000,
)

agent.initialize()


playerList = [agent for i in range(numberOfPlayers)]

training_size = 1000000
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
Market = ClearingHouse(numberOfPlayers, 4, .7, .3)

play_size = 100000
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

f = open("market.pkl", "wb") 
pickle.dump(Market.market, f)

