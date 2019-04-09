from tensorforce.agents import TRPOAgent
import tensorflow as tf
import numpy as np
from arena import ClearingHouse
import pickle
from tqdm import tqdm

numberOfPlayers = 10

#training phase

Market = ClearingHouse(numberOfPlayers, 4, .7, .3)

initialState = Market.get_state()

agent = TRPOAgent(
    states=dict(type='float', shape=(len(initialState), len(initialState[0]),)),
    actions=dict(type='int', shape=(Market.numOfResources,), num_actions=3),
    network=[
        dict(type="flatten"),
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batching_capacity=10,
)


playerList = [agent for i in range(numberOfPlayers)]

training_size = 1000000
for i in tqdm(range(training_size)):
    # print("iteration #", i, "/", training_size)

    state = Market.get_state()


    actionVector = []
    for player in playerList:
        action = player.act(state)
        actionVector.append(action)


    rewards = Market.computeRewards(actionVector)

    x = 0
    for player in playerList:
        player.observe(reward=rewards[x], terminal=False)
        x += 1

#playing phase 
Market = ClearingHouse(numberOfPlayers, 4, .7, .3)

initialState = Market.get_state()

play_size = 100000
for i in tqdm(range(play_size)):
    # print("iteration #", i, "/", play_size)

    state = Market.get_state()


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

