from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np


class MyClient(object):
    def __init__(self, *args, **kwargs):
        pass
    def get_state(self):
        state = [0, int(np.random.rand() > 0.5)]
        # print('Initial state: {}'.format(state))
        return state
    def execute(self, action, state):
        state[0] = action
        # print('State after action: {}'.format(state))
        if state == [0,0]:
            return 2
        if state == [0,1]:
            return 0
        if state == [1,0]:
            return 3
        if state == [1,1]:
            return 1

def get_print_func(verbose):
    if verbose:
        return print
    def anom_print(msg):
        pass
    return anom_print


def test_reinforceio_homepage(iterations=1000, verbose=False):
    """
    Code example from the homepage and README.md.
    """
    from tensorforce.agents import TRPOAgent
    print_v = get_print_func(verbose)
    # Create a Trust Region Policy Optimization agent
    agent = TRPOAgent(
        states_spec={'shape': (2,), 'type': 'float'},
        actions_spec={'type': 'int', 'num_actions': 2},
        network_spec=[{'type': 'dense', 'size': 50}, {'type': 'dense', 'size': 50}],
        batch_size=100,
    )
    # Get new data from somewhere, e.g. a client to a web app
    client = MyClient('http://127.0.0.1', 8080)
    for i in range(iterations):
        print_v('-'*10)
        # Poll new state from client
        state = client.get_state()
        # Get prediction from agent, execute
        action = agent.act(states=state)
        print_v('Executing action: {}'.format(action))
        reward = client.execute(action, state)
        print_v('Reward: {}'.format(reward))
        # Add experience, agent automatically updates model according to batch size
        agent.observe(reward=reward, terminal=True)
    # agent.close()
    return agent

def test_trained_agent(agent, iterations=1000):
    # If the other person chooses war
    # test the percent of times this agent chooses
    # war too
    warTotal = 0
    for i in range(iterations):
        action = agent.act(states=(0,1))
        if action == 1:
            warTotal += 1
    print("If the other person chooses war...")
    print('Percent I choose war: {}%'.format((warTotal / iterations)*100.0))    
    warTotal = 0
    for i in range(iterations):
        action = agent.act(states=(0,0))
        if action == 1:
            warTotal += 1
    print("If the other person chooses peace...")
    print('Percent I choose war: {}%'.format((warTotal / iterations)*100.0))

if __name__=="__main__":
    agent = test_reinforceio_homepage(iterations=int(1e4))
    test_trained_agent(agent)
    agent.close()
