from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import TRPOAgent
import unittest
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Player:
    """
    A player of the Peace and War game

    This implementation of the gamedoes not have a concept of the
    moves the other player has done, so the policy is only updated
    based on the rewards for a given action.

    @authors
        Lorenzo Barberis Canonico
        Jesus Andres Castaneda
    """
    def __init__(self):
        # Create a Trust Region Policy Optimization agent
        self.agent = TRPOAgent(
            states_spec={'shape': (2,), 'type': 'float'},
            actions_spec={'type': 'int', 'num_actions': 2},
            network_spec=[{'type': 'dense', 'size': 50}, {'type': 'dense', 'size': 50}],
            batch_size=100,
        )

    @staticmethod
    def get_initial_state():
        """The initial state is arbitrary.
        Since this is a game where the players make a move at
        the same time, we feed both players an arbitrary state
        so they begin playing"""
        return [0, 0]

    @staticmethod
    def get_reward(state):
        """Returns the reward for a very specific game.
        Following this matrix
        A..B  Peace |  War
        Peace (2,2)   (0,1)
        War   (1,0)   (1,1)
        """
        if state == [0,0]:
            return 2
        if state == [0,1]:
            return 0
        if state == [1,0]:
            return 3
        if state == [1,1]:
            return 1

    @staticmethod
    def get_state_name(state):
        """We use 1 to represent War,
        and we use 0 to represent Peace"""
        return [
            "War" if n == 1 else "Peace" for n in state
        ]

    def get_action(self, state):
        """Take a prediction for a given state.
        The state in this particular game will
        be a consitent arbitrary state, since
        this is a single move game, where all moves
        are done simultaneous"""
        action = self.agent.act(states=state)
        return action

    def update(self, reward):
        """Update policy given a reward"""
        self.agent.observe(reward=reward, terminal=True)


def train_players(iterations=10):
    print("Training over {} iterations".format(iterations))
    # Initialize players.
    player_a = Player()
    player_b = Player()
    for i in range(iterations):
        # First get the actions for both players.
        action_a = player_a.get_action(Player.get_initial_state())
        action_b = player_b.get_action(Player.get_initial_state())
        # Now get the reward for each player,
        # notice how the order of the actions
        # are flipped. Because we get the reward
        # from the perspective of the given player.
        player_a_reward = Player.get_reward([ action_a, action_b ])
        player_b_reward = Player.get_reward([ action_b, action_a ])
        # Update the policy of each player given the reward,
        # notice how the update function only depends on the
        # reward and nothing else, in this particular
        # implementation.
        player_a.update(player_a_reward)
        player_b.update(player_b_reward)
    return (player_a, player_b)

def test_players(player_a, player_b, iterations=10):
    print("Testing over {} iterations".format(iterations))
    # Initialize a counter for each possible state.
    outcomes = {
        (0,0):0,
        (0,1):0,
        (1,0):0,
        (1,1):0
    }
    for i in range(iterations):
        # Get action, and add to the outcome counter.
        # we don't need to update the policy since
        # we are only testing.
        initial_state = Player.get_initial_state()
        action_a = player_a.get_action(initial_state)
        action_b = player_b.get_action(initial_state)
        outcomes[(action_a, action_b)] += 1
    return outcomes

def normalize_outcomes(outcomes):
    """Normalizes the outcomes over the total
    number of outcomes"""
    total = 0
    for key in outcomes:
        total += outcomes[key]
    for key in outcomes:
        outcomes[key] = outcomes[key] / float(total)
    return outcomes

def print_outcomes(outcomes):
    """Pretty prints the outcomes"""
    for key in outcomes:
        print("{}: {}%".format(Player.get_state_name(key), outcomes[key]*100))

if __name__=="__main__":
    # Create two players, and train them
    player_a, player_b = train_players(iterations=10000)
    # Now test the players.
    outcomes = test_players(player_a, player_b, iterations=100)
    # Print the outcomes in a nice way.
    outcomes_noramlized = normalize_outcomes(outcomes)
    print_outcomes(outcomes_noramlized)