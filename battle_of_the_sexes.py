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
    A player of the Opera and Football game

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
    def get_reward(state, sex='female'):
        """Returns the reward for a very specific game.
        Following this matrix
        F..M       Opera |  Football
        Opera      (3,2)   (0,0)
        Football   (0,0)   (2,3)
        """
        reward = [0,0]
        # F - Opera, M - Opera
        if state == [0,0]:
            reward = [3, 2]
        # F - Opera, M - Football
        elif state == [0,1]:
            reward = [1,1]
        # F - Football, M - Opera
        elif state == [1,0]:
            reward = [0,0]
        # F - Football, M - Football
        elif state == [1,1]:
            reward = [2, 3]
        if sex == "female":
            # print("I am a girl: {}".format(reward[0]))
            return reward[0]
        else:
            # print("I am a guy: {}".format(reward[1]))
            return reward[1]

    @staticmethod
    def get_state_name(state):
        """We use 1 to represent Football,
        and we use 0 to represent Opera"""
        return [
            "Football" if n == 1 else "Opera" for n in state
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
    female_player = Player()
    male_player = Player()

    # Train them to be dominant
    for i in range(iterations):
        # First get the actions for both players.
        action_a = female_player.get_action(Player.get_initial_state())
        action_b = male_player.get_action(Player.get_initial_state())
        # Now get the reward for each player.
        female_player_reward = Player.get_reward([ action_a, 0 ], sex='female')
        male_player_reward = Player.get_reward([ 1, action_b ], sex='male')
        # Update the policy of each player given the reward,
        # notice how the update function only depends on the
        # reward and nothing else, in this particular
        # implementation.
        female_player.update(female_player_reward)
        male_player.update(male_player_reward)

    # Play them against each other.
    for i in range(iterations):
        # First get the actions for both players.
        action_a = female_player.get_action(Player.get_initial_state())
        action_b = male_player.get_action(Player.get_initial_state())
        # Now get the reward for each player.
        female_player_reward = Player.get_reward([ action_a, action_b ], sex='female')
        male_player_reward = Player.get_reward([ action_a, action_b ], sex='male')
        # Update the policy of each player given the reward,
        # notice how the update function only depends on the
        # reward and nothing else, in this particular
        # implementation.
        female_player.update(female_player_reward)
        male_player.update(male_player_reward)
    return (female_player, male_player)

def test_players(female_player, male_player, iterations=10):
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
        action_a = female_player.get_action(initial_state)
        action_b = male_player.get_action(initial_state)
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
    female_player, male_player = train_players(iterations=15000)
    # Now test the players.
    outcomes = test_players(female_player, male_player, iterations=100)
    # Print the outcomes in a nice way.
    outcomes_noramlized = normalize_outcomes(outcomes)
    print_outcomes(outcomes_noramlized)