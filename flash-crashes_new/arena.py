import numpy as np
import matplotlib.pyplot as plt


class ClearingHouse():

	def __init__(self, numOfPlayers, numOfResources, highThreshold, lowThreshold):
		self.numOfPlayers = numOfPlayers
		self.numOfResources = numOfResources
		self.highThreshold = highThreshold
		self.lowThreshold = lowThreshold
		self.market = {}
		for i in range(numOfResources):
			self.market[i] = {
				"short": [],
				"long": []
			}
		#self.state = [[0 for i in range(self.numOfResources)] for x in range(self.numOfPlayers)]

		self.state = np.full((numOfPlayers, 1, numOfResources), 0)
		self.score = np.asarray([0 for i in range(self.numOfPlayers)])


	def computeRewards(self, tensor):

		#use the combined vector of each decision to update the state
		self.state = tensor

		#calculate the overall payoff for each resources
		payoffs = []
		for i in range(self.numOfResources):
			longSide = 0
			shortSide = 0
			for decisionVector in tensor:
				if decisionVector[i] > 1:
					longSide += 1
				elif decisionVector[i] < 1:
					shortSide += 1

			longSide = longSide / float(self.numOfPlayers)
			shortSide = shortSide / float(self.numOfPlayers)
			participation = longSide

			self.market[i]["short"].append(shortSide)
			self.market[i]["long"].append(longSide)

			if self.lowThreshold <= longSide <= self.highThreshold:
				payoff = self.highThreshold / float(participation)
			elif longSide < self.lowThreshold:
				if participation > 0:
					payoff = (participation - self.lowThreshold) / float(participation)
				else:
					payoff = participation - self.lowThreshold
			elif longSide > self.highThreshold:
				payoff = (self.highThreshold - participation) / float(self.highThreshold)

			payoffs.append(payoff)

		#use the payoff for each resouce to output a reward vector for agent
		rewards = []

		for decisionVector in tensor:
			reward = 0
			i = 0
			for decision in decisionVector:
				payoff = payoffs[i]
			if decision < 1:
				payoff *= -1
			elif decision > 1:
				payoff = payoff
			elif decision == 1:
				payoff = 0

			reward += payoff 

			i += 1

			rewards.append(reward)  

		self.score = np.add(self.score, rewards)

		return np.asarray(rewards)

	def get_state(self):

		return self.state
