import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import math

fileList = [x for x in os.listdir() if ".pkl" in x]

def breakdown():

	for file_name in fileList:

		f = open(file_name, "rb")

		txt = pickle.load(f)

		overallBubbles = []
		overallCorrections = []
		overallGrowth = []

		for i in range(len(txt)):

			data = txt[i]["long"]
			data2 = txt[i]["short"]

			bubbles = []
			correction = []
			growth = []

			for x in data:
				if x >= .7:
					bubbles.append(x)
				elif x <= .3:
					correction.append(x)
				else:
					growth.append(x)

		# 	# print(i + 1)
		# 	# print("bubbles %", len(bubbles) * 100 / float(len(data)))
		# 	# print("correction %", len(correction) * 100  / float(len(data)))
		# 	# print("growth %", len(growth) * 100 / float(len(data)))
		# 	# print("median:", np.median(data) * 100 )
		# 	# print("std:", np.std(data) * 100)
		# 	# print("corr:", np.corrcoef(data, data2)[1, 0])
		# 	# print("------------")

			overallBubbles.append(len(bubbles) * 100 / float(len(data)))
			overallCorrections.append(len(correction) * 100  / float(len(data)))
			overallGrowth.append(len(growth) * 100 / float(len(data)))


		print(file_name)
		print("bubbles", np.mean(overallBubbles))
		print("growth", np.mean(overallGrowth))
		print("corrections", np.mean(overallCorrections))
		print("---------------------")


def participation():

	for file_name in fileList:

		f = open(file_name, "rb")

		txt = pickle.load(f)

		overallShortInterest = []
	
		for i in range(len(txt)):

			longData = txt[i]["long"]
			shortData = txt[i]["short"]

			shortInterest = []
			x = []
			y = []
			for j in range(len(longData)):
				if longData[j] < .3:
					_ = shortData[j] / float(1 - longData[j])
					shortInterest.append(_)

					x.append(longData[j])
					y.append(shortData[j])

			overallShortInterest.append(np.median(shortInterest) * 100)

		overallShortInterest_clean = [x for x in overallShortInterest if math.isnan(x) == False ]
		print(overallShortInterest_clean)

		print(file_name)

		if len(overallShortInterest_clean) > 0:
			print("median", np.median(overallShortInterest_clean))
			print("mean", np.mean(overallShortInterest_clean))
			print("std", np.std(overallShortInterest_clean))
		else:
			print("median", 0.0)
			print("mean", 0.0)
			print("std", 0.0)
		print("---------")

participation()


