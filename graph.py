import pickle
import matplotlib.pyplot as plt
import numpy as np

f = open("market.pkl", "rb")

txt = pickle.load(f)

for i in range(len(txt)):

	data = txt[i]["long"]
	data2 = txt[i]["short"]

	#center = (bins[:-1] + bins[1:]) / 2
	#plt.bar(center, hist, align='center', width=width)
	# plt.hist(data, 5, facecolor='blue', align="mid")
	# plt.show()

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

	print(i + 1)
	print("bubbles")
	print("%", len(bubbles) / float(len(data)))
	print("correction")
	print("%", len(correction) / float(len(data)))
	print("growth")
	print("%", len(growth) / float(len(data)))
	print("mean:", np.mean(data))
	print("std:", np.std(data))
	print("corr:", np.corrcoef(data, data2)[1, 0])
	print("------------")

	#plotting the values
	x = [i for i in range(len(data))]
	plt.plot(x, data)
	plt.plot(x, data2)

