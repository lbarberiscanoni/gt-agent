import pickle
import matplotlib.pyplot as plt
import numpy as np

f = open("market.pkl", "rb")

txt = pickle.load(f)

# for i in range(len(txt)):

# 	data = txt[i]["long"]
# 	data2 = txt[i]["short"]

# 	#center = (bins[:-1] + bins[1:]) / 2
# 	#plt.bar(center, hist, align='center', width=width)
# 	# plt.hist(data, 5, facecolor='blue', align="mid")
# 	# plt.show()

# 	bubbles = []
# 	correction = []
# 	growth = []

# 	for x in data:
# 		if x >= .7:
# 			bubbles.append(x)
# 		elif x <= .3:
# 			correction.append(x)
# 		else:
# 			growth.append(x)

# 	print(i + 1)
# 	print("bubbles %", len(bubbles) / float(len(data)))
# 	print("correction %", len(correction) / float(len(data)))
# 	print("growth %", len(growth) / float(len(data)))
# 	print("median:", np.median(data))
# 	print("std:", np.std(data))
# 	print("corr:", np.corrcoef(data, data2)[1, 0])
# 	print("------------")

# 	#plotting the values
# 	x = [i for i in range(len(data))]
# 	plt.scatter(data, data2)
# 	plt.savefig("resource" + str(i + 1) + ".png")

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

	print("resource", i + 1)
	print("median", np.median(shortInterest))
	print("mean", np.mean(shortInterest))
	print("std", np.std(shortInterest))
	print("corr", np.corrcoef(x, y)[1, 0])
	print("---------")


