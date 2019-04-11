import subprocess
import itertools

# combos = [
# 	"trpo", "vpg", "dqn", "trpo vpg", "vpg dqn", "dqn", "trpo", "trpo vpg dqn"
# ]

combos = [
	"vpg", "dqn", "trpo vpg", "vpg dqn", "dqn", "trpo", "trpo vpg dqn"
]

resources = [1, 5, 10]
# thresholds = [
# 	[.7, .3], [.6, .4], [.8, .2]
# ]

for resource in resources:
	for combo in combos:
		print(resource, combo)
		subprocess.call("python factory.py --population=12 --resources=" + str(resource) + " --agent " + combo + " --ceiling=.7 --floor=.3", shell=True)