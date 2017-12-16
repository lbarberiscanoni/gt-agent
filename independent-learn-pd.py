import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.reset_default_graph()

def rewards(a1, a2):
  """Computes rewards for two actions."""
  return tf.case({
      (tf.equal(a1, 1) & tf.equal(a2, 1)): lambda: (5.0, 5.0),
      (tf.equal(a1, 0) & tf.equal(a2, 1)): lambda: (3.0, -3.0),
      (tf.equal(a1, 1) & tf.equal(a2, 0)): lambda: (-3.0, 3.0),
      (tf.equal(a1, 0) & tf.equal(a2, 0)): lambda: (1.0, 1.0),
  }, exclusive=True)

# Prepare the policies.
logit_1 = tf.get_variable('logit_1', (), 
                          initializer=tf.random_normal_initializer(0.01))
logit_2 = tf.get_variable('logit_2', (), 
                          initializer=tf.random_normal_initializer(0.01))
prob_1, prob_2 = tf.sigmoid(logit_1), tf.sigmoid(logit_2)
policy_1 = tf.distributions.Bernoulli(logits=logit_1)
policy_2 = tf.distributions.Bernoulli(logits=logit_2)

# Sample a pair of actions.
action_1 = tf.stop_gradient(policy_1.sample())
action_2 = tf.stop_gradient(policy_2.sample())

# Compute the rewards.
reward_1, reward_2 = rewards(action_1, action_2)

# Compute the independent policy gradients loss.
loss_1 = -policy_1.log_prob(action_1) * tf.stop_gradient(reward_1)
loss_2 = -policy_2.log_prob(action_2) * tf.stop_gradient(reward_2)

# Optimize the loss.
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_ops = tf.group(
    optimizer.minimize(loss_1, var_list=[logit_1]), 
    optimizer.minimize(loss_2, var_list=[logit_2]))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
 
  # Repeatedly run the game and train the agents.
  rewards_1 = []
  rewards_2 = []
  probs_1 = []
  probs_2 = []
  num_steps = 1000
  for i in range(num_steps):
    _, reward_1_val, reward_2_val, prob_1_val, prob_2_val = sess.run(
        [train_ops, reward_1, reward_2, prob_1, prob_2])
    rewards_1.append(reward_1_val)
    rewards_2.append(reward_2_val)
    probs_1.append(prob_1_val)
    probs_2.append(prob_2_val)

  plt.figure(figsize=(16, 10))
  plt.plot(np.arange(num_steps), rewards_1, label='Agent 1 reward')
  plt.plot(np.arange(num_steps), rewards_2, label='Agent 2 reward')
  plt.title('Rewards over time')
  plt.legend()
  plt.show()

  plt.figure(figsize=(16, 10))
  plt.plot(np.arange(num_steps), probs_1, label='Agent 1 prob')
  plt.plot(np.arange(num_steps), probs_2, label='Agent 2 prob')
  plt.title('Cooperation probability over time')
  plt.legend()
  plt.show()