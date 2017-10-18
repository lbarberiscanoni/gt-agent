from tensorforce import Configuration
from tensorforce.agents import TRPOAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments import Environment

config = Configuration(
    batch_size=100,
    states=dict(shape=(10,), type='float'),
    actions=dict(continuous=False, num_actions=2),
    network=layered_network_builder([dict(type='dense', size=50), dict(type='dense', size=50)])
)

# Create a Trust Region Policy Optimization agent
agent = TRPOAgent(config=config)

# Get new data from somewhere, e.g. a client to a web app
client = MyClient('http://127.0.0.1', 8080)

# Poll new state from client
state = client.get_state()

# Get prediction from agent, execute
action = agent.act(state=state)
reward = client.execute(action)

# Add experience, agent automatically updates model according to batch size
agent.observe(reward=reward, terminal=False)
