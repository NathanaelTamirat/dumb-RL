import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

is_slippery=True

def load_enviroment(is_training,is_slippery):
    env=gym.make(
                 "Frozenlake-v1",
                 desc=None,
                 map_name="8x8",
                 is_slippery=is_slippery,
                 render_mode=None if is_training else "human"
                 )
    return env,env.observation_space.n,env.action_space.n

def epsilon_decay(epsilon,epsilon_min):
    if epsilon>epsilon_min:
        epsilon-=0.0001
    return epsilon

def train(
        lr, gamma, epsilon,
        epsilon_min, epsilon_decay,
        n_episodes,
):
    rewards = np.zeros(n_episodes)
    # Load the environment
    env, n_states, n_actions = load_environment(True, is_slippery)
    # Initialize Q table
    Q = np.zeros((n_states, n_actions))
    for episode in range(n_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            # Choose an action based on epsilon-greedy strategy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # Take the action, observe the reward and next state
            next_state, reward, terminated, truncated, info = env.step(action)
            rewards[episode] = reward

            # Update the Q-value using the Q-Learning update rule
            Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

            # If the next state is terminal, break the loop
            done = terminated or truncated
            state = next_state

        # Perform epsilon decay
        epsilon = epsilon_decay(epsilon, epsilon_min)
        if epsilon < epsilon_min:
            # Stabilize the Q value in exploitation state
            lr = 0.0001

        # Print log message
        if episode > 0 and episode % 1000 == 0:
            print(
                f"Episode: {episode:5d}/{n_episodes}, Epsilon: {epsilon:.5f}), Accuracy (last 100 episodes): {np.mean(rewards[episode - 100:episode]) * 100}%")

    env.close()

    return Q, rewards
# Hyper Parameters
lr = 0.9    # Learning rate alpha
gamma = 0.9 # Discount factor gamma

epsilon = 1  # Exploration rate epsilon
epsilon_min = 0

n_episodes = 15000

# Train the agent
Q, rewards = train(
    lr, gamma, epsilon,
    epsilon_min, epsilon_decay,
    n_episodes
)

# Save the Q-table
np.save('./output/frozen_lake_q_table.npy', Q)

# Plot the rewards
sum_rewards = np.zeros(n_episodes)
for i in range(n_episodes):
    # Calculate the sum of rewards for every 100 episodes
    sum_rewards[i] = np.sum(rewards[max(0, i - 100):(i + 1)])
plt.title('Frozen Lake - Q-Learning')
plt.plot(sum_rewards)
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards (last 100 episodes)')
plt.savefig('./output/frozen_lake_rewards.png')

# Evaluate the agent
Q = np.load('output/frozen_lake_q_table.npy')
# Load the environment
env, n_states, n_actions = load_environment(False, is_slippery)
state = env.reset()[0]
done = False
while not done:
    env.render()
    action = np.argmax(Q[state, :]) #all action at that state
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    print(info)

env.close()