# to run python exercise_7.py

# İlknur Baş
# Student Number 151226814
# DATA.ML.100 Introduction to Pattern Recognition and Machine Learning
# Exercise - Week 7: Reinforcement learning

import gym
import numpy as np
import matplotlib.pyplot as plt


def create_env(slippery):
    # environment has 4 actions, 16 state
    environment = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode="ansi")
    return environment
    # print(env.render())  # env.render()


def deterministic(env):
    # parameters
    gamma = 0.9  # discount factor
    no_episodes = 1000  # 1000
    no_steps = 100  # 100
    average_reward = []

    # Q-table for total rewards
    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # 16x4

    # Q-learning algorithm
    for episodes in range(no_episodes):
        state = env.reset()
        state = state[0]
        for steps in range(no_steps):
            random_action = env.action_space.sample()  # chooses a random action
            # Take the action
            # terminated: whether a `terminal state` is reached.
            # truncated: whether a truncation condition outside the scope is satisfied.
            # can be used to end the episode prematurely before a `terminal state` is reached.
            new_state, reward, terminated, truncated, info = env.step(random_action)
            # uncomment for debugging purposes
            # print(f'state: {state}')
            # print(f'random_action: {random_action}')
            # print(f'{new_state, reward, terminated, truncated}')

            # deterministic version
            # Q(s_t,a_t) = r_t + gamma * max(Q(s_t+1,a_t))
            max_row = np.max(q_table[new_state, :])  # max(Q(s_t+1,a_t))
            q_table[state, random_action] = reward + (gamma * max_row)
            state = new_state

            if terminated:
                break
        if episodes % 50 == 0:
        # if episodes % 6 == 0:
            # print(f'--{episodes}. episode is done')
            # print(f'{reward}. reward')
            # print(f'{q_table}. q_table')
            temp_average_reward = reward_evaluate(q_table, env)
            # print(f'{temp_average_reward}. average_reward')
            average_reward.append(temp_average_reward)

    return average_reward


def reward_evaluate(q_table, env):
    no_episodes = 1000  # 1000
    no_steps = 100  # 100
    reward_evaluate = []

    for episodes in range(no_episodes):
        state = env.reset()
        steps = 0
        state = state[0]
        terminated = False
        total_reward_evaluate = 0

        for steps in range(no_steps):
            action = np.argmax(q_table[state, :])  # chooses action that has a max value
            new_state, reward, terminated, truncated, info = env.step(action)
            total_reward_evaluate += reward

            if terminated:
                reward_evaluate.append(total_reward_evaluate)
                break
            state = new_state
    env.close()

    return sum(reward_evaluate)/no_episodes


def non_deterministic(env):
    # parameters
    gamma = 0.9  # discount factor
    alpha = 0.5  # learning rate, used in non-deterministic
    no_episodes = 1000  # 1000
    no_steps = 100  # 100
    average_reward_non = []

    # Q-table for total rewards
    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # 16x4

    # Q-learning algorithm
    for episodes in range(no_episodes):
        state = env.reset()
        state = state[0]
        for steps in range(no_steps):
            random_action = env.action_space.sample()  # chooses a random action
            # Take the action
            # terminated: whether a `terminal state` is reached.
            # truncated: whether a truncation condition outside the scope is satisfied.
            # can be used to end the episode prematurely before a `terminal state` is reached.
            new_state, reward, terminated, truncated, info = env.step(random_action)
            # uncomment for debugging purposes
            # print(f'state: {state}')
            # print(f'random_action: {random_action}')
            # print(f'{new_state, reward, terminated, truncated}')

            # non-deterministic version
            # Q(s_t,a_t) = Q(s_t,a_t) + alpha * [ r_t + gamma * max(Q(s_t+1,a_t)) -  Q(s_t,a_t) ]
            max_row = np.max(q_table[new_state, :])  # max(Q(s_t+1,a_t))
            q_table[state, random_action] += alpha * (reward + (gamma * max_row) - q_table[state, random_action])
            state = new_state

            if terminated:  # breakM steps taken?????????????
                break

        if episodes % 50 == 0 :
            # print(f'--{episodes}. episode is done')
            # print(f'{reward}. reward')
            # print(f'{q_table}. q_table')
            temp = reward_evaluate(q_table, env)
            # print(f'{temp_average_reward}. average_reward')
            average_reward_non.append(temp)

    return average_reward_non


# main
print(f'Plotting 1...')
# deterministic version non-slippery
env_deterministic = create_env(False)
plt.figure(1, figsize=(14,5))
x = np.linspace(0, 1000, 20,  dtype=int)
# x = np.linspace(0, 300, 50,  dtype=int)
# print(f'x: {x.shape}')
for i in range(10):
    average_reward_single_deterministic_1 = deterministic(env_deterministic)  # 10 eval values for 1 q-table
    # print(f'{len(average_reward_single_deterministic_1)}') # 20
    plt.plot(x, average_reward_single_deterministic_1)
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], loc='best')
plt.title("deterministic version non-slippery")
plt.show()
env_deterministic.close()

# deterministic version slippery
print(f'Plotting 2...')
env_deterministic = create_env(True)
# plt.figure(2)
plt.figure(2, figsize=(14,5))
# x = np.arange(0, 9, 1)
x = np.linspace(0, 1000, 20,  dtype=int)
for i in range(10):
    average_reward_single_deterministic_2 = deterministic(env_deterministic)  # 10 eval values for 1 q-table
    plt.plot(x, average_reward_single_deterministic_2)
    # print(f'{average_reward_single_deterministic_2}')
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], loc='best')
plt.title("deterministic version slippery")
plt.show()
env_deterministic.close()

# non_deterministic version slippery
print(f'Plotting 3...')
env_non_deterministic = create_env(True)
plt.figure(3, figsize=(14,5))
# x = np.arange(0, 9, 1)
x = np.linspace(0, 1000, 20,  dtype=int)
for i in range(10):
    average_reward_single_non_deterministic = non_deterministic(env_non_deterministic)  # 10 eval values for 1 q-table
    # print(f'{average_reward_single_non_deterministic}')
    plt.plot(x, average_reward_single_non_deterministic)
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], loc='best')
plt.title("non_deterministic version slippery")
plt.show()
