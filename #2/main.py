import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils


def observe_taxi():
    env = gym.make("Taxi-v3", render_mode="ansi")  # human

    # information
    action_size = env.action_space.n
    print("Action size: ", action_size)  # 6
    state_size = env.observation_space.n
    print("State size: ", state_size)  # 500

    env.reset()
    terminated = False
    truncated = False
    print(env.render())

    while not terminated or not truncated:
        # perform a random action
        action = env.action_space.sample()
        # terminated: whether a `terminal state` is reached.
        # truncated: whether a truncation condition outside the scope is satisfied.
        new_state, reward, terminated, truncated, info = env.step(action)
        print("Action taken:", action)
        print("Reward earned:", reward)
        print(env.render())
        # env.render()


# This function is taken from the lecture notes
def eval_policy_better(env_, pi_, gamma_, t_max_, episodes_):
    v_pi_rep = np.empty(episodes_)
    for e in range(episodes_):
        s_t = env_.reset()
        s_t = s_t[0]
        v_pi = 0
        for t in range(t_max_):
            a_t = pi_[s_t]
            s_t, r_t, terminated, truncated, info = env_.step(a_t)
            v_pi += gamma_ ** t * r_t
            if terminated:
                break
        v_pi_rep[e] = v_pi
        env_.close()
    return np.mean(v_pi_rep), np.min(v_pi_rep), np.max(v_pi_rep), np.std(v_pi_rep)


def q_learning():
    env = gym.make("Taxi-v3", render_mode="ansi")  # human
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))

    epsilon = 0.99  # e-greedy
    alpha = 0.5  # learning rate - 1.
    gamma = 0.9  # reward decay rate

    episodes = 3000  # num of training episodes
    interactions = 300  # max num of interactions per episode
    total_rewards = np.zeros(episodes)

    hist = []  # evaluation history

    for episode in range(episodes):
        state = env.reset()
        state = state[0]
        episode_reward = 0

        for interact in range(interactions):
            # exploitation vs. exploratin by e-greedy sampling of actions
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                action = np.random.randint(0, 5)

            # Observe
            # terminated: whether a `terminal state` is reached.
            # truncated: whether a truncation condition outside the scope is satisfied.
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q-table
            qtable[state, action] = qtable[state, action] + alpha * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

            # update state
            state = new_state
            episode_reward = episode_reward + reward

            # Check if terminated
            # it indicates that the episode is over and the agent has reached a terminal state
            # the agent moves on to the next episode.
            if terminated:
                # print("In episode", {episode + 1}, "terminated ")
                break

        total_rewards[episode] = episode_reward

        # Now, I have some qtable and using that find some insights about current performance
        if episode % 10 == 0 or episode == 1:
            # print("Episode", (episode + 1), "/", episodes, ", Reward: ", episode_reward)
            # get the optimal table
            pi = np.argmax(qtable, axis=1)
            val_mean, val_min, val_max, val_std = eval_policy_better(env, pi, gamma, interactions, interactions)
            hist.append([episode, val_mean, val_min, val_max, val_std])

    env.reset()

    # Learning is done
    # Print the optimal policy, it is done by finding the action that give the maximum value for each state
    policy = np.argmax(qtable, axis=1)
    # print("Optimal policy:", policy)

    return hist, policy


def plotting(hist):
    hist = np.array(hist)
    print(hist.shape)
    # how the mean value/standard deviation function changes over time as the agent learns to make better decisions
    plt.plot(hist[:, 0], hist[:, 1])
    plt.title("Mean vs Episodes")
    plt.xlabel("Number of episodes")
    plt.ylabel("Mean value")
    plt.show()

    plt.plot(hist[:, 0], hist[:, 4])
    plt.title("Standard deviation vs Episodes")
    plt.xlabel("Number of episodes")
    plt.ylabel("Standard deviation")
    plt.show()

    hist = np.array(hist)
    # print(hist.shape)

    plt.plot(hist[:, 0], hist[:, 1])
    # plt.fill_between(hist[:,0], np.maximum(hist[:,1]-hist[:,4],np.zeros(hist.shape[0])),hist[:,1]+hist[:,4],
    plt.fill_between(hist[:, 0], hist[:, 1] - hist[:, 4], hist[:, 1] + hist[:, 4],
                     alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
                     linewidth=0)
    plt.show()


# print best average reward and standard deviation using optimal q table while playing the game
def play_game(opt_policy):
    env_ = gym.make("Taxi-v3", render_mode="ansi")  # human
    gamma = 0.9  # reward decay rate

    episodes = 1100  # num of training episodes
    interactions = 100  # max num of interactions per episode

    v_pi_rep = np.empty(episodes)
    for e in range(episodes):
        s_t = env_.reset()
        s_t = s_t[0]
        v_pi = 0
        for t in range(interactions):
            a_t = opt_policy[s_t]
            s_t, r_t, terminated, truncated, info = env_.step(a_t)
            v_pi += gamma ** t * r_t
            if terminated:
                break
        v_pi_rep[e] = v_pi
        env_.close()

    env_.reset()
    return np.mean(v_pi_rep), np.std(v_pi_rep), np.min(v_pi_rep), np.max(v_pi_rep)


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(500)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Dense(6, activation='linear')
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  )
    model.summary()

    return model


def plotting_2(hist):
    hist = np.array(hist)
    print(hist.shape)
    # how the mean value/standard deviation function changes over time as the agent learns to make better decisions
    plt.plot(hist[:, 0], hist[:, 1])
    plt.title("Mean vs No of Training Samples")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Mean value")
    plt.show()

    plt.plot(hist[:, 0], hist[:, 4])
    plt.title("Standard deviation vs No of Training Samples")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Standard deviation")
    plt.show()

    hist = np.array(hist)
    # print(hist.shape)

    plt.plot(hist[:, 0], hist[:, 1])
    # plt.fill_between(hist[:,0], np.maximum(hist[:,1]-hist[:,4],np.zeros(hist.shape[0])),hist[:,1]+hist[:,4],
    plt.fill_between(hist[:, 0], hist[:, 1] - hist[:, 4], hist[:, 1] + hist[:, 4],
                     alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
                     linewidth=0)
    plt.show()


def q_network(opt_policy):
    env = gym.make("Taxi-v3", render_mode="ansi")  # human
    model = create_model()

    # information
    action_size = env.action_space.n
    # print("Action size: ", action_size)  # 6
    state_size = env.observation_space.n
    # print("State size: ", state_size)  # 500
    hist = []  # evaluation history

    # print("opt_policy : ", opt_policy)
    # print("opt_policy.shape before: ", opt_policy.shape)  # (500,)
    opt_policy = opt_policy.reshape(-1, 1)
    # print("opt_policy after : ", opt_policy)
    # print("opt_policy.shape after: ", opt_policy.shape)  # (500, 1)

    # create x from optimal Q table
    x = np.arange(0, state_size)
    x = np_utils.to_categorical(x, num_classes=500, dtype='float32')
    print("x.shape: ", x.shape)  # expected (500, 500), got  (500, 500)

    # create y values from optimal Q table
    # max_index = np.argmax(opt_policy, axis=1)
    one_hot_encoded_rows = np.eye(action_size)[opt_policy.flatten()]
    y = one_hot_encoded_rows
    print("y.shape: ", y.shape)  # expected (500, 6), got (500, 6)

    model.fit(x, y, batch_size=1, epochs=10)  # try changing epochs

    ## this is for the evaluation
    # for i in range(0,499,10):
    #    model = create_model()
    #    model.fit(x[0:i+1,:], y[0:i+1,:], batch_size=1, epochs=1)
    #    # evaluate network
    #    mean_r, std_r, min_r, max_r = play_game_network(model)
    #    hist.append([i, mean_r, min_r, max_r, std_r])

    return model, hist


def play_game_network(model):
    env_ = gym.make("Taxi-v3", render_mode="ansi")  # human
    episodes = 1100  # num of training episodes
    interactions = 100  # max num of interactions per episode
    rewards = []

    for episode in range(episodes):
        print("episode", episode, "/", episodes)
        state = env_.reset()
        state = state[0]
        total_rewards = 0

        for step in range(interactions):
            # state = np.array(np.eye(500)[state])
            # print("state", state)
            state = np.array(state)
            # state = state.reshape(1, -1)
            # state = np.eye(500)[state]
            # state = state.reshape(1, 500)
            state = np_utils.to_categorical(state, num_classes=500, dtype='float32')
            state = state.reshape(1, 500)

            # print("state.shape", state.shape)  # state.shape (500,)
            action = np.argmax(model.predict(state))
            new_state, reward, terminated, truncated, info = env_.step(action)
            total_rewards += reward

            if terminated:
                break
            state = new_state
        rewards.append(total_rewards)

    env_.close()
    # print(rewards)
    return sum(rewards) / episodes, np.std(rewards), np.min(rewards), np.max(rewards)


if __name__ == '__main__':
    # task 1 q-learning
    print("Task 1 starts")
    # observe_taxi()
    history, optimal_policy = q_learning()
    plotting(history)
    average_reward, std_dev_reward, min_reward, max_reward = play_game(optimal_policy)
    print("Part 1 , Q-learning evaluation, Average of rewards: ", average_reward)
    print("Part 1 , Q-learning evaluation, Standard deviation of rewards: ", std_dev_reward)
    print("Part 1 , Q-learning evaluation, Min of rewards: ", min_reward)
    print("Part 1 , Q-learning evaluation, Max of rewards: ", max_reward)

    # task 2
    # part 1 q-network
    print("Task 2 starts")
    model_1, hist_1 = q_network(optimal_policy)
    # plotting_2(hist_1)
    avg_reward_1, std_reward_1, min_reward_1, max_reward_1 = play_game_network(model_1)
    print("Part 2, Q-Network evaluation (sampling the Q-table), Average of rewards: ", avg_reward_1)
    print("Part 2, Q-Network evaluation (sampling the Q-table), Standard deviation of rewards: ", std_reward_1)
    print("Part 2, Q-Network evaluation (sampling the Q-table), Min of rewards: ", min_reward_1)
    print("Part 2, Q-Network evaluation (sampling the Q-table), Max of rewards: ", max_reward_1)
