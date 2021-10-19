# Inspired by: https://github.com/JackFurby/CartPole-v0
import gym
import numpy as np


# TODO create bins for the observations
class Bucketed_Q_Table():
    def __init__(self, observation_space, action_space, num_of_buckets):
        #self._buckets = self._create_buckets(observation_space, num_of_buckets)
        self._buckets = self._create_buckets_v2()
        self._table = np.zeros(([num_of_buckets] * len(observation_space.high) + [action_space.n]))
        print(self._buckets)
        #print(self._table.shape)

    def _create_buckets(self, observation_space, num_of_buckets):
        num_of_observations = len(observation_space.high)
        #print(observation_space.high)
        #print(observation_space.low)
        bucks = []
        for i in range(num_of_observations):
            # Since a bucket contains values between linspace entries the linspace needs one more element than the number of buckets
            bucks.append(np.linspace(observation_space.low[i], observation_space.high[i], num_of_buckets + 1))
        return bucks

    def _create_buckets_v2(self):
        numBins = 20
        obsSpaceSize = len(env.observation_space.high)

        # Get the size of each bucket
        return [
            np.linspace(-4.8, 4.8, numBins),
            np.linspace(-4, 4, numBins),
            np.linspace(-.418, .418, numBins),
            np.linspace(-4, 4, numBins)
        ]

    def _get_bucket_index(self, observation_num, observation):
        for i in range(len(self._buckets[observation_num])-1):
            if i == 20:
                print(observation)
                print(observation_num)
            if self._buckets[observation_num][i] <= observation < self._buckets[observation_num][i + 1]:
                return i
        print("OBS")
        return -1

    def optimal_choice(self, observations):
        return np.argmax(self._get_action_q_values(observations))

    def optimal_val(self, observations):
        return np.max(self._get_action_q_values(observations))

    def _get_action_q_values(self, observations):
        arr = self._table
        for i in range(len(observations)):
            arr = arr[self._get_bucket_index(i, observations[i])]
        return arr

    def set_q_value(self, observations, action_index, q_value):
        self._get_action_q_values(observations)[action_index] = q_value
        # self._table[observation_index, self._get_bucket_index(observation_index, observations), action_index] = q_value

    def get_q_value(self, observations, action_index):
        return self._get_action_q_values(observations)[action_index]
        # return self._table[observation_index, self._get_bucket_index(observation_index, observation), action_index]

    def print_table(self):
        print(self._table)

    def get_max(self):
        return np.max(self._table)


def run_finished(q_table, k):
    env = gym.make("CartPole-v0")
    print("Running final")
    for episode_num in range(k):
        done = False
        observation = env.reset()
        while not done:
            env.render()
            action = q_table.optimal_choice(observation)
            observation, reward, done, info = env.step(action)
    env.close()

if __name__ == '__main__':
    NUM_OF_BUCKETS = 20
    NUM_OF_EPISODES = 10000
    DISCOUNT_FACTOR = 0.95
    LEARNING_RATE = 0.1
    eps = lambda episode: 0.3*(NUM_OF_EPISODES-episode)

    env = gym.make("CartPole-v0")

    q_table = Bucketed_Q_Table(env.observation_space, env.action_space, NUM_OF_BUCKETS)
    """""
    """""
    #q_table.print_table()
    epochs_run_tbl = []
    for episode_num in range(NUM_OF_EPISODES):
        done = False
        observation = env.reset()
        epochs_run = 0
        #TODO not using new and old state
        while not done:
            #env.render()
            # TODO fix if
            if np.random.uniform(0, 1) < eps(episode_num):
                action = env.action_space.sample()
            else:
                action = q_table.optimal_choice(observation)
            new_observation, reward, done, info = env.step(action)

            current_q = q_table.get_q_value(observation, action)
            max_future_q = q_table.optimal_val(new_observation)
            # TODO this formula is technically not right
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
            q_table.set_q_value(observation, action, new_q)
            observation = new_observation
            epochs_run += 1
            #print(action)
            # print(current_q)
            #print(max_q)
            #print(new_q)
        if episode_num % 100 == 0:
            print(f"Episode num: {episode_num}, epochs run: {epochs_run},")
            print(q_table.get_max())
        epochs_run_tbl.append(epochs_run)
    q_table.print_table()
    env.close()
    run_finished(q_table, 20)
