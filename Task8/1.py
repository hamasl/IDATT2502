import gym
import numpy as np

#TODO create bins for the observations
class Bucketed_Q_Table():
    def __init__(self, observation_space, action_space, num_of_buckets):
        self._buckets = self._create_buckets(observation_space, num_of_buckets)
        self._table = np.zeros((len(observation_space.high), num_of_buckets, action_space.n))
        print(self._buckets)
        print(self._table)

    def _create_buckets(self, observation_space, num_of_buckets):
        num_of_observations = len(observation_space.high)
        bucks = []
        for i in range(num_of_observations):
            #Since a bucket contains values between linspace entries the linspace needs one more element than the number of buckets
            bucks.append(np.linspace(observation_space.low[i], observation_space.high[i], num_of_buckets+1))
        return bucks

    def _get_bucket_index(self, observation_num, observation):
        for i in range(len(self._buckets[observation_num]-1)):
            if self._buckets[observation_num][i] <= observation < self._buckets[observation_num][i+1]:
                return i
        return -1

    def argmax(self):
        return self._table.argmax()

    def set_q_value(self, observation_index, observation, action_index, q_value):
        self._table[observation_index, self._get_bucket_index(observation_index, observation), action_index] = q_value

    def get_q_value(self, observation_index, observation, action_index):
        return self._table[observation_index, self._get_bucket_index(observation_index, observation), action_index]


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    num_of_buckets = 20
    num_of_episodes = 80
    q_table = Bucketed_Q_Table(env.observation_space, env.action_space, num_of_buckets)
    for episode_num in range(num_of_episodes):
        done = False
        observation = env.reset()
        while not done:
            env.render()
            # TODO use q-table to decide on action, and update it after every action
            observation, reward, done, info = env.step(env.action_space.sample())

    """""
    for episode_num in range(80):
        observation = env.reset()
        for _ in range(100):
            env.render()
            #sample() returns 0 or 1, where 0 is to the left and 1 is to the rigth
            observation, reward, done, info = env.step(env.action_space.sample())
            print(observation, reward)
            
            #Update q value
            if done:
                break
    """
    env.close()
