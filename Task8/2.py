import time

import numpy as np
import pygame
from matplotlib import pyplot as plt


def get_random_pos(dimension):
    return np.random.randint(0, dimension), np.random.randint(0, dimension)


class GridWorld:
    def __init__(self, block_size, dimension, finish_pos):
        self._dimension = dimension
        self._finish_pos = finish_pos
        self._block_size = block_size
        self._pos = get_random_pos(dimension)
        self._prev_pos = None
        self._actions = [
            self._up,
            self._left,
            self._down,
            self._right
        ]
        self._tiles = []
        self._grid = self._create_grid()
        print(self._tiles[1][1])

    def _update_pos(self):
        pygame.draw.circle(self._grid, pygame.Color(""))
        pygame.display.flip()

    def _up(self):
        if self._pos[0] > 0:
            self._pos = self._pos[0] - 1, self._pos[1]

    def _left(self):
        if self._pos[1] > 0:
            self._pos = self._pos[0], self._pos[1] - 1

    def _down(self):
        if self._pos[0] < self._dimension - 1:
            self._pos = self._pos[0] + 1, self._pos[1]

    def _right(self):
        if self._pos[1] < self._dimension - 1:
            self._pos = self._pos[0], self._pos[1] + 1

    def _create_grid(self):
        surface = pygame.display.set_mode((self._block_size * self._dimension, self._block_size * self._dimension))
        # i loops in the y direction
        for i in range(self._dimension):
            row = []
            # j loops in the x direction
            for j in range(self._dimension):
                pygame.draw.rect(surface, pygame.Color("grey"),
                                 pygame.Rect(j * self._block_size, i * self._block_size, self._block_size,
                                             self._block_size), 0)
                color = "black"
                if i == self._finish_pos[0] and j == self._finish_pos[1]:
                    color = "green"
                row.append((pygame.draw.rect(surface, pygame.Color(color),
                                            pygame.Rect(j * self._block_size + 2, i * self._block_size + 2,
                                                        self._block_size - 4,
                                                        self._block_size - 4)), color))
            self._tiles.append(row)
        print(self._tiles)
        pygame.display.flip()
        pygame.display.set_caption("Gridworld")
        return surface

    def get_random_action(self):
        return np.random.randint(0, len(self._actions))

    def num_of_actions(self):
        return len(self._actions)

    def render(self,):
        pygame.draw.circle(self._grid, pygame.Color("blue"),
                           (self._block_size * (self._pos[1] + 0.5), self._block_size * (self._pos[0] + 0.5)), 20)
        if self._prev_pos is not None:
            color = self._tiles[self._prev_pos[0]][self._prev_pos[1]][1]
            pygame.draw.circle(self._grid, color,
                               (self._block_size * (self._prev_pos[1] + 0.5), self._block_size * (self._prev_pos[0] + 0.5)), 20)
        pygame.display.flip()

    """Reward is the return value"""

    def step(self, action):
        self._prev_pos = self._pos
        self._actions[action]()
        # If the game is done or not
        if self._pos == self._finish_pos:
            return self._pos, 1, True
        else:
            return self._pos, -1, False

    def reset(self):
        self._pos = get_random_pos(self._dimension)
        return self._pos


def run_finished(q_table, k, block_size, dimension, finish_pos):
    env = GridWorld(block_size, dimension, finish_pos)
    for episode_num in range(k):
        done = False
        observation = env.reset()
        while not done:
            env.render()
            action = np.argmax(q_table[observation])
            observation, reward, done = env.step(action)
            time.sleep(0.3)

if __name__ == '__main__':

    NUM_OF_EPISODES = 30000
    DISCOUNT_FACTOR = 0.95
    LEARNING_RATE = 0.1

    eps = lambda episode: max((NUM_OF_EPISODES - 2 * episode) / NUM_OF_EPISODES, 0)

    DIMENSION = 8
    BLOCK_SIZE = 50
    FINISH_POS = (3, 2)
    env = GridWorld(50, DIMENSION, FINISH_POS)
    pygame.init()

    # TODO implement proper q-table
    q_table = np.zeros([DIMENSION] * 2 + [env.num_of_actions()])

    epochs_run_tbl = []
    for episode_num in range(NUM_OF_EPISODES):
        done = False
        observation = env.reset()
        epochs_run = 0
        while not done:
            if np.random.uniform(0, 1) < eps(episode_num):
                action = env.get_random_action()
            else:
                action = np.argmax(q_table[observation])
            new_observation, reward, done = env.step(action)
            current_q = q_table[observation + (action,)]
            max_future_q = np.max(q_table[new_observation])
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
            #print(q_table[observation])
            #print(q_table[observation + (action,)])
            #print(new_q)
            q_table[observation + (action,)] = new_q
            observation = new_observation
            epochs_run += 1
        if episode_num % 100 == 0:
            print(f"Episode num: {episode_num}, epochs run: {epochs_run},")
        epochs_run_tbl.append(epochs_run)

    plt.plot(epochs_run_tbl, label="Epochs run for each episode")
    plt.show()
    run_finished(q_table, 20, BLOCK_SIZE, DIMENSION, FINISH_POS)
    pygame.quit()
