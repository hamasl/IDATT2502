import time

import numpy as np
import pygame
from matplotlib import pyplot as plt


def get_random_pos(dimension):
    return {"x": np.random.randint(0, dimension), "y": np.random.randint(0, dimension)}


class GridWorld:
    def __init__(self, block_size, dimension, finish_pos):
        self._dimension = dimension
        self.finish_pos = finish_pos
        self._block_size = block_size
        self.pos = get_random_pos(dimension)
        self.actions = [
            self._up,
            self._left,
            self._down,
            self._right
        ]
        self._tiles = []
        self._grid = self._create_grid()

    def _update_pos(self):
        pygame.draw.circle(self._grid, pygame.Color(""))
        pygame.display.flip()

    def _up(self):
        if self.pos["y"] > 0:
            self.pos["y"] -= 1

    def _left(self):
        if self.pos["x"] > 0:
            self.pos["x"] -= 1

    def _down(self):
        if self.pos["y"] < self._dimension - 1:
            self.pos["y"] += 1

    def _right(self):
        if self.pos["x"] < self._dimension - 1:
            self.pos["x"] += 1

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
                if j == self.finish_pos["x"] and i == self.finish_pos["y"]:
                    row.append(pygame.draw.rect(surface, pygame.Color("green"),
                                                pygame.Rect(j * self._block_size + 2, i * self._block_size + 2,
                                                            self._block_size - 4,
                                                            self._block_size - 4)))
                else:
                    row.append(pygame.draw.rect(surface, pygame.Color("black"),
                                                pygame.Rect(j * self._block_size + 2, i * self._block_size + 2,
                                                            self._block_size - 4,
                                                            self._block_size - 4)))
            self._tiles.append(row)
        print(self._tiles)
        pygame.display.flip()
        pygame.display.set_caption("Gridworld")
        return surface

    def render(self, old_pos=None):
        pygame.draw.circle(self._grid, pygame.Color("blue"),
                           (self._block_size * (self.pos["x"] + 0.5), self._block_size * (self.pos["y"] + 0.5)), 20)
        if old_pos is not None:
            color = self._grid[old_pos["y"]][old_pos["x"]].color
            pygame.draw.circle(self._grid, color, (self._block_size * (old_pos["x"] + 0.5), self._block_size * (old_pos["y"] + 0.5)), 20)
        pygame.display.flip()

    """Reward is the return value"""

    def step(self, action):
        old_pos = self.pos.copy()
        self.actions[action]()
        self._render_step(old_pos)
        # If the game is done or not
        if self.pos == self.finish_pos:
            return 1, True
        else:
            return -1, False

    def reset(self):
        self.pos = get_random_pos(self._dimension)


if __name__ == '__main__':

    NUM_OF_BUCKETS = 20
    NUM_OF_EPISODES = 10000
    DISCOUNT_FACTOR = 0.95
    LEARNING_RATE = 0.1

    eps = lambda episode: max((NUM_OF_EPISODES - 2 * episode) / NUM_OF_EPISODES, 0)

    env = GridWorld(50, 8, {"x": 2, "y": 3})
    pygame.init()

    #TODO implement proper q-table
    q_table = []

    epochs_run_tbl = []
    for episode_num in range(NUM_OF_EPISODES):
        done = False
        observation = env.reset()
        epochs_run = 0
        while not done:
            if np.random.uniform(0, 1) < eps(episode_num):
                action = env.action_space.sample()
            else:
                action = q_table.optimal_choice(observation)
            new_observation, reward, done, info = env.step(action)
            if done and epochs_run < 150:
                reward = -200
            current_q = q_table.get_q_value(observation, action)
            max_future_q = q_table.optimal_val(new_observation)
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
            q_table.set_q_value(observation, action, new_q)
            observation = new_observation
            epochs_run += 1
        if episode_num % 100 == 0:
            print(f"Episode num: {episode_num}, epochs run: {epochs_run},")
        epochs_run_tbl.append(epochs_run)
    plt.plot(epochs_run_tbl, label="Epochs run for each episode")
    plt.show()
    run_finished(q_table, 20)
    pygame.quit()
