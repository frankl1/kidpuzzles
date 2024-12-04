import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from scipy.spatial.distance import hamming
from .actions import Action, action_to_digit, action_to_direction
from .color import Color

class DigitsPuzzleEnv(gym.Env):
    """A 7 x 4 Grid environment for training an agent on the digit game.

    The goal is to move the digit from 0 to 9 to their respective locations at the center of the grid
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, window_width = 512, window_height = 256):
        self.width = 7  # The width of the grid
        self.height = 4 # The height of the grid
        self.window_width = window_width  # The width of the PyGame window
        self.window_height = window_height  # The height of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        high_bound = np.column_stack(([self.width]*10, [self.height]*10))
        self.observation_space = spaces.Dict(
            {
                "digits_positions": spaces.Box(0, high_bound, shape=(10, 2), dtype=int),
                "target_digits_positions": spaces.Box(0, high_bound, shape=(10, 2), dtype=int),
            }
        )

        # The target digits' positions as (y, x). The desired observation to win the game
        self._target_digits_positions = np.array([
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
            [5, 1],
            [1, 2],
            [2, 2],
            [3, 2],
            [4, 2],
            [5, 2],
        ])
        
        # The current observation as a 2s numpy array of 10 x 2
        self._digits_positions: np.array = None

        self.action_space = spaces.Discrete(len(Action))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "digits_positions": self._digits_positions, 
            "target_digits_positions": self._target_digits_positions,
        }

    def _get_info(self):
    
        return {
            "distance": hamming(
                self._digits_positions.flatten(), self._target_digits_positions.flatten()
            )
        }

    def reset(self, seed=None, options=None):
        def init_digits_at_borders():
             # Choose the digits' locations uniformly at random, choosing x from [0, 3] and y from [0, 6]
            # Make sure each position is used once
            grid = np.zeros((self.width, self.height), dtype=int)

            # Mark the border cells
            grid[0, :] = 1
            grid[self.width-1, :] = 1
            grid[:, 0] = 1
            grid[:, self.height-1] = 1

            # Get the coordinates of the border cells
            border_cells = np.argwhere(grid == 1)

            # Randomly sample 10 border cells
            self._digits_positions = border_cells[np.random.choice(border_cells.shape[0], 10, replace=False)]

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        init_digits_at_borders()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = action_to_direction(action)
        digit = action_to_digit(action)
        self._digits_positions[digit] = np.clip(
            self._digits_positions[digit] + direction, 0, (self.width - 1, self.height - 1)
        )
        # We use `np.clip` to make sure we don't leave the grid
        self._digits_positions = np.clip(
            self._digits_positions + direction, 0, (self.width - 1, self.height - 1)
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._digits_positions, self._target_digits_positions)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        def create_digit_img(digit: int, color: Color = Color.BLUE) -> pygame.Surface:
            font = pygame.font.SysFont(None, 62)
            img = font.render(str(digit), True, color)
            return img 
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        pix_cell_size = np.array([
            self.window_width / self.width,
            self.window_height / self.height
        ])  # The size of a single grid rectangle in pixels

        # First we draw the targets's positions
        for digit in range(10):
            digit_img = create_digit_img(digit, Color.GREY)
            digit_img_size = np.array(digit_img.get_size())
            offset = (pix_cell_size - digit_img_size) / 2
            canvas.blit(digit_img, (pix_cell_size * self._target_digits_positions[digit] + offset))

        # Now we draw the digits
        for digit in range(10):
            digit_img = create_digit_img(digit)
            digit_img_size = np.array(digit_img.get_size())
            offset = (pix_cell_size - digit_img_size) / 2

            canvas.blit(digit_img, (pix_cell_size * self._digits_positions[digit]) + offset)

        # add some horizontal lines
        for y in range(1, self.height):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_cell_size[1] * y),
                (self.window_width, pix_cell_size[1] * y),
                width=3,
            )

        # add some vertical lines
        for x in range(1, self.width):
            pygame.draw.line(
                canvas,
                0,
                (pix_cell_size[0] * x, 0),
                (pix_cell_size[0] * x, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
