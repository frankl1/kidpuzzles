import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from scipy.spatial.distance import cityblock
from .actions import create_action, action_to_digit, action_to_direction
from .color import Color

class DigitsPuzzleEnv(gym.Env):
    """A 7 x 4 Grid environment for training an agent on the digit game.

    The goal is to move the digit from 0 to 9 to their respective locations at the center of the grid
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 render_mode:str = None,
                 n_digits:int = 10,
                 window_width: int = 512,
                 window_height:int = 256,
                 reward_terminate: float = 10,
                 reward_clipped: float = -1,
                 reward_enter_target_area: float = 0.1,
                 reward_exit_target_area: float = -1,
                 reward_reach_a_target_pos: float = 0.2,
                 reward_leave_a_target_pos: float = -1):
        assert 0 < n_digits < 11, "n_digits must be in [1, 10]"
        self.n_digits = n_digits

        self.reward_terminate = reward_terminate
        self.reward_clipped = reward_clipped
        self.reward_enter_target_area = reward_enter_target_area
        self.reward_exit_target_area = reward_exit_target_area
        self.reward_reach_target_pos = reward_reach_a_target_pos
        self.reward_leave_target_pos = reward_leave_a_target_pos

        self.width = 7 if self.n_digits > 5 else self.n_digits + 2  # The width of the grid
        self.height = 4 if self.n_digits > 5 else 3 # The height of the grid
        self.window_width = window_width  # The width of the PyGame window
        self.window_height = window_height  # The height of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        high_bound = np.column_stack(([self.width]*self.n_digits, [self.height]*self.n_digits))
        self.observation_space = spaces.Dict(
            {
                "digits_positions": spaces.Box(0, high_bound, shape=(self.n_digits, 2), dtype=int),
                "target_digits_positions": spaces.Box(0, high_bound, shape=(self.n_digits, 2), dtype=int),
            }
        )

        # The target digits' positions as (x, y). The desired observation to win the game
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
        ])[:self.n_digits]
        
        # The current observation as a 2s numpy array of n_digit x 2
        self._digits_positions: np.array = None

        actions = create_action(self.n_digits)

        self.action_space = spaces.Discrete(len(actions))

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

    def get_distance(self, normalize:bool = True):
        """Return the Manhattan distance between the current position and the target position

        Args:
            normalize (bool): weither to return the normalized distance
        Returns:
            int: the distance
        """
        dist =  cityblock(
                self._digits_positions.flatten(), self._target_digits_positions.flatten()
            )
        if normalize:
            dist /= (self.width + self.height - 2)
            dist /= self.n_digits
        
        return dist
    
    def _get_info(self):
    
        return {
            "distance": self.get_distance(normalize=False)
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

            # Randomly sample n_digits border cells
            self._digits_positions = border_cells[np.random.choice(border_cells.shape[0], self.n_digits, replace=False)]

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

        # We use `np.clip` to make sure we don't leave the grid
        curr_digit_pos = self._digits_positions[digit]
        next_digit_pos = curr_digit_pos + direction
        self._digits_positions[digit] = np.clip(
            next_digit_pos, 0, (self.width - 1, self.height - 1)
        )

        clipped = not np.array_equal(next_digit_pos, self._digits_positions[digit])

        reward, terminated = self.get_reward(curr_digit_pos,
                                             self._digits_positions[digit],
                                             self._target_digits_positions[digit],
                                             clipped=clipped)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def get_reward(self, curr_digit_pos: list[int, int], next_digit_pos: list[int, int], target_digit_pos: list[int, int], clipped: bool = False):
        """Compute the reward of an action

        Args:
            curr_digit_pos (list[int, int]): the position of the digit before the action.
            next_digit_pos (list[int, int]): the position of the digit after the action.
            target_digit_pos (list[int, int]): the target position of the digit.
            clipped (bool, optional): wether the position has been clipped or not. Defaults to False.

        Returns:
            (float, bool): a tuple with the reward and a boolean telling if the episode is terminated.
        """
        reward, terminated = 0, False
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._digits_positions, self._target_digits_positions)
        if terminated:
            reward = self.reward_terminate # generous reward if terminated
        else:
            reward = -self.get_distance()
            if clipped:
                reward += self.reward_clipped # penalty for going out of the world
            else:
                # Penalize the agent for moving a digit from the target area to 
                # borders of the world as this action can never improve the policy.
                # Encourage the agent to move digit inside the target area by adding a bonus.
                next_x, next_y = next_digit_pos
                curr_x, curr_y = curr_digit_pos
                next_on_boundaries = next_x in (0, self.width - 1) or next_y in (0, self.height - 1)
                curr_on_boundaries = curr_x in (0, self.width - 1) or curr_y in (0, self.height - 1)
                if next_on_boundaries and not curr_on_boundaries: # exit target area
                    reward += self.reward_exit_target_area
                elif curr_on_boundaries and not next_on_boundaries: # entered target area
                    reward += self.reward_enter_target_area
                elif np.array_equal(target_digit_pos, next_digit_pos): # the digit reaches its target location
                    reward += self.reward_reach_target_pos
                elif np.array_equal(target_digit_pos, curr_digit_pos): # the digit moves away from its target location
                    reward += self.reward_leave_target_pos
                else: # stayed in the same area
                    pass
        
        return reward, terminated

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        def create_digit_img(digit: int, color: Color = Color.BLUE) -> pygame.Surface:
            font = pygame.font.SysFont(None, 62)
            img = font.render(str(digit), True, color)
            return img 
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(Color.WHITE)
        cell_size = np.array([
            self.window_width / self.width,
            self.window_height / self.height
        ])  # The size of a single grid rectangle in pixels

        # fill the target area in Orange
        canvas.fill(Color.ORANGE, pygame.Rect(cell_size, (cell_size[0]*(self.width-2), cell_size[1]*(self.height-2))))

        # First we draw the targets's positions
        for digit in range(self.n_digits):
            digit_img = create_digit_img(digit, Color.WHITE)
            digit_img_size = np.array(digit_img.get_size())
            offset = (cell_size - digit_img_size) / 2
            canvas.blit(digit_img, (cell_size * self._target_digits_positions[digit] + offset))

        # Now we draw the digits
        for digit in range(self.n_digits):
            digit_img = create_digit_img(digit)
            digit_img_size = np.array(digit_img.get_size())
            offset = (cell_size - digit_img_size) / 2

            canvas.blit(digit_img, (cell_size * self._digits_positions[digit]) + offset)

        # add some horizontal lines
        for y in range(1, self.height):
            pygame.draw.line(
                canvas,
                0,
                (0, cell_size[1] * y),
                (self.window_width, cell_size[1] * y),
                width=3,
            )

        # add some vertical lines
        for x in range(1, self.width):
            pygame.draw.line(
                canvas,
                0,
                (cell_size[0] * x, 0),
                (cell_size[0] * x, self.window_height),
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
