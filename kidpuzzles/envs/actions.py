from enum import Enum
import numpy as np

def create_action(n_digits: int):
    """Create an Enum for the actions up to n_digits

    Args:
        n_digits (int): The number of digits to create actions for.

    Returns:
        CustomAction: an Enum named Action
    """
    item_dicts = [{f"RIGHT{i}": 0+i*4, f"UP{i}": 1+i*4, f"LEFT{i}": 2+i*4, f"DOWN{i}": 3+i*4} for i in range(n_digits)]
    items = {k: v for d in item_dicts for k, v in d.items()}
    return Enum("Action", items)

def action_to_digit(action_val: int) -> int:
    """Returns the digit corresponding to the action value

    Args:
        action_num (int): the action value

    Returns:
        int: the corresponding digit
    """
    return action_val // 4 

def action_to_direction (action_val: int) -> np.array:
    """Returns the direction of the action

    Args:
        action_val (int): the action value

    Returns:
        np.array: the direction to move the corresponding digit
    """
    directions = {
        0: np.array([1, 0]), # right
        1: np.array([0, -1]), # up
        2: np.array([-1, 0]), # left
        3: np.array([0, 1]), # down
    }

    direction = action_val % 4
    return directions[direction]
