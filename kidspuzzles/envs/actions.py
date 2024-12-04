from enum import Enum
import numpy as np

class Action(Enum):
    RIGHT0 = 0
    UP0 = 1
    LEFT0 = 2
    DOWN0 = 3

    RIGHT1 = 4
    UP1 = 5
    LEFT1 = 6
    DOWN1 = 7

    RIGHT2 = 8
    UP2 = 9
    LEFT2 = 10
    DOWN2 = 11

    RIGHT3 = 12
    UP3 = 13
    LEFT3 = 14
    DOWN3 = 15

    RIGHT4 = 16
    UP4 = 17
    LEFT4 = 18
    DOWN4 = 19

    RIGHT5 = 20
    UP5 = 21
    LEFT5 = 22
    DOWN5 = 23

    RIGHT6 = 24
    UP6 = 25
    LEFT6 = 26
    DOWN6 = 27

    RIGHT7 = 28
    UP7 = 29
    LEFT7 = 30
    DOWN7 = 31

    RIGHT8 = 32
    UP8 = 33
    LEFT8 = 34
    DOWN8 = 35

    RIGHT9 = 36
    UP9 = 37
    LEFT9 = 38
    DOWN9 = 39

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
