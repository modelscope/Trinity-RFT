"""
Utils for the FrozenLake environment.
Modified from https://github.com/rllm-org/rllm/blob/main/rllm/environments/frozenlake/frozenlake.py
"""

from typing import Optional, Tuple

import numpy as np

# Map gym state in integer
MAP_LOOKUP = {
    b"P": 0,
    b"F": 1,
    b"H": 2,
    b"G": 3,
}

# Define rules to transform to rendered text observation of the environment
GRID_LOOKUP = {
    0: " P \t",  # player
    1: " _ \t",  # frozen
    2: " O \t",  # hole
    3: " G \t",  # goal
    4: " X \t",  # player fall into hole
    5: " √ \t",  # player on goal
}

ACTION_LOOKUP = {
    0: "None",
    1: "Left",
    2: "Down",
    3: "Right",
    4: "Up",
}

# Prompting format inspired by the RAGEN project: https://github.com/RAGEN-AI/RAGEN
SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""


def is_valid(board: list[list[str]], max_size: int, max_steps: int) -> bool:
    """DFS to check that it's a valid path.

    Args:
        board: The board representation as a list of lists.
        max_size: Maximum size of the board.
        max_steps: Maximum number of steps allowed.

    Returns:
        True if there's a valid path from start to goal within max_steps, False otherwise.
    """
    frontier, discovered = [], set()
    # find the start point
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0], 0))  # row, col steps
    # dfs to check if there is a path from start to goal
    while frontier:
        r, c, steps = frontier.pop()
        if steps > max_steps:
            continue

        if (r, c) not in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new, steps + 1))
    return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: int = 0, max_steps: int = 5
) -> Tuple[list[str], Tuple[int, int]]:
    """Generates a random valid map (one that has a path from start to goal).

    Args:
        size: Size of each side of the grid.
        p: Probability that a tile is frozen.
        seed: Seed to ensure the generation of reproducible maps.
        max_steps: Maximum number of steps allowed.

    Returns:
        A tuple containing a random valid map and the goal position (row, col).
    """
    valid = False
    board: list[list[str]] = []  # initialize to make pyright happy

    try:
        from gymnasium.utils import seeding

        np_random, _ = seeding.np_random(seed)
    except ImportError:
        raise ImportError(
            "Gymnasium is not installed. Please install gymnasium first before running the frozen_lake workflow."
        )

    # generate random start and end points
    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()

        while True:
            start_r = int(np_random.integers(0, size))
            start_c = int(np_random.integers(0, size))
            goal_r = int(np_random.integers(0, size))
            goal_c = int(np_random.integers(0, size))

            # Ensure start and goal are different positions
            if (start_r, start_c) != (goal_r, goal_c):
                break

        board[start_r][start_c] = "S"
        board[goal_r][goal_c] = "G"

        valid = is_valid(board, size, max_steps)
    return ["".join(x) for x in board], (goal_r, goal_c)


def get_goal_position(random_map: np.ndarray) -> Optional[Tuple[int, int]]:
    """Get the goal position from a random map.

    Args:
        random_map: The map as a numpy array.

    Returns:
        Tuple of (row, col) if goal found, None otherwise.
    """
    positions = np.argwhere(random_map == b"G")
    if positions.size == 0:
        return None  # G not found
    return tuple(positions[0])  # returns (row, col)
