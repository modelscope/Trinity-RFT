import copy
from typing import Dict, Optional, Tuple

import numpy as np

from examples.agentscope_frozenlake.utils import generate_random_map, get_goal_position
from trinity.utils.log import get_logger

try:
    from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
except ImportError:
    GymFrozenLakeEnv = object


class FrozenLakeEnv(GymFrozenLakeEnv):
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
        "still": 0,
        "left": 1,
        "down": 2,
        "right": 3,
        "up": 4,
    }

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1

    def __init__(
        self,
        max_steps: int = 8,
        desc: Optional[str] = None,
        is_slippery: bool = False,
        size: int = 8,
        p: float = 0.8,
        seed: int = 42,
    ):
        self.logger = get_logger()
        self.max_steps = max_steps or 8
        self.desc = desc
        self.is_slippery = is_slippery
        self.size = size
        self.p = p
        self.seed = seed
        try:
            import gymnasium as gym
            from gymnasium.envs.toy_text.frozen_lake import (
                FrozenLakeEnv as GymFrozenLakeEnv,
            )
        except ImportError as e:
            error_message = (
                f"Gymnasium is not installed. Please install gymnasium first before "
                f"running the frozen_lake workflow. Error: {str(e)}"
            )
            self.logger.error(error_message)
            raise ImportError(error_message)

        if self.desc is None:
            random_map, goal_position = generate_random_map(
                size=self.size, p=self.p, seed=self.seed, max_steps=self.max_steps
            )
        else:
            random_map = np.asarray(copy.deepcopy(self.desc), dtype="c")
            goal_position = get_goal_position(random_map)

        self.goal_position = goal_position

        GymFrozenLakeEnv.__init__(self, desc=random_map[:], is_slippery=self.is_slippery)
        self.action_space = gym.spaces.Discrete(4, start=1)

        self.map_kwargs = {
            "size": size,
            "p": p,
        }
        self.env_kwargs = {
            "is_slippery": is_slippery,
            "desc": copy.deepcopy(desc),
            "seed": seed,
        }

        self.action_map = {
            1: 0,  # left
            2: 1,  # down
            3: 2,  # right
            4: 3,  # up
        }

    def _get_player_position(self) -> Tuple[int, int]:
        return (self.s // self.ncol, self.s % self.ncol)  # (row, col)

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute a step in the environment.

        Maps custom action to gymnasium FrozenLakeEnv action and takes the step.
        Checks if the action is effective (whether player moves in the env).

        Args:
            action: The action to take.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self.success():
            return self.render(), 1, True, {"action_is_effective": False}

        action_id: int = self.ACTION_LOOKUP.get(action.lower(), 0)

        if not action_id:
            action_id = self.INVALID_ACTION

        if action_id == self.INVALID_ACTION or action_id not in self.action_map:
            return self.render(), 0, False, {"action_is_effective": False}

        prev_player_position = int(self.s)

        player_pos, reward, done, _, _ = GymFrozenLakeEnv.step(self, self.action_map[action_id])

        obs = self.render()
        return obs, reward, done, {"action_is_effective": prev_player_position != int(player_pos)}

    def render(self, mode="tiny_rgb_array"):
        """Render the environment.

        Args:
            mode: Rendering mode. Options: "tiny_rgb_array", "list", "state", "rgb_array", "ansi".

        Returns:
            Rendered observation based on the mode.
        """
        assert mode in ["tiny_rgb_array", "list", "state", "rgb_array", "ansi"]
        if mode in ["rgb_array", "ansi"]:
            prev_render_mode = self.render_mode
            self.render_mode = mode
            obs = GymFrozenLakeEnv.render(self)
            self.render_mode = prev_render_mode
            return obs
        room_state = copy.deepcopy(self.desc)

        # replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b"S")
        room_state[position_S] = b"F"

        # replace the position of the player with 'P'
        position_P = self._get_player_position()
        room_state[position_P] = b"P"

        if mode == "state":
            # transform 'S', 'F', 'H', 'G' to numpy integer array
            room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
            # add player in hole or player on goal
            if self.desc[position_P] == b"H":
                room_state[position_P] = 4
            elif self.desc[position_P] == b"G":
                room_state[position_P] = 5
            return room_state

        room_state = self.render(mode="state").tolist()

        if mode == "list":

            def lookup(cell):
                return self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()

            return [" ".join(lookup(cell) for cell in row) for row in room_state]

        if mode == "tiny_rgb_array":

            def lookup(cell):
                return self.GRID_LOOKUP.get(cell, "?")

            result = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
            return result

    def reset(self, task: Optional[Dict] = None):
        task = task or {}
        self.__init__(  # type: ignore [misc]
            size=task.get("size", self.map_kwargs["size"]),
            p=task.get("p", self.map_kwargs["p"]),
            seed=task.get("seed", self.env_kwargs["seed"]),
            is_slippery=task.get("is_slippery", self.env_kwargs["is_slippery"]),
        )
        GymFrozenLakeEnv.reset(self, seed=self.seed)
        return self.render(mode="tiny_rgb_array"), {}

    def finished(self) -> bool:
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"GH"  # type: ignore [index,operator]

    def success(self):
        """
        Check if the agent has reached the goal (G).
        """
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"G"
