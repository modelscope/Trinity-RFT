import re

from trinity.common.experience import Experience
from trinity.common.workflows.workflow import Workflow

from .sudoku_generator import SudokuGenerator
from .sudoku_judge import SudokuJudge


class SudokuWorkflow(Workflow):
    """
    Multi-step Sudoku solving workflow.

    This workflow follows a FrozenLake-style agentic interaction pattern:
    - Maintains an internal environment state (Sudoku board)
    - Interacts with the model step by step
    - Provides explicit rules, task description, and strict output format
    - Gives feedback on invalid or ineffective actions
    - Terminates on success or failure
    """

    can_reset = True

    def __init__(self, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

        # Initialize puzzle
        if "puzzle" in task.raw_task and "solution" in task.raw_task:
            self.board = [row[:] for row in task.raw_task["puzzle"]]
            self.solution = [row[:] for row in task.raw_task["solution"]]
        else:
            generator = SudokuGenerator()
            self.board, self.solution = generator.generate()

        self.judge = SudokuJudge()
        self.max_steps = 20

        # State tracking (FrozenLake-style)
        self.current_step = 0
        self.last_board = None
        self.last_action = None

    def reset(self, task):
        """Reset the workflow state for a new task."""
        self.board = [row[:] for row in task.raw_task["puzzle"]]
        self.solution = [row[:] for row in task.raw_task["solution"]]
        self.current_step = 0
        self.last_board = None
        self.last_action = None

    def _build_prompt(self):
        """
        Build a detailed, step-aware prompt inspired by the Frozen Lake example.
        """
        prompt = (
            "You are playing a Sudoku game.\n\n"
            "Game Rules:\n"
            "- The board is a 9x9 grid.\n"
            "- A value of 0 represents an empty cell.\n"
            "- Each row must contain the numbers 1 through 9 exactly once.\n"
            "- Each column must contain the numbers 1 through 9 exactly once.\n"
            "- Each 3x3 sub-grid must contain the numbers 1 through 9 exactly once.\n"
            "- You may only place numbers in empty cells.\n\n"
            "Task:\n"
            "- At each step, output ONE valid move to progress toward solving the puzzle.\n\n"
            "Output Format (STRICT):\n"
            "```row col value```\n\n"
            "Example:\n"
            "```0 2 4```\n\n"
            f"Current Step: {self.current_step}\n"
            f"Remaining Steps: {self.max_steps - self.current_step}\n\n"
            f"Current Board:\n{self.board}\n"
        )

        if self.last_board is not None and self.board == self.last_board:
            prompt += (
                "\nYour last response was invalid or had no effect. "
                "Please recheck the Sudoku rules and the required output format."
            )

        return prompt

    def parse_action(self, text):
        """
        Parse model output.

        Expected format:
        ```row col value```
        """
        matches = re.findall(r"```(.*?)```", text, re.DOTALL)
        if not matches:
            return None

        try:
            parts = matches[-1].strip().split()
            if len(parts) != 3:
                return None

            r, c, v = map(int, parts)
            if not (0 <= r <= 8 and 0 <= c <= 8 and 1 <= v <= 9):
                return None

            return r, c, v
        except ValueError:
            return None

    def apply_move(self, r, c, v):
        """Apply a move to the board if the cell is empty."""
        if self.board[r][c] == 0:
            self.board[r][c] = v

    def run(self):
        """
        Execute the Sudoku workflow step by step.
        """
        experiences = []

        for _ in range(self.max_steps):
            prompt = self._build_prompt()

            responses = self.model.chat([{"role": "user", "content": prompt}])
            resp = responses[0]

            self.last_board = [row[:] for row in self.board]

            action = self.parse_action(resp.response_text)
            if action is None:
                reward = -1.0
                experiences.append(
                    Experience(
                        tokens=resp.tokens,
                        prompt_length=resp.prompt_length,
                        reward=reward,
                        logprobs=resp.logprobs,
                    )
                )
                break

            r, c, v = action
            self.apply_move(r, c, v)

            # Invalid or ineffective action
            if self.board == self.last_board or not self.judge.is_valid(self.board):
                reward = -1.0
                experiences.append(
                    Experience(
                        tokens=resp.tokens,
                        prompt_length=resp.prompt_length,
                        reward=reward,
                        logprobs=resp.logprobs,
                    )
                )
                break

            # Solved
            if self.judge.is_solved(self.board, self.solution):
                reward = 1.0
                experiences.append(
                    Experience(
                        tokens=resp.tokens,
                        prompt_length=resp.prompt_length,
                        reward=reward,
                        logprobs=resp.logprobs,
                    )
                )
                break

            # Intermediate step
            reward = 0.0
            experiences.append(
                Experience(
                    tokens=resp.tokens,
                    prompt_length=resp.prompt_length,
                    reward=reward,
                    logprobs=resp.logprobs,
                )
            )

            self.last_action = action
            self.current_step += 1

        return experiences
