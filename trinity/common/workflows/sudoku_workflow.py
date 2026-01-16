from trinity.common.experience import Experience
from trinity.common.workflows.workflow import Workflow

from .sudoku_generator import SudokuGenerator
from .sudoku_judge import SudokuJudge


class SudokuWorkflow(Workflow):
    """
    Multi-step Sudoku solving workflow.
    - Shows current puzzle board to model
    - Model returns a move: "r c v"
    - Workflow applies move
    - Judge checks validity
    - Continues for max_steps
    """

    can_reset = True

    def __init__(self, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

        # If no dataset provided, generate puzzle
        if "puzzle" in task.raw_task:
            self.board = [row[:] for row in task.raw_task["puzzle"]]
            self.solution = [row[:] for row in task.raw_task["solution"]]
        else:
            generator = SudokuGenerator()
            self.board, self.solution = generator.generate()

        self.judge = SudokuJudge()
        self.max_steps = 20

    def reset(self, task):
        """Reset puzzle for new task."""
        self.board = [row[:] for row in task.raw_task["puzzle"]]
        self.solution = [row[:] for row in task.raw_task["solution"]]

    def parse_action(self, text):
        """Expected model output: 'row col value'"""
        try:
            parts = text.strip().split()
            if len(parts) != 3:
                return None
            r, c, v = map(int, parts)
            if not (0 <= r <= 8 and 0 <= c <= 8 and 1 <= v <= 9):
                return None
            return r, c, v
        except Exception:
            return None

    def apply_move(self, r, c, v):
        if self.board[r][c] == 0:
            self.board[r][c] = v

    def run(self):
        experiences = []

        for step in range(self.max_steps):
            prompt = f"""
Solve Sudoku by giving moves one at a time.
Current board (0 = empty):

{self.board}

Respond ONLY with: row col value
"""

            # Call model
            responses = self.model.chat([{"role": "user", "content": prompt}])
            resp = responses[0]

            action = self.parse_action(resp.response_text)
            if action is None:
                reward = -1.0
                break

            r, c, v = action
            self.apply_move(r, c, v)

            # Check validity
            if not self.judge.is_valid(self.board):
                reward = -1.0
                break

            # Check solved
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

            # Neutral step reward
            reward = 0.0

            # Add experience
            experiences.append(
                Experience(
                    tokens=resp.tokens,
                    prompt_length=resp.prompt_length,
                    reward=reward,
                    logprobs=resp.logprobs,
                )
            )

        return experiences
