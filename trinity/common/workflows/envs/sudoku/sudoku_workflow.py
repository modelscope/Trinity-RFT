from trinity.common.experience import Experience
from trinity.common.workflows.workflow import Workflow

from .sudoku_generator import SudokuGenerator
from .sudoku_judge import SudokuJudge


class SudokuWorkflow(Workflow):
    """
    Agentic multi-step Sudoku solving workflow.

    The workflow:
    - Presents the current Sudoku board to the model
    - Allows the model to propose multiple moves per step
    - Applies moves incrementally and validates them
    - Terminates on success, invalid action, or step limit
    """

    can_reset = True

    def __init__(self, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

        # Load puzzle from task if provided, otherwise generate a new one
        if "puzzle" in task.raw_task and "solution" in task.raw_task:
            self.board = [row[:] for row in task.raw_task["puzzle"]]
            self.solution = [row[:] for row in task.raw_task["solution"]]
        else:
            generator = SudokuGenerator()
            self.board, self.solution = generator.generate()

        self.judge = SudokuJudge()

        # Workflow configuration
        self.max_steps = 20
        self.max_moves_per_step = 5

        # Runtime state
        self.current_step = 0
        self.last_board = None
        self.last_action = None

    def reset(self, task):
        """
        Reset workflow state for a new task instance.
        """
        self.board = [row[:] for row in task.raw_task["puzzle"]]
        self.solution = [row[:] for row in task.raw_task["solution"]]
        self.current_step = 0
        self.last_board = None
        self.last_action = None

    def render_board(self):
        """
        Render the board into a human-readable string format
        for inclusion in the prompt.
        """
        return "\n".join(" ".join(str(v) for v in row) for row in self.board)

    def _build_prompt(self):
        """
        Build a step-aware prompt describing:
        - Sudoku rules
        - Current board state
        - Allowed action format
        """
        prompt = (
            "You are playing a Sudoku game.\n\n"
            "Rules:\n"
            "- The board is 9x9.\n"
            "- 0 means empty.\n"
            "- Numbers 1–9 must appear exactly once in every row, column, and 3x3 block.\n"
            "- You may only fill empty cells.\n\n"
            "Task:\n"
            "- In each step, output ONE OR MORE valid moves.\n"
            f"- You may output up to {self.max_moves_per_step} moves per step.\n\n"
            "Output format (STRICT):\n"
            "row col value\n"
            "row col value\n\n"
            "Example:\n"
            "0 2 4\n"
            "1 3 5\n\n"
            f"Current step: {self.current_step}\n"
            f"Remaining steps: {self.max_steps - self.current_step}\n\n"
            f"Current board:\n{self.render_board()}\n"
        )

        # Feedback when the previous step made no progress
        if self.last_board is not None and self.board == self.last_board:
            prompt += (
                "\nYour previous response was invalid or had no effect. "
                "Please follow the rules and output format strictly."
            )

        return prompt

    def parse_action(self, text):
        """
        Parse model output into a list of (row, col, value) moves.

        Expected format:
            row col value
            row col value
        """
        lines = text.strip().splitlines()
        actions = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                return None

            try:
                r, c, v = map(int, parts)
            except ValueError:
                return None

            if not (0 <= r <= 8 and 0 <= c <= 8 and 1 <= v <= 9):
                return None

            actions.append((r, c, v))

        if not actions or len(actions) > self.max_moves_per_step:
            return None

        return actions

    def run(self):
        """
        Execute the Sudoku workflow until:
        - The puzzle is solved
        - An invalid action is produced
        - The maximum number of steps is reached
        """
        experiences = []

        for _ in range(self.max_steps):
            prompt = self._build_prompt()
            responses = self.model.chat([{"role": "user", "content": prompt}])
            resp = responses[0]

            # Snapshot board to detect no-op steps
            self.last_board = [row[:] for row in self.board]

            actions = self.parse_action(resp.response_text)
            if actions is None:
                experiences.append(
                    Experience(
                        tokens=resp.tokens,
                        prompt_length=resp.prompt_length,
                        reward=-1.0,
                        logprobs=resp.logprobs,
                    )
                )
                break

            board_changed = False
            invalid_move = False

            for r, c, v in actions:
                if self.board[r][c] != 0:
                    invalid_move = True
                    break
                self.board[r][c] = v
                board_changed = True

            # Invalid or ineffective step
            if invalid_move or not board_changed or not self.judge.is_valid(self.board):
                experiences.append(
                    Experience(
                        tokens=resp.tokens,
                        prompt_length=resp.prompt_length,
                        reward=-1.0,
                        logprobs=resp.logprobs,
                    )
                )
                break

            # Solved successfully
            if self.judge.is_solved(self.board, self.solution):
                experiences.append(
                    Experience(
                        tokens=resp.tokens,
                        prompt_length=resp.prompt_length,
                        reward=1.0,
                        logprobs=resp.logprobs,
                    )
                )
                break

            # Intermediate step
            experiences.append(
                Experience(
                    tokens=resp.tokens,
                    prompt_length=resp.prompt_length,
                    reward=0.0,
                    logprobs=resp.logprobs,
                )
            )

            self.last_action = actions
            self.current_step += 1

        return experiences
