import math
import random


class SudokuGenerator:
    """
    Sudoku puzzle generator using randomized backtracking.

    Features:
    - Supports arbitrary square sizes (e.g., 9x9, 4x4)
    - Generates a fully solved board first
    - Removes cells based on difficulty to create a puzzle
    - Avoids relying on a single canonical solution
    """

    def __init__(self, size: int = 9):
        """
        Initialize the generator.

        Args:
            size (int): Size of the Sudoku board (must be a perfect square).
                        Examples: 9 for 9x9, 4 for 4x4.
        """
        self.size = size
        self.block = int(math.sqrt(size))
        assert self.block * self.block == size, "Size must be a perfect square"

    def generate(self, difficulty: str = "medium"):
        """
        Generate a Sudoku puzzle and its solution.

        Args:
            difficulty (str): Difficulty level ("easy", "medium", "hard").

        Returns:
            tuple: (puzzle, solution), where puzzle contains zeros for empty cells.
        """
        holes_map = {
            "easy": self.size * self.size // 3,
            "medium": self.size * self.size // 2,
            "hard": self.size * self.size * 2 // 3,
        }
        holes = holes_map.get(difficulty, holes_map["medium"])

        board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._fill_board(board)

        solution = [row[:] for row in board]
        self._remove_cells(board, holes)

        return board, solution

    def _fill_board(self, board):
        """
        Recursively fill the board using backtracking.

        Args:
            board (list[list[int]]): Current board state.

        Returns:
            bool: True if the board is completely filled.
        """
        empty = self._find_empty(board)
        if not empty:
            return True

        r, c = empty
        nums = list(range(1, self.size + 1))
        random.shuffle(nums)

        for v in nums:
            if self._is_valid(board, r, c, v):
                board[r][c] = v
                if self._fill_board(board):
                    return True
                board[r][c] = 0

        return False

    def _find_empty(self, board):
        """
        Find the next empty cell in the board.

        Args:
            board (list[list[int]]): Current board state.

        Returns:
            tuple | None: (row, col) of empty cell, or None if full.
        """
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    return i, j
        return None

    def _is_valid(self, board, r, c, v):
        """
        Check whether placing value v at (r, c) is valid.

        Args:
            board (list[list[int]]): Current board state.
            r (int): Row index.
            c (int): Column index.
            v (int): Value to place.

        Returns:
            bool: True if valid, False otherwise.
        """
        if v in board[r]:
            return False

        for i in range(self.size):
            if board[i][c] == v:
                return False

        br = (r // self.block) * self.block
        bc = (c // self.block) * self.block
        for i in range(br, br + self.block):
            for j in range(bc, bc + self.block):
                if board[i][j] == v:
                    return False

        return True

    def _remove_cells(self, board, holes):
        """
        Remove cells from a solved board to create a puzzle.

        Args:
            board (list[list[int]]): Solved board.
            holes (int): Number of cells to clear.
        """
        cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(cells)

        for i in range(min(holes, self.size * self.size)):
            r, c = cells[i]
            board[r][c] = 0
