import random


class SudokuGenerator:
    """
    Sudoku puzzle generator inspired by standard backtracking-based generators.

    - Generates a fresh solved Sudoku board using backtracking
    - Removes cells based on difficulty (number of empty cells)
    - Avoids relying on a single canonical solution
    """

    def generate(self, difficulty="medium"):
        holes_map = {
            "easy": 30,
            "medium": 40,
            "hard": 50,
        }
        holes = holes_map.get(difficulty, 40)

        board = [[0 for _ in range(9)] for _ in range(9)]
        self._fill_board(board)

        solution = [row[:] for row in board]
        self._remove_cells(board, holes)

        return board, solution

    def _fill_board(self, board):
        empty = self._find_empty(board)
        if not empty:
            return True

        r, c = empty
        nums = list(range(1, 10))
        random.shuffle(nums)

        for v in nums:
            if self._is_valid(board, r, c, v):
                board[r][c] = v
                if self._fill_board(board):
                    return True
                board[r][c] = 0

        return False

    def _find_empty(self, board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return None

    def _is_valid(self, board, r, c, v):
        if v in board[r]:
            return False

        if v in [board[i][c] for i in range(9)]:
            return False

        br, bc = (r // 3) * 3, (c // 3) * 3
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if board[i][j] == v:
                    return False

        return True

    def _remove_cells(self, board, holes):
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)

        for i in range(min(holes, 81)):
            r, c = cells[i]
            board[r][c] = 0
