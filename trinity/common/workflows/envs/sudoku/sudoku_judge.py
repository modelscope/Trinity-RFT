import math


class SudokuJudge:
    """
    Judge Sudoku board state.

    - Supports both 9x9 and 4x4 Sudoku boards
    - Allows incomplete boards (zeros are treated as empty cells)
    - Checks:
        * Row validity
        * Column validity
        * Sub-grid validity (3x3 for 9x9, 2x2 for 4x4)
    """

    @staticmethod
    def is_valid(board):
        size = len(board)
        block = int(math.sqrt(size))

        # Check rows
        for row in board:
            nums = [v for v in row if v != 0]
            if len(nums) != len(set(nums)):
                return False

        # Check columns
        for c in range(size):
            nums = []
            for r in range(size):
                v = board[r][c]
                if v != 0:
                    nums.append(v)
            if len(nums) != len(set(nums)):
                return False

        # Check sub-grids
        for br in range(0, size, block):
            for bc in range(0, size, block):
                nums = []
                for r in range(br, br + block):
                    for c in range(bc, bc + block):
                        v = board[r][c]
                        if v != 0:
                            nums.append(v)
                if len(nums) != len(set(nums)):
                    return False

        return True

    @staticmethod
    def is_solved(board, solution):
        return board == solution
