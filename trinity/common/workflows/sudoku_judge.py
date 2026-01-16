class SudokuJudge:
    """
    Judge Sudoku board state.
    - Checks row validity
    - Checks column validity
    - Checks 3x3 block validity
    """

    @staticmethod
    def is_valid(board):
        # Check rows
        for row in board:
            nums = [v for v in row if v != 0]
            if len(nums) != len(set(nums)):
                return False

        # Check columns
        for col in range(9):
            nums = []
            for row in range(9):
                v = board[row][col]
                if v != 0:
                    nums.append(v)
            if len(nums) != len(set(nums)):
                return False

        # Check 3x3 sub-grids
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                nums = []
                for r in range(br, br + 3):
                    for c in range(bc, bc + 3):
                        v = board[r][c]
                        if v != 0:
                            nums.append(v)
                if len(nums) != len(set(nums)):
                    return False

        return True

    @staticmethod
    def is_solved(board, solution):
        return board == solution
