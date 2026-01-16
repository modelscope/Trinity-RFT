import random


class SudokuGenerator:
    """
    Very simple Sudoku generator.
    - Uses a fixed solved grid
    - Removes 'holes' positions to create a puzzle
    """

    BASE_SOLUTION = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]

    def generate(self, holes=40):
        """Return (puzzle, solution) tuple."""
        solution = [row[:] for row in self.BASE_SOLUTION]
        puzzle = [row[:] for row in solution]

        for _ in range(holes):
            r = random.randint(0, 8)
            c = random.randint(0, 8)
            puzzle[r][c] = 0

        return puzzle, solution
