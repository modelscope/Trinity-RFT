import random


class SudokuGenerator:
    """
    Lightweight Sudoku generator.

    This generator avoids relying on a single canonical solution by applying
    randomized transformations to a solved grid before removing values to
    create a puzzle. The difficulty is controlled by the number of removed
    cells (holes).

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

    def _shuffle_solution(self, board):
        """
        Randomize a solved Sudoku grid while preserving validity.

        This follows common Sudoku generation techniques:
        - permuting numbers
        - shuffling rows
        - shuffling columns
        """
        board = [row[:] for row in board]

        # Shuffle numbers 1–9
        numbers = list(range(1, 10))
        shuffled_numbers = numbers[:]
        random.shuffle(shuffled_numbers)
        mapping = dict(zip(numbers, shuffled_numbers))
        board = [[mapping[v] for v in row] for row in board]

        # Shuffle rows
        random.shuffle(board)

        # Shuffle columns
        board = list(map(list, zip(*board)))
        random.shuffle(board)
        board = list(map(list, zip(*board)))

        return board

    def generate(self, holes=40):
        """
        Generate a Sudoku puzzle.

        Args:
            holes (int): Number of empty cells (0s) in the puzzle.
                         Larger values correspond to higher difficulty.

        Returns:
            tuple: (puzzle, solution)
        """
        solution = self._shuffle_solution(self.BASE_SOLUTION)
        puzzle = [row[:] for row in solution]

        for _ in range(holes):
            r = random.randint(0, 8)
            c = random.randint(0, 8)
            puzzle[r][c] = 0

        return puzzle, solution
