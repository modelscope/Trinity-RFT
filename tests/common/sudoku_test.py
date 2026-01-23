from trinity.common.workflows.envs.sudoku.sudoku_generator import SudokuGenerator
from trinity.common.workflows.envs.sudoku.sudoku_judge import SudokuJudge

# ---------- Generator Tests (9x9) ----------


def test_9x9_generator_produces_valid_solution():
    gen = SudokuGenerator()
    puzzle, solution = gen.generate()

    assert len(puzzle) == 9
    assert len(solution) == 9
    assert SudokuJudge.is_valid(solution)


def test_9x9_generator_creates_holes():
    gen = SudokuGenerator()
    puzzle, _ = gen.generate()

    zero_count = sum(row.count(0) for row in puzzle)
    assert zero_count > 0


def test_9x9_solution_is_fully_filled():
    gen = SudokuGenerator()
    _, solution = gen.generate()

    for row in solution:
        assert 0 not in row


# ---------- Judge Tests (9x9) ----------


def test_judge_allows_incomplete_board():
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    assert SudokuJudge.is_valid(board)


def test_judge_detects_row_violation():
    board = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
    ] + [[0] * 9 for _ in range(8)]

    assert not SudokuJudge.is_valid(board)


def test_judge_detects_column_violation():
    board = [
        [5, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 0],
    ] + [[0] * 9 for _ in range(7)]

    assert not SudokuJudge.is_valid(board)


def test_judge_detects_block_violation():
    board = [
        [1, 2, 3, 0, 0, 0, 0, 0, 0],
        [4, 1, 0, 0, 0, 0, 0, 0, 0],
    ] + [[0] * 9 for _ in range(7)]

    assert not SudokuJudge.is_valid(board)


# ---------- Generator & Judge Tests (4x4) ----------


def test_4x4_generator_produces_valid_solution():
    gen = SudokuGenerator(size=4)
    puzzle, solution = gen.generate()

    assert len(puzzle) == 4
    assert len(solution) == 4
    assert SudokuJudge.is_valid(solution)


def test_4x4_solution_is_fully_filled():
    gen = SudokuGenerator(size=4)
    _, solution = gen.generate()

    for row in solution:
        assert 0 not in row


def test_4x4_judge_detects_row_violation():
    board = [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    assert not SudokuJudge.is_valid(board)


def test_4x4_judge_detects_block_violation():
    board = [
        [1, 2, 0, 0],
        [3, 1, 0, 0],  # duplicate "1" in top-left 2x2 block
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    assert not SudokuJudge.is_valid(board)
