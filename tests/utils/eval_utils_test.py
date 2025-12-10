# -*- coding: utf-8 -*-
"""Test for the evaluation utils module."""

import unittest

from trinity.common.rewards.eval_utils import compute_score_v0, is_equiv
from trinity.common.rewards.qwen25_eval import extract_answer, verify_math_answer


class TestComputeScore(unittest.TestCase):
    """
    A suite of unit tests for the compute_score function.
    """

    def test_both_boxed_and_equivalent(self):
        """
        Tests the case where both solution and ground truth have equivalent boxed answers.
        Expected score: 1.0
        """
        solution = "The final answer is \\boxed{42}"
        truth = "The correct result is \\boxed{42}"
        self.assertEqual(compute_score_v0(solution, truth), 1.0)

    def test_solution_raw_and_ground_truth_boxed_equivalent(self):
        """
        Tests the case where the solution is a raw string and the ground truth is boxed, but they are equivalent.
        Expected score: 1.0
        """
        solution = "The answer is \\boxed{42}"
        truth = "The answer is \\boxed{42}"
        self.assertEqual(compute_score_v0(solution, truth), 1.0)

    def test_solution_boxed_truth_raw_and_equivalent(self):
        """
        Tests the case where the solution is boxed and the ground truth is a raw, equivalent string.
        Expected score: 1.0
        """
        solution = "Let's see, the result is \\boxed{100}"
        truth = "100"
        self.assertEqual(compute_score_v0(solution, truth), 1.0)

    def test_both_boxed_and_not_equivalent(self):
        """
        Tests the case where both have boxed answers, but they are not equivalent.
        Expected score: 0.0
        """
        solution = "I think the answer is \\boxed{-1}"
        truth = "The answer is \\boxed{1}"
        self.assertEqual(compute_score_v0(solution, truth), 0.0)

    def test_solution_boxed_truth_raw_and_not_equivalent(self):
        """
        Tests the case where the solution is boxed and the ground truth is a raw, non-equivalent string.
        Expected score: 0.0
        """
        solution = "The answer is \\boxed{apple}"
        truth = "orange"
        self.assertEqual(compute_score_v0(solution, truth), 0.0)

    def test_solution_not_boxed(self):
        """
        Tests the case where the solution string does not contain a boxed answer.
        Expected score: 0.0, regardless of the ground truth.
        """
        solution = "The answer is 42, but I'm not boxing it."
        truth_boxed = "The answer is \\boxed{42}"
        truth_raw = "42"
        self.assertEqual(compute_score_v0(solution, truth_boxed), 0.0)
        self.assertEqual(compute_score_v0(solution, truth_raw), 0.0)

    def test_empty_solution_string(self):
        """
        Tests behavior with an empty solution string.
        Expected score: 0.0
        """
        solution = ""
        truth = "\\boxed{10}"
        self.assertEqual(compute_score_v0(solution, truth), 0.0)

    def test_empty_ground_truth(self):
        """
        Tests behavior with an empty ground truth string.
        Expected score: 0.0 unless the boxed answer is also empty.
        """
        solution_correct = "The answer is \\boxed{}"
        solution_incorrect = "The answer is \\boxed{1}"
        truth = ""
        self.assertEqual(compute_score_v0(solution_correct, truth), 1.0)
        self.assertEqual(compute_score_v0(solution_incorrect, truth), 0.0)

    def test_multiple_boxed_answers_in_solution(self):
        """
        Tests that only the *last* boxed answer in the solution is used for scoring.
        """
        solution = "First I thought it was \\boxed{A}, but then I realized it is \\boxed{B}"
        truth_correct = "\\boxed{B}"
        truth_incorrect = "\\boxed{A}"
        self.assertEqual(compute_score_v0(solution, truth_correct), 1.0)
        self.assertEqual(compute_score_v0(solution, truth_incorrect), 0.0)


class TestMathEvalUtils(unittest.TestCase):
    def test_extract_answer(self):
        test_cases = [
            ("The answer is \\boxed{42}", "42", "Basic boxed extraction"),
            ("The result is \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", "Boxed with LaTeX"),
            ("Therefore, the final answer is 100.", "100", "English 'answer is' extraction"),
            ("My final answer is: 3.14", "3.14", "English 'answer is' with colon"),
            ("所以，答案是x^2", "x^2", "Chinese 'answer is' extraction"),
            (
                "The cost is 10 dollars and the profit is 20 dollars.",
                "20",
                "Extract the last number",
            ),
            (
                "There are 1,000 apples and 2,000 oranges.",
                "2000",
                "Extract the last number with commas",
            ),
            ("The probability is 0.75.", "0.75", "Extract the last decimal"),
            ("This sentence has no answer.", None, "No answer case"),
            ("The box is empty \\boxed{}", None, "Empty boxed"),
            (12345, None, "Input is not a string"),
        ]

        for i, (input_str, expected_output, description) in enumerate(test_cases):
            with self.subTest(f"Case {i + 1}: {description}"):
                actual_output = extract_answer(input_str)
                self.assertEqual(
                    actual_output,
                    expected_output,
                    f"Failed on input: '{input_str}'\nExpected: '{expected_output}', Got: '{actual_output}'",
                )

    def test_verify_math_answer(self):
        test_cases = [
            ("The answer is \\boxed{42}", "42", True, "Simple integer equality"),
            ("The result is 1,000.", "1000", True, "Number with commas"),
            ("The answer is -50.", "-50", True, "Negative number equality"),
            ("The solution is 5", "x=5", True, "Equivalence of value and equation"),
            ("The answer is \\boxed{42}", "43", False, "Simple numerical inequality"),
            ("The answer is \\boxed{x+1}", "x-1", False, "Symbolic expression inequality"),
            (
                "The matrix is \\boxed{\\begin{pmatrix}1 & 1 \\\\ 0 & 1\\end{pmatrix}}",
                "\\begin{pmatrix}1&0\\\\0&1\\end{pmatrix}",
                False,
                "Matrix inequality",
            ),
            ("The speed is 50 km/h", "50", True, "Judgment after stripping units"),
        ]

        for i, (response, ground_truth, expected_correct, description) in enumerate(test_cases):
            with self.subTest(f"Case {i + 1}: {description}"):
                accuracy, details = verify_math_answer(response, ground_truth)
                is_correct = accuracy == 1.0
                self.assertEqual(
                    is_correct,
                    expected_correct,
                    f"Failed on response: '{response}' with truth: '{ground_truth}'\n"
                    f"Expected correct: {expected_correct}, Got: {is_correct}\nDetails: {details}",
                )


if __name__ == "__main__":
    unittest.main()


class TestEvalUtils(unittest.TestCase):
    def test_is_equiv(self):
        test_cases = [
            # str1, str2, expected_output, description
            ("  123  ", "123", True, "Equivalence with whitespace"),
            ("50%", "50", True, "Equivalence with percentage sign"),
            ("$50", "50", True, "Equivalence with dollar sign"),
            ("hello", "world", False, "Basic inequality"),
            ("123", "1234", False, "Numerical inequality"),
            (None, None, True, "Both inputs are None"),
            ("Some string", None, False, "One input is None (str1)"),
            (None, "Some string", False, "One input is None (str2)"),
        ]

        for i, (str1, str2, expected_output, description) in enumerate(test_cases):
            with self.subTest(f"Case {i + 1}: {description}"):
                actual_output = is_equiv(str1, str2)
                self.assertEqual(
                    actual_output,
                    expected_output,
                    f"Failed on inputs: ('{str1}', '{str2}')\nExpected: {expected_output}, Got: {actual_output}",
                )
