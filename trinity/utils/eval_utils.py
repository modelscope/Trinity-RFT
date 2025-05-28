# -*- coding: utf-8 -*-
import regex as re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def simple_answer_parser(response: str) -> str:
    from math_verify import parse

    search_ans = re.search(r"<answer>(.*?)</answer>", response)
    if search_ans:
        response = search_ans.group(1)

    return parse(response)


def find_boxed_answer(string):
    """
    Find answers from solutions where the answers are enclosed in LaTeX's `\boxed` tag
    """
    pattern = r"\\boxed\s*(({(?:\\.|[^{}]|(?2))*})|(.))"
    res = re.findall(pattern, string)
    if res:
        answer = res[-1][0]  # regard the last boxed as the answer
        if answer.startswith("{"):
            answer = answer[1:-1]
        return answer
    else:
        return None


# copy from Jiayi-Pan/TinyZero verl/utils/reward_score/countdown.py
def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


# copy from Jiayi-Pan/TinyZero verl/utils/reward_score/countdown.py
def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception as e:  # noqa: F841
        return False


# copy from Jiayi-Pan/TinyZero verl/utils/reward_score/countdown.py
def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:  # noqa: F841
        return None


def validate_think_pattern(text):
    """Validate whether the <think> </think> tag is properly formatted."""
    start_tag = "<think>"
    end_tag = "</think>"

    start_count = text.count(start_tag)
    end_count = text.count(end_tag)

    if start_count == 1 and end_count == 1:
        start_pos = text.find(start_tag)
        end_pos = text.find(end_tag)
        if start_pos < end_pos:
            return True
    return False
