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


def compute_response_metrics(responses):
    if not responses:
        return responses

    acc_key = next((k for k, v in responses[0].metrics.items() if "accuracy" in k), None)
    if acc_key is None:
        raise ValueError("No accuracy metric found in responses.")

    total_response_length = 0
    total_correct_length = 0
    total_incorrect_length = 0
    pass_k_count = 0
    num_responses = len(responses)

    for response in responses:
        tokens_length = len(response.tokens) - response.prompt_length
        is_correct = response.metrics.get(acc_key) == 1.0

        total_response_length += tokens_length
        if is_correct:
            pass_k_count += 1
            total_correct_length += tokens_length
        else:
            total_incorrect_length += tokens_length

    avg_response_length = total_response_length / num_responses
    avg_pass_k = pass_k_count / num_responses
    avg_correct_length = total_correct_length / pass_k_count if pass_k_count > 0 else None
    avg_incorrect_length = (
        total_incorrect_length / (num_responses - pass_k_count)
        if num_responses > pass_k_count
        else None
    )

    metrics = {
        "response_length": avg_response_length,
        "pass@k": avg_pass_k,
        **(
            {"response_length_correct": avg_correct_length}
            if avg_correct_length is not None
            else {}
        ),
        **(
            {"response_length_wrong": avg_incorrect_length}
            if avg_incorrect_length is not None
            else {}
        ),
    }

    for response in responses:
        response.metrics.update(metrics)

    return responses
