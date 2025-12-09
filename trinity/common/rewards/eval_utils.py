# -*- coding: utf-8 -*-
import concurrent.futures

import regex as re
from math_verify import parse, verify

from trinity.common.rewards.qwen25_eval import strip_string

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
parse.__globals__["TIMEOUT_WARNING_SHOWN"] = True
verify.__globals__["TIMEOUT_WARNING_SHOWN"] = True


def parse_with_timeout(pred: str, parsing_timeout: int = 5, **kwargs) -> list[str]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(parse, pred=pred, parsing_timeout=None, **kwargs)
        try:
            return future.result(timeout=parsing_timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Parsing timed out")


def verify_with_timeout(gold: str, target: str, timeout_seconds: int = 5, **kwargs) -> bool:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(verify, gold=gold, target=target, timeout_seconds=None, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Verification timed out")


def simple_answer_parser(response: str) -> list[str]:
    search_ans = re.search(r"<answer>(.*?)</answer>", response)
    if search_ans:
        response = search_ans.group(1)

    return parse_with_timeout(response)


def find_boxed_answer(raw_answer, timeout=10):
    """
    Find answers from solutions where the answers are enclosed in LaTeX's `\\boxed` tag

    Args:
        raw_answer (`str`): raw answer from model
        timeout (`int`): timeout in seconds for regex

    Returns:
        `str`: answer if found, otherwise None
    """
    pattern = r"\\boxed\s*(({(?:\\.|[^{}]|(?2))*})|(.))"
    res = re.findall(pattern, raw_answer, timeout=timeout)
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


# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
def compute_score_v0(solution_str, ground_truth) -> float:
    """
    Compute the score of the solution string against the ground truth.
    This function suits easily-verifiable problems; the answer is put within `\boxed{}`.
    """
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        original_ground_truth = ground_truth
        boxed_ground_truth = last_boxed_only_string(ground_truth)
        # Determine if ground_truth was raw (had boxed content) or already processed
        ground_truth_was_raw = boxed_ground_truth is not None
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if ground_truth_was_raw:
                # Ground truth had boxed content - remove it
                ground_truth = remove_boxed(boxed_ground_truth)
            else:
                # Ground truth had no boxed content - use as is
                ground_truth = original_ground_truth
            if is_equiv(answer, ground_truth):
                retval = 1.0

    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
def last_boxed_only_string(string):
    """Extracts the last `\boxed{...}` or `\fbox{...}` substring from the input string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string
