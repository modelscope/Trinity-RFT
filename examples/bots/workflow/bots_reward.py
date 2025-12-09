# Adapted from Reasoning360: https://github.com/LLM360/Reasoning360/blob/main/verl/utils/reward_score/naive_dapo.py

import contextlib
import math
import re
from math import isclose
from typing import Optional, Union

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from verl.utils.py_functional import timeout_limit
from verl.utils.reward_score.prime_math.grader import math_equal as verl_math_equal

from trinity.common.rewards.eval_utils import remove_right_units
from trinity.common.rewards.naive_dapo_score import grade_answer, match_answer
from trinity.common.rewards.qwen25_eval import fix_fracs

_fix_fracs = fix_fracs
_remove_right_units = remove_right_units


def handle_base(x):
    if isinstance(x, str) and "_" in x:
        # Due to base
        x = x.split("_")[0]
        x = float(x)
        return int(x)
    return x


def handle_pi(string, pi):
    if isinstance(string, str) and "\\pi" in string:
        # Find the first occurrence of "\pi"
        idx = string.find("\\pi")

        # Iterate over the string and find all occurrences of "\pi" with a valid previous character
        while idx != -1:
            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\pi" with "*math.pi" if the previous character is a digit
                string = string[:idx] + f"*{pi}" + string[idx + 3 :]
            else:
                # Replace "\pi" with "1*math.pi" if the previous character is not a digit
                string = string[:idx] + f"1*{pi}" + string[idx + 3 :]

            # Find the next occurrence of "\pi"
            idx = string.find("\\pi", idx + 1)

        # Evaluate the expression using eval() function
        with contextlib.suppress(Exception):
            string = eval(string)

    return string


def normalize(answer, pi) -> str:
    # checking if answer is $<number> and removing $ in that case to compare
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]

    # checking if answer is <number>% or <number>\\% and removing %
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")

    # handle base
    answer = handle_base(answer)

    # handle pi
    answer = handle_pi(answer, pi)

    return answer


def is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num

        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, 0.0


def format_intervals(prediction) -> str:
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)

            if key == "Interval(":  # Intarval(a, b) == [a, b]
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":  # Intarval.Ropen(a, b) == [a, b)
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":  # Intarval.Lopen(a, b) == (a, b]
                return f"({inner_content}]"
            elif key == "Interval.open(":  # Intarval.open(a, b) == (a, b)
                return f"({inner_content})"

    return str(prediction)


def symbolic_equal(a, b, tolerance, timeout=10.0):
    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with timeout_limit(seconds=timeout):
                    return f(s)
            except TimeoutError:
                print(f"Parsing timed out for {s}")
                continue
            except Exception:
                continue
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with timeout_limit(seconds=timeout):
            if simplify(a - b) == 0:
                return True
    except TimeoutError:
        print(f"Simplification timed out for {a} - {b}")
        pass
    except Exception:
        pass

    try:
        with timeout_limit(seconds=timeout):
            if isclose(N(a), N(b), rel_tol=tolerance):
                return True
    except TimeoutError:
        print(f"Numerical evaluation timed out for {a}, {b}")
        pass
    except Exception:
        pass
    return False


def math_equal(  # noqa
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    pi: float = math.pi,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """

    prediction = normalize(prediction, pi)
    reference = normalize(reference, pi)

    if isinstance(prediction, str) and len(prediction) > 1000:  # handling weird corner-cases
        prediction = prediction[:1000]

    # 0. string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:  # 1. numerical equal
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]
            # number questions
            gt_result = (
                [float(reference) / 100.0, float(reference), float(reference) * 100.0]
                if include_percentage
                else [float(reference)]
            )
            for item in gt_result:
                try:
                    if isclose(float(item), float(prediction), rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ]
        ):
            return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            return bool(
                all(
                    [
                        math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance)
                        for i in range(len(pred_parts))
                    ]
                )
            )

    # if we have point == tuple of values
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ]
        ):
            return True

    # if reference is a matrix
    if "\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items) and all(
                [
                    math_equal(pred, ref, include_percentage, tolerance)
                    for ref, pred in zip(ref_matrix_items, pred_matrix)
                ]
            ):
                return True
        except Exception:
            pass
    elif "\begin{pmatrix}" in reference and prediction.startswith("[") and prediction.endswith("]"):
        if isinstance(eval(prediction), list):
            try:
                pred_matrix = eval(prediction)
                # ref_matrix_items = reference.split()[1:-1:2]
                ref_matrix_items = (
                    reference.lstrip("\\begin{pmatrix}")
                    .lstrip("\begin{pmatrix}")
                    .rstrip("\\end{pmatrix}")
                    .rstrip("\\end{pmatrix}")
                )  # noqa: B005
                ref_matrix_items = ref_matrix_items.split("\\")
                # ref_matrix_items = [
                #     row.split("&") if "&" in row else row for row in ref_matrix_items
                # ]
                if len(pred_matrix) == len(ref_matrix_items) and all(
                    [
                        math_equal(pred, ref, include_percentage, tolerance)
                        for ref, pred in zip(ref_matrix_items, pred_matrix)
                    ]
                ):
                    return True
            except Exception:
                pass

    return symbolic_equal(prediction, reference, tolerance, timeout)


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:  # noqa: E722
        return answer


def compute_score_bots(solution_str: str, ground_truth: Optional[str]) -> float:
    """Compute the reward score for a solution. This draws heavily from the LLM-as-judge and PRIME reward functions

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        extra_info: dict with additional info for the score computation

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # First assert intended generation and gt type
    model_output = str(solution_str)
    ground_truth = str(ground_truth)

    # Extract answer from generated output
    is_matched, extracted_model_output = match_answer(model_output)

    # TWK NOTE: WE REMOVED THE RESPONSE TRUNCATION FROM math_dapo.compute_score

    # Verify the solution, first check simple comparisons.
    correct, pred = grade_answer(extracted_model_output, ground_truth)

    if not correct:
        try:
            # Use verl's math_equal for additional verification
            if "\\pi" in extracted_model_output or "\\pi" in ground_truth:
                # Try with different pi values
                equivs = []
                for pi_val in [math.pi, 3.14]:
                    try:
                        equivs.append(verl_math_equal(extracted_model_output, ground_truth))
                    except Exception:
                        # Fallback to local math_equal if verl's doesn't work
                        equivs.append(
                            math_equal(
                                extracted_model_output, ground_truth, timeout=True, pi=pi_val
                            )
                        )
                correct = any(equivs)
            else:
                try:
                    correct = verl_math_equal(extracted_model_output, ground_truth)
                except Exception:
                    # Fallback to local math_equal
                    correct = math_equal(extracted_model_output, ground_truth, timeout=True)
        except Exception:
            correct = False

    reward = 1.0 if correct else 0.0

    return reward
