# Adapted from Reasoning360: https://github.com/LLM360/Reasoning360/blob/main/verl/utils/reward_score/naive_dapo.py

import contextlib
import math
import re
from math import isclose
from typing import Optional, Union

import sympy
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from verl.utils.py_functional import timeout_limit


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


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        return abs(float(x) - int(round(float(x)))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  # implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "liter",
    ]:
        expr = re.sub(r"{}(es)?(s)? *(\^[0-9]+)?".format(unit), "", expr)
    expr = re.sub(r"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


# @timeout(timeout_seconds=10)
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


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


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
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


def grade_answer(given_answer: str, ground_truth: str) -> tuple[bool, str]:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True, given_answer_normalized_mathd

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False, given_normalized

    if ground_truth_normalized == given_normalized:
        return True, given_normalized

    if len(given_normalized) == 0:
        return False, given_normalized

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct, given_normalized


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()


def match_answer(response):
    is_matched = False
    response = response.split("</think>")[-1]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    return is_matched, response


def compute_score(solution_str: str, ground_truth: Optional[str]) -> float:
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
            if "\\pi" in extracted_model_output or "\\pi" in ground_truth:
                equivs = []
                for pi in [math.pi, 3.14]:
                    equivs.append(
                        math_equal(extracted_model_output, ground_truth, timeout=True, pi=pi)
                    )
                    correct = any(equivs)
            else:
                correct = math_equal(extracted_model_output, ground_truth, timeout=True)
        except Exception:
            correct = False

    # reward = 1.0 if correct else -1.0
    reward = 1.0 if correct else 0.0

    return reward
