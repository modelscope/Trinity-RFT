# -*- coding: utf-8 -*-
"""
Utility functions for strictly parsing and evaluating mathematical answers.

This module is a modified and simplified version of the official evaluation code
for Qwen2.5-Math, designed for easier standalone use.

Original source: https://github.com/QwenLM/Qwen2.5-Math

Key modifications include:

1.  Retained only the core parsing logic for the common `qwen_boxed` prompt format.
2.  Consolidated essential parsing and evaluation functions from multiple files
    into this single module.
3.  Simplified benchmark handling and conditional logic for broader applicability.
4.  Simplified or removed calls to external tools like TIR.
"""

import re
from math import isclose
from typing import Any, Dict, Optional, Tuple

import sympy
from sympy import Matrix, N, simplify, sympify
from sympy.parsing.latex import parse_latex
from word2number import w2n


def verify_math_answer(response_text: str, ground_truth: str) -> Tuple[float, Dict[str, Any]]:
    """Strictly compare the equality of response and groundtruth."""
    # Parse the response
    parsed_prediction = extract_answer(response_text)

    # Parse the ground truth
    parsed_truth = extract_answer(str(ground_truth))

    is_correct = math_equal(prediction=parsed_prediction, reference=parsed_truth)

    accuracy = 1.0 if is_correct else 0.0

    eval_details = {
        "parsed_prediction": parsed_prediction,
        "ground_truth": parsed_truth,
        "is_correct": is_correct,
    }

    return accuracy, eval_details


def extract_answer(response_text: str) -> Optional[str]:
    """Extract the equation from the string."""
    if not isinstance(response_text, str):
        return None

    # Extract '\boxed{...}'
    if "boxed" in response_text:
        ans_part = response_text.split("boxed")[-1]
        if not ans_part:
            return None

        if ans_part.startswith("{"):
            stack = 1
            extracted_ans = ""
            for char in ans_part[1:]:
                if char == "{":
                    stack += 1
                    extracted_ans += char
                elif char == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    extracted_ans += char
                else:
                    extracted_ans += char

            if stack == 0:
                return strip_string(extracted_ans)

        match = re.search(r"\{?([^$}]+)\}?", ans_part)
        if match:
            return strip_string(match.group(1))

    # Extract 'answer is ...'
    search_patterns = [r"(?:final|the)\s+answer\s+is\s*:?\s*(.+)", r"答案是\s*:?\s*(.+)"]
    for pattern in search_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            pred = strip_string(match.group(1))
            if pred and pred.endswith("."):
                pred = pred[:-1]
            return pred

    # Extract the last number
    text_no_commas = response_text.replace(",", "")
    numeric_finds = re.findall(r"[-+]?\d*\.?\d+", text_no_commas)
    if numeric_finds:
        return numeric_finds[-1]

    return None


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts if not t.endswith("s")])


def strip_string(input_str: Optional[str]) -> Optional[str]:
    """Clean and normalize math answer strings."""
    if input_str is None:
        return None

    string = str(input_str).strip()

    # Basic cleaning and formatting
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("%", "").replace("\\%", "")

    # Normalization of LaTeX format
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")
    string = re.sub(r"\\mbox{.*?}", "", string)
    string = string.replace("\\mathbf", "")

    string = string.replace("^{\\circ}", "").replace("^\\circ", "")

    # Remove text and units
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)

    for _ in range(2):
        for unit_text in unit_texts:
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Clean numerical values
    try:
        string = str(w2n.word_to_num(string))
    except ValueError:
        pass

    string = re.sub(r"^[a-zA-Z]\s*=\s*", "", string)

    string = string.replace("infinity", "\\infty").replace("inf", "\\infty")

    string = re.sub(r"(\d+)\.0+([^\d]|$)", r"\1\2", string)

    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string.startswith("."):
        string = "0" + string

    # Fix the structure and final cleanup
    string = string.replace(" ", "")

    string = fix_sqrt(string)
    string = fix_fracs(string)
    string = fix_a_slash_b(string)

    if (string.startswith("{") and string.endswith("}")) or (
        string.startswith("(") and string.endswith(")")
    ):
        string = string[1:-1]

    if not string:
        return None

    return string.strip()


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
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


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    pattern = r"\\sqrt\s*(\w+)"
    replacement = r"\\sqrt{\1}"

    return re.sub(pattern, replacement, string)


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except Exception:
        pass
    return text


def _compare_numerical(pred: str, ref: str) -> Optional[bool]:
    """Helper for numerical comparison in math_equal."""
    try:
        if _is_digit(pred) and _is_digit(ref):
            pred_num = _parse_digits(pred)
            ref_num = _parse_digits(ref)
            # Allow for percentage conversions
            possible_ref_values = {ref_num, ref_num / 100, ref_num * 100}
            for val in possible_ref_values:
                if numeric_equal(pred_num, val):
                    return True
            return False
    except (ValueError, TypeError):
        pass
    return None


def _compare_structures(pred: str, ref: str) -> Optional[bool]:
    """Helper for structural comparison (intervals, matrices) in math_equal."""
    is_pred_interval = pred.startswith(("(", "[")) and pred.endswith((")", "]"))
    is_ref_interval = ref.startswith(("(", "[")) and ref.endswith((")", "]"))
    if is_pred_interval and is_ref_interval:
        pred_parts = pred[1:-1].split(",")
        ref_parts = ref[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(p.strip(), r.strip()) for p, r in zip(pred_parts, ref_parts)
        ):
            return True

    is_pred_matrix = pred.startswith("\\begin{pmatrix}")
    is_ref_matrix = ref.startswith("\\begin{pmatrix}")
    if is_pred_matrix and is_ref_matrix:
        pred_mat_str = pred[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")]
        ref_mat_str = ref[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")]
        pred_rows = [row.split("&") for row in pred_mat_str.split("\\\\")]
        ref_rows = [row.split("&") for row in ref_mat_str.split("\\\\")]
        if (
            len(pred_rows) == len(ref_rows)
            and len(pred_rows[0]) == len(ref_rows[0])
            and all(
                math_equal(p_elem.strip(), r_elem.strip())
                for p_row, r_row in zip(pred_rows, ref_rows)
                for p_elem, r_elem in zip(p_row, r_row)
            )
        ):
            return True
    return None


def _compare_equations(pred: str, ref: str) -> Optional[bool]:
    """Helper for equation comparison in math_equal."""
    if pred.count("=") == 1 and ref.count("=") == 1:
        pred_lhs, pred_rhs = (p.strip() for p in pred.split("="))
        ref_lhs, ref_rhs = (r.strip() for r in ref.split("="))
        if symbolic_equal(f"({pred_lhs})-({pred_rhs})", f"({ref_lhs})-({ref_rhs})"):
            return True

    if pred.count("=") == 1 and ref.count("=") == 0:
        var, val = (p.strip() for p in pred.split("="))
        if len(var) <= 2 and math_equal(val, ref):
            return True

    if ref.count("=") == 1 and pred.count("=") == 0:
        var, val = (r.strip() for r in ref.split("="))
        if len(var) <= 2 and math_equal(pred, val):
            return True
    return None


def math_equal(prediction: Optional[str], reference: Optional[str]) -> bool:
    """Checks the mathematical equality of two strings by trying different methods."""
    if prediction is None or reference is None:
        return False
    if prediction == reference:
        return True

    comparisons = [
        _compare_numerical,
        _compare_structures,
        _compare_equations,
    ]
    for func in comparisons:
        result = func(prediction, reference)
        if result is not None:
            return result

    return symbolic_equal(prediction, reference)


def numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def _is_digit(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _parse_digits(s: str) -> float:
    return float(s.replace(",", ""))


def _parse_symbolic(s: str) -> Any:
    """Parse a string into a sympy expression, trying different methods."""
    s_cleaned = s.replace("\\\\", "\\")
    try:
        # Use a local dict to handle functions like exp
        return parse_latex(s_cleaned, locals={"exp": sympy.exp})
    except Exception:  # Broad exception is okay here as we have fallbacks
        try:
            return sympify(s_cleaned, evaluate=True)
        except (sympy.SympifyError, TypeError, SyntaxError):
            return s  # Return original string if all parsing fails


def _check_direct_or_simplified_equality(a_sym: Any, b_sym: Any) -> Optional[bool]:
    """Check for direct or simplified symbolic equality."""
    try:
        if a_sym == b_sym:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if simplify(a_sym - b_sym) == 0:
            return True
    except (AttributeError, TypeError, ValueError, sympy.SympifyError):
        pass
    return None


def _check_equation_equality(a_sym: Any, b_sym: Any) -> Optional[bool]:
    """Check for symbolic equality of two equations."""
    try:
        if isinstance(a_sym, sympy.Eq) and isinstance(b_sym, sympy.Eq):
            if (a_sym.lhs - a_sym.rhs).equals(b_sym.lhs - b_sym.rhs):
                return True
    except (AttributeError, TypeError):
        pass
    return None


def _check_numeric_value_equality(a_sym: Any, b_sym: Any) -> Optional[bool]:
    """Check for equality of the numerical values of two symbolic expressions."""
    try:
        if numeric_equal(float(N(a_sym)), float(N(b_sym))):
            return True
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def _check_matrix_equality(a_sym: Any, b_sym: Any) -> Optional[bool]:
    """Check for symbolic equality of two matrices."""
    try:
        if isinstance(a_sym, Matrix) and isinstance(b_sym, Matrix):
            if a_sym.shape == b_sym.shape and simplify(a_sym - b_sym).is_zero_matrix:
                return True
    except (AttributeError, TypeError):
        pass
    return None


def symbolic_equal(a: str, b: str) -> bool:
    """Compares two strings for symbolic equivalence using sympy."""
    a_sym = _parse_symbolic(a)
    b_sym = _parse_symbolic(b)

    equality_checks = [
        _check_direct_or_simplified_equality,
        _check_equation_equality,
        _check_numeric_value_equality,
        _check_matrix_equality,
    ]

    for check_func in equality_checks:
        result = check_func(a_sym, b_sym)
        if result is True:
            return True

    return False
