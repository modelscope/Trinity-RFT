"""
We use this script to create the toolace dataset files.
This script is adapted from https://github.com/NVlabs/Tool-N1/
The original license and copyright notice are retained below.
#
# =================================================================================
#
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""

import ast
import json
import os
import random
import re

from datasets import Dataset, load_dataset

random.seed(42)

DEFAULT_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. \nBased on the question, you will need to make one or more function/tool calls to achieve the purpose. \nIf none of the function can be used, point it out. If the given question lacks the parameters required by the function,\nalso point it out. You should only return the function call in tools call sections. Here is a list of functions in JSON format that you can invoke:
"""


def _split_top_level_args(params_str):
    items = []
    current = []

    bracket_level = 0
    brace_level = 0
    paren_level = 0

    quote_char = None

    for char in params_str:
        if quote_char:
            if char == quote_char:
                quote_char = None
                current.append(char)
            else:
                current.append(char)
            continue
        else:
            if char in ["'", '"']:
                quote_char = char
                current.append(char)
                continue

        if char == "(":
            paren_level += 1
            current.append(char)
        elif char == ")":
            paren_level -= 1
            current.append(char)
        elif char == "[":
            bracket_level += 1
            current.append(char)
        elif char == "]":
            bracket_level -= 1
            current.append(char)
        elif char == "{":
            brace_level += 1
            current.append(char)
        elif char == "}":
            brace_level -= 1
            current.append(char)
        elif char == "," and bracket_level == 0 and brace_level == 0 and paren_level == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        items.append("".join(current).strip())

    return items


def _parse_params(params_str):
    result = {}
    pairs = _split_top_level_args(params_str)

    for item in pairs:
        if "=" in item:
            key, val = item.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                parsed_val = ast.literal_eval(val)
            except Exception:
                parsed_val = val.strip("'\"")
            result[key] = parsed_val
    return result


def _parse_function_string(function_string):
    function_pattern = r"([a-zA-Z0-9\._\?\|\s\-]+)\((.*?)\)"
    matches = re.findall(function_pattern, function_string)
    parsed_functions = []
    for func_name, params_str in matches:
        func_name = func_name.strip()
        params_dict = _parse_params(params_str)
        parsed_functions.append((func_name, params_dict))

    return parsed_functions


def _extract_functions_from_system(text):
    pattern = r"Here is a list of functions in JSON format that you can invoke:\n(.*?)\nShould you decide to return the function call\(s\)."
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL allows '.' to match newlines
    if match:
        s = match.group(1).strip()
        s = s[:-2] + "]"
        s = json.loads(s)
        return s
    else:
        return None


def _validate_function_format(s):
    if not s or not isinstance(s, str):
        return False

    pattern = (
        r"^\[\s*([a-zA-Z0-9\._\?\|\s\-/]+\(.*?\)\s*)(,\s*[a-zA-Z0-9\._\?\|\s\-/]+\(.*?\)\s*)*\]$"
    )
    return bool(re.match(pattern, s.strip()))


def _process_toolace(data, include_nocall=False):
    filtered = []
    for d in data:
        tools = _extract_functions_from_system(d["system"])
        # print(tools)
        if not tools:
            continue
        item = {"tools": json.dumps(tools)}
        marker = "JSON format that you can invoke:\n"
        idx = d["system"].find(marker)
        item["system"] = d["system"][: idx + len(marker)]
        convs = []
        for c in d["conversations"]:
            # if c["from"] == "gpt" and c["value"].startswith("[") and c["value"].endswith("]"):
            if c["from"] == "assistant" and c["value"].startswith("[") and c["value"].endswith("]"):
                funcs = _parse_function_string(c["value"])
                if funcs and _validate_function_format(c["value"]):
                    c["from"] = "function_call"
                    c["value"] = json.dumps([{"name": f[0], "arguments": f[1]} for f in funcs])
            convs.append(c)
        item["conversations"] = convs
        filtered.append(item)

    print("filtered len")
    print(len(filtered))

    results = []
    for d in filtered:
        cs, sys, tools = d["conversations"], d["system"], d["tools"]
        for i, c in enumerate(cs):
            if (
                c["from"] == "function_call"
                and c["value"].startswith("[")
                and c["value"].endswith("]")
            ):
                results.append(
                    {
                        "system": sys,
                        "conversations": cs[:i],
                        "answer": json.loads(c["value"]),
                        "tools": tools,
                    }
                )
            if c["from"] == "assistant" and include_nocall:
                results.append(
                    {"system": sys, "conversations": cs[:i], "answer": [], "tools": tools}
                )

    results = [r for r in results if r["tools"]]
    final = []
    for r in results:
        if len(r["conversations"]) >= 2:
            id = "toolace_multiple_turn"
        else:
            id = "toolace_single_turn"
        out = {
            "raw_system": r["system"],
            "tools": json.loads(r["tools"]),
            "conversations": r["conversations"],
            "answer": r["answer"],
            "id": id,
        }
        for c in out["conversations"]:
            if (
                c["from"] == "function_call"
                and c["value"].startswith("[")
                and c["value"].endswith("]")
            ):
                c["from"] = "function_call"
        final.append(out)

    return [
        f
        for f in final
        if all(ans["name"] in {tool["name"] for tool in f["tools"]} for ans in f["answer"])
    ]


def pre_process_data(data_names, include_nocall=False):
    # toolace - single turn
    if data_names == "toolace_single_turn":
        dataset_dict = load_dataset("Team-ACE/ToolACE", split="train").to_list()
        print("raw_toolace_single_turn", len(dataset_dict))
        result = _process_toolace(dataset_dict)
    else:
        raise ValueError(
            f"Error: The dataset '{data_names}' is not supported. Please check the dataset name or implement support for it."
        )

    return result


def dict2hg(data):
    hg_data = {"tools": [], "conversations": [], "answer": [], "raw_system": []}
    for item in data:
        tool = json.dumps(item["tools"])
        hg_data["tools"].append(tool)
        hg_data["conversations"].append(item["conversations"])
        answer = json.dumps(item["answer"])
        hg_data["answer"].append(answer)
        hg_data["raw_system"].append(item["raw_system"])
    hg_data = {
        "tools": hg_data["tools"][:],
        "conversations": hg_data["conversations"][:],
        "answer": hg_data["answer"][:],
        "raw_system": hg_data["raw_system"][:],
    }
    hg_data = Dataset.from_dict(hg_data)
    return hg_data


def create_dataset_files(output_dir):
    # make the output directory
    os.makedirs(output_dir, exist_ok=True)

    data_sum = []
    single_turn_toolace = pre_process_data("toolace_single_turn")
    print("toolace_single_turn", len(single_turn_toolace))
    data_sum.extend(single_turn_toolace)

    hg_data = dict2hg(data_sum)
    list_new_data = hg_data.to_list()
    # randomly shuffle
    random.shuffle(list_new_data)

    with open(os.path.join(output_dir, "toolace_clean_data.json"), "w", encoding="utf-8") as f:
        json.dump(list_new_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = f"{current_file_path}/toolace_data"
    create_dataset_files(output_dir)
