# -*- coding: utf-8 -*-
"""
We include the customized toolcall workflows in this file.
Code adapted from https://github.com/NVlabs/Tool-N1
Reference Paper https://arxiv.org/pdf/2505.00024 for further details.
"""

import json
import re
from collections import Counter
from typing import List

from trinity.common.experience import Experience
from trinity.common.workflows.workflow import SimpleWorkflow, Task

# Adapted from https://github.com/NVlabs/Tool-N1
qwen_tool_prompts = """# Tool

<tools>
{tools}
</tools>

In each action step, you MUST:
1. Think about the reasoning process in the mind and enclosed your reasoning within <think> </think> XML tags.
2. Then, provide a json object with function names and arguments within <tool_call></tool_call> XML tags. i.e., <tool_call>[{{"name": <function-name>, "arguments": <args-json-object>}}, {{"name": <function-name2>, "arguments": <args-json-object2>}}, ...]</tool_call>
3. Make sure both the reasoning and the tool call steps are included together in one single reply.
A complete reply example is: <think>To address the query, I need to send the email to Bob and then buy the banana through walmart. </think> <tool_call> [{{"name": "email", "arguments": {{"receiver": "Bob", "content": "I will bug banana through walmart"}}}}, {{"name": "walmart", "arguments": {{"input": "banana"}}}}]</tool_call>. Please make sure the type of the arguments is correct.
"""


# Adapted from https://github.com/NVlabs/Tool-N1
def construct_prompt(dp):
    def format_tools(tools):
        tools = json.loads(tools)
        string = ""
        for tool in tools:
            string += json.dumps({"type": "function", "function": tool}) + "\n"
        if string[-1] == "\n":
            string = string[:-1]
        return string

    tools = format_tools(dp["tools"])
    tool_prompt = qwen_tool_prompts.format(tools=tools)
    system = dp["raw_system"]
    conversations = dp["conversations"]
    prompt = []
    prompt.append({"role": "system", "content": system + tool_prompt})
    for tem in conversations:
        if tem["from"] == "human" or tem["from"] == "user":
            prompt.append({"role": "user", "content": tem["value"]})
        elif tem["from"] == "gpt" or tem["from"] == "assistant":
            prompt.append({"role": "assistant", "content": tem["value"]})
        elif tem["from"] == "observation" or tem["from"] == "tool":
            prompt.append({"role": "tool", "content": tem["value"]})
        elif tem["from"] == "function_call":
            prompt.append({"role": "assistant", "content": json.dumps(tem["value"])})
    return prompt


# Adapted from https://github.com/NVlabs/Tool-N1
def validate_result(result, answer):
    if len(result) == 0 or len(answer) == 0:
        if len(result) == len(answer):
            return 2
        else:
            return 0
    else:
        try:
            counter1_full = Counter(
                (item["name"], json.dumps(item["arguments"], sort_keys=True)) for item in result
            )
            counter2_full = Counter(
                (item["name"], json.dumps(item["arguments"], sort_keys=True)) for item in answer
            )
        except TypeError:
            return 0
        if counter1_full == counter2_full:
            return 2

        counter1_name = Counter(item["name"] for item in result)
        counter2_name = Counter(item["name"] for item in answer)

        if counter1_name == counter2_name:
            return 1

        return 0


# Adapted from https://github.com/NVlabs/Tool-N1
def validate_format(tool_call_list):
    for item in tool_call_list:
        if not isinstance(item, dict):
            return 0
    for item in tool_call_list:
        if "name" not in item.keys() or "arguments" not in item.keys():
            return 0
    return 1


# Adapted from https://github.com/NVlabs/Tool-N1
def extract_solution_v0_toolN1(tool_call_str):
    output_string = tool_call_str

    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, output_string
    last_content = matches[-1].group(1).strip()
    try:
        return json.loads(last_content), output_string
    except json.JSONDecodeError:
        return None, output_string


# Adapted from https://github.com/NVlabs/Tool-N1
def compute_score_v0_toolN1(  # noqa: C901
    solution_str,
    ground_truth,
    do_print=False,
):
    answer = json.loads(ground_truth)

    result, output_string = extract_solution_v0_toolN1(solution_str)

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = None

    if isinstance(result, dict):
        tem = []
        tem.append(result)
        result = tem

    if isinstance(answer, str):
        answer = json.loads(answer)

    if do_print:
        print("************solution_str************")
        print(solution_str)
        print(f"Extracted result: {result}")
        print(f"Solution string: {answer}")

    if result is not None:
        if "<think>" not in output_string or "</think>" not in output_string:
            if do_print:
                print("--------" * 5 + "\n\n")
                print("not thinking:", -1)
            return 0

    if result is None:
        if do_print:
            print("--------" * 5 + "\n\n")
            print("result is None:", -1)
        return 0

    # added rule1
    if solution_str.count("<think>") != 1 or solution_str.count("</think>") != 1:
        if do_print:
            print("--------" * 5 + "\n\n")
            print(
                f"Fail, think tag appear not once: "
                f"<think> appear {solution_str.count('<think>')} times, "
                f"</think> appear {solution_str.count('</think>')} times",
                -1,
            )
        return 0

    # added rule2
    think_end_pos = solution_str.find("</think>")
    tool_call_start_pos = solution_str.find("<tool_call>")

    if tool_call_start_pos != -1:
        if think_end_pos > tool_call_start_pos:
            if do_print:
                print("--------" * 5 + "\n\n")
                print("Fail: <think> tag must before <tool_call> tag", -1)
            return 0

    if not validate_format(result):
        if do_print:
            print("--------" * 5 + "\n\n")
            print("result wrong formate:", -1)
        return 0

    if validate_result(result, answer) == 2:
        if do_print:
            print("--------" * 5 + "\n\n")
            print("get full core:", 1)
        return 1
    else:
        if do_print:
            print("--------" * 5 + "\n\n")
            print("wrong answer", -1)
        return 0


def compute_toolcall_reward(
    solution_str: str,
    ground_truth: str,
) -> float:
    res = compute_score_v0_toolN1(solution_str, ground_truth)
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


class ToolCallWorkflow(SimpleWorkflow):
    """
    A workflow for toolcall tasks.
    Prompt construction and reward function from https://github.com/NVlabs/Tool-N1

    Only support qwen model for now. You can change the prompt construction and reward calculation by yourself for other models.
    """

    is_async: bool = True

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        self.workflow_args = task.workflow_args
        self.reward_fn_args = task.reward_fn_args

    def format_prompt(self):
        raw_task = self.raw_task
        messages = construct_prompt(raw_task)
        return messages

    async def run_async(self) -> List[Experience]:
        messages = self.format_prompt()

        self.logger.debug("start chat")
        responses = await self.model.chat_async(messages, **self.rollout_args)

        for i, response in enumerate(responses):
            reward = 0.0

            if self.raw_task is not None:
                ground_truth = self.raw_task.get("answer")
                if ground_truth is not None:
                    reward = compute_toolcall_reward(
                        solution_str=response.response_text,
                        ground_truth=ground_truth,
                    )
                else:
                    self.logger.error(
                        "Key 'answer' not found in self.raw_task. Assigning default reward."
                    )
            else:
                self.logger.error("self.raw_task is None. Assigning default reward.")
            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
            response.reward = reward
            response.eid.run = i + self.run_id_base
        return responses
