"""
This script is used to use VLLM to generate rollout samples from the converted checkpoints.
"""

import argparse
import copy
import gc
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from trinity.common.constants import PLUGIN_DIRS_ENV_VAR
from trinity.utils.plugin_loader import load_plugins


def init_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device_count = torch.cuda.device_count()
    print(f"device_count={device_count}")
    if device_count < 1:
        raise RuntimeError("No GPU available for multi-card inference.")
    print(f"Loading model from: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=device_count)
    print("Model loaded successfully!")
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=512,
        repetition_penalty=1.2,
    )
    return llm, tokenizer, sampling_params


def rollout(llm, tokenizer, sampling_params, input_file_path, output_file_path, rollout_repeat=3):
    from trinity.plugins.prompt_learn2ask import rollout_prompt_med as rollout_prompt

    with open(input_file_path, "r") as lines:
        sample_list = [json.loads(line.strip()) for line in lines]
    print(f"loaded samples: {len(sample_list)}")

    for index, sample in enumerate(sample_list):
        record = copy.deepcopy(sample)
        print(f"index: {index}, session_id: {sample['session_id']}")
        messages = [{"role": "system", "content": rollout_prompt}] + sample["messages"]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        response_list = []
        for i in range(rollout_repeat):
            time_probe = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params=sampling_params)
            print(f"time cost: {time.perf_counter() - time_probe}")
            for output in outputs:
                response = output.outputs[0].text
                response_list.append(response)
                print(f"rollout #{i}: {response}\n")
        record["rollouts"] = response_list

        with open(output_file_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def eval_sample(llm, tokenizer, sampling_params, input_file_path, output_file_path):
    from trinity.plugins.prompt_learn2ask import reward_prompt_med as grader_prompt

    print(f"input_file_path: {input_file_path}")
    print(f"output_file_path: {output_file_path}")

    with open(input_file_path, "r") as lines:
        sample_list = [json.loads(line.strip()) for line in lines]
    print(f"Total records: {len(sample_list)}")

    def res_formater(res_content):
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.findall(pattern, res_content)
        result = {}
        for tag_name, content in matches:
            result[tag_name] = content
        return result

    def msg2str(msg_list):
        result_str = ""
        for msg in msg_list:
            if msg["role"] == "user":
                result_str += f"patient: {msg['content']}\n"
            if msg["role"] == "assistant":
                result_str += f"doctor: {msg['content']}\n"
        return result_str

    for index, sample in enumerate(sample_list):
        print(f"index: {index}, cid: {sample['cid']}")
        action_truth = sample["decision_truth"]
        info_truth = sample["info_truth"] if sample["info_truth"] else "None."
        print(f"action_truth: {action_truth}, info_truth:{info_truth}")

        sys_prompt = grader_prompt.format(info_truth)
        history = msg2str(sample["messages"])

        sample["grades"] = []
        for rollout in sample["rollouts"]:
            time_probe = time.perf_counter()
            action_score, content_score, format_score, res_think = 0, 0, 0, "NA"
            if "<stop />" in rollout:
                action_rollout = "stop"
            else:
                action_rollout = "continue"
            if action_truth == action_rollout:
                action_score = 1
                if action_truth == "continue":
                    user_content = history + f"doctor: {rollout}"
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    outputs = llm.generate([prompt], sampling_params=sampling_params)
                    for output in outputs:
                        response = output.outputs[0].text
                        print(f"Response: {response}\n")
                    res_dict = res_formater(response)
                    try:
                        format_score = float(res_dict.get("format_score", 0.0))
                        content_score = float(res_dict.get("content_score", 0.0))
                        res_think = res_dict.get("think", "None")
                    except Exception as e:
                        print(e)
                else:
                    content_score = 1.0
                    format_score = 1.0 if rollout == "<stop />" else 0.0
            else:
                action_score, format_score, content_score = 0, 0, 0
            grade_result = {
                "think": res_think,
                "action_score": action_score,
                "format_score": format_score,
                "content_score": content_score,
            }
            sample["grades"].append(grade_result)
            print(f"grade_result:{json.dumps(grade_result, ensure_ascii=False, indent=2)}")
            print(f"time_cost:{time.perf_counter() - time_probe}")
        with open(output_file_path, "a") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print("\n======================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_repeat", type=int, default=3)

    # Ckpt for testing
    parser.add_argument("--eval_model_path", type=str, required=True)

    # Model to empower the grading, Qwen2.5-32b-instruct is recommended
    parser.add_argument("--grader_model_path", type=str, required=True)

    # Your test sample path [input]
    parser.add_argument("--test_file_path", type=str, required=True)

    # Rollout results given test samples [output]
    parser.add_argument("--rollout_file_path", type=str, required=True)

    # Final output given rollout results [output]
    parser.add_argument("--eval_file_path", type=str, required=True)

    args = parser.parse_args()

    os.environ[PLUGIN_DIRS_ENV_VAR] = os.path.join(os.path.dirname(__file__), "..", "workflow")
    load_plugins()

    # rollout stage
    llm, tokenizer, sampling_params = init_llm(args.eval_model_path)
    rollout(
        llm,
        tokenizer,
        sampling_params,
        args.test_file_path,
        args.rollout_file_path,
        args.rollout_repeat,
    )
    del llm  # clean up the memory after the inference
    gc.collect()
    torch.cuda.empty_cache()  # release gpu memory

    # eval stage
    llm2, tokenizer2, sampling_params2 = init_llm(args.grader_model_path)
    eval_sample(llm2, tokenizer2, sampling_params2, args.rollout_file_path, args.eval_file_path)
