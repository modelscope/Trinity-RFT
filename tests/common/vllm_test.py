import os
import unittest

import ray
import torch
from transformers import AutoTokenizer

from trinity.common.config import load_config
from trinity.common.models import create_rollout_models
from trinity.common.models.model import ModelWrapper
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)

config_dir = os.path.join(os.path.dirname(__file__), "tmp", "template_config.yaml")


def get_model_path() -> str:
    path = os.environ.get("MODEL_PATH")
    if not path:
        raise EnvironmentError(
            "Please set `export MODEL_PATH=<your_model_checkpoint_dir>` before running this test."
        )
    return path


CHAT_TEMPLATE = r"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n'}}{% generation %}{{- message.content + '<|im_end|>' + '\n' }}{% endgeneration %}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}{% generation %}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


class BaseTestModelWrapper:
    def test_generate(self):
        prompts = ["Hello, world!", "Hello, my name is"]
        results = self.model_wrapper.generate(prompts)
        self.assertEqual(len(results), len(prompts) * self.config.explorer.repeat_times)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
            {
                "role": "assistant",
                "content": "I'm sorry, but as an AI language model, I don't have access to real-time weather information. To get accurate weather information for your location, you can check a weather website or app, or look outside if possible.",
            },
            {"role": "user", "content": "OK, thanks!"},
        ]
        results = self.model_wrapper.chat(messages)
        self.assertEqual(len(results), self.config.explorer.repeat_times)
        logprobs = self.model_wrapper.logprobs(results[0].tokens)
        self.assertEqual(logprobs.shape[0], results[0].tokens.shape[0])
        messages.append(
            {
                "role": "assistant",
                "content": results[0].response_text,
            }
        )
        exp = self.model_wrapper.convert_messages_to_experience(messages)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        result_dict = tokenizer.apply_chat_template(
            messages,
            chat_template=CHAT_TEMPLATE,
            add_generation_prompt=False,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        self.assertTrue(torch.equal(result_dict["assistant_masks"][0], exp.action_mask))
        self.assertTrue(torch.equal(result_dict["input_ids"][0], exp.tokens))


class TestModelWrapperSync(BaseTestModelWrapper, unittest.TestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = load_config(config_dir)
        self.config.model.model_path = get_model_path()
        self.config.explorer.engine_type = "vllm"
        self.config.explorer.engine_num = 1
        self.config.explorer.chat_template = CHAT_TEMPLATE
        self.engines = create_rollout_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], model_type="vllm")


class TestModelWrapperAsync(BaseTestModelWrapper, unittest.TestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = load_config(config_dir)
        self.config.model.model_path = get_model_path()
        self.config.explorer.engine_type = "vllm_async"
        self.config.explorer.engine_num = 1
        self.config.explorer.chat_template = CHAT_TEMPLATE
        self.engines = create_rollout_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], model_type="vllm_async")


class TestTokenizer(unittest.TestCase):
    def test_assistant_token_mask(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
            {
                "role": "assistant",
                "content": "I'm sorry, but as an AI language model, I don't have access to real-time weather information. To get accurate weather information for your location, you can check a weather website or app, or look outside if possible.",
            },
            {"role": "user", "content": "OK, thanks!"},
            {
                "role": "assistant",
                "content": "You're welcome! If you have any other questions, feel free to ask.",
            },
        ]
        tokenizer = AutoTokenizer.from_pretrained(get_model_path())
        token_ids, action_mask = tokenize_and_mask_messages_default(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE,
        )
        token_ids_hf, action_mask_hf = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE,
        )
        self.assertEqual(token_ids.shape, token_ids_hf.shape)
        self.assertEqual(action_mask.shape, action_mask_hf.shape)
        self.assertTrue(torch.equal(token_ids, token_ids_hf))
        self.assertTrue(torch.equal(action_mask, action_mask_hf))
