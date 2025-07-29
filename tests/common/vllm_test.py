import os
import unittest

import torch
from parameterized import parameterized_class
from transformers import AutoTokenizer

from tests.tools import RayUnittestBase, RayUnittestBaseAysnc, get_template_config
from trinity.common.models import create_inference_models
from trinity.common.models.model import ModelWrapper
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)


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


@parameterized_class(
    ("tensor_parallel_size", "engine_num", "use_v1", "repeat_times", "enable_history", "use_async"),
    [
        (1, 2, False, 2, True, False),
        (2, 2, False, 1, False, True),
        (2, 2, True, 2, True, False),
        (1, 2, True, 1, False, True),
        (2, 1, True, 3, True, True),
    ],
)
class ModelWrapperTest(RayUnittestBaseAysnc):
    def setUp(self):
        # configure the model
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_num = self.engine_num
        self.config.explorer.rollout_model.tensor_parallel_size = self.tensor_parallel_size
        self.config.explorer.rollout_model.use_v1 = self.use_v1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.algorithm.repeat_times = self.repeat_times
        self.config.explorer.rollout_model.enable_history = self.enable_history
        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(
            self.engines[0], model_type="vllm_async", enable_history=self.enable_history
        )

    async def test_generate(
        self,
    ):
        prompts = ["Hello, world!", "Hello, my name is"]
        n = self.config.algorithm.repeat_times
        if self.use_async:
            generate_results = await self.model_wrapper.generate_async(
                prompts, n=n, temperature=1.0
            )
        else:
            generate_results = self.model_wrapper.generate(prompts, n=n, temperature=1.0)
        self.assertEqual(len(generate_results), len(prompts) * n)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history(
                clear_history=False
            )
            self.assertEqual(len(history_experiences), len(generate_results))
            for exp, history_exp in zip(generate_results, history_experiences):
                self.assertEqual(exp.response_text, history_exp.response_text)
                self.assertEqual(exp.tokens.tolist(), history_exp.tokens.tolist())
                self.assertEqual(exp.prompt_length, history_exp.prompt_length)
                self.assertEqual(exp.logprobs.tolist(), history_exp.logprobs.tolist())
        else:
            with self.assertRaises(ValueError):
                self.model_wrapper.extract_experience_from_history(clear_history=False)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
            {
                "role": "assistant",
                "content": "I'm sorry, but as an AI language model, I don't have access to real-time weather information. To get accurate weather information for your location, you can check a weather website or app, or look outside if possible.",
            },
            {"role": "user", "content": "OK, thanks!"},
        ]
        if self.use_async:
            results = await self.model_wrapper.chat_async(messages, n=n, temperature=1.0)
        else:
            results = self.model_wrapper.chat(messages, n=n, temperature=1.0)
        self.assertEqual(len(results), n)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertEqual(len(history_experiences) - len(generate_results), len(results))
            for exp, history_exp in zip(results, history_experiences[len(generate_results) :]):
                self.assertEqual(exp.response_text, history_exp.response_text)
                self.assertEqual(exp.tokens.tolist(), history_exp.tokens.tolist())
                self.assertEqual(exp.prompt_length, history_exp.prompt_length)
                self.assertEqual(exp.logprobs.tolist(), history_exp.logprobs.tolist())
        for result in results:
            self.assertTrue(torch.any(result.logprobs != 0))
        if self.use_async:
            logprobs = await self.model_wrapper.logprobs_async(results[0].tokens.tolist())
        else:
            logprobs = self.model_wrapper.logprobs(results[0].tokens.tolist())
        self.assertEqual(logprobs.shape[0], results[0].tokens.shape[0] - 1)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertTrue(len(history_experiences) == 0)
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
        prompt_length = torch.argmax(result_dict["assistant_masks"][0]).item()
        self.assertTrue(
            torch.equal(result_dict["assistant_masks"][0][prompt_length:], exp.action_mask)
        )
        self.assertTrue(torch.equal(result_dict["input_ids"][0], exp.tokens))
        self.assertRaises(ValueError, self.model_wrapper.get_openai_client)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertTrue(len(history_experiences) == 0)


class TestAPIServer(RayUnittestBase):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.use_v1 = True
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(
            self.engines[0], model_type="vllm_async", enable_history=True
        )
        self.model_wrapper_no_history = ModelWrapper(
            self.engines[0], model_type="vllm_async", enable_history=False
        )

    def test_api(self):
        openai_client = self.model_wrapper.get_openai_client()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        model_id = openai_client.models.list().data[0].id
        response = openai_client.chat.completions.create(model=model_id, messages=messages, n=1)
        self.assertEqual(1, len(response.choices))
        self.assertTrue(len(response.choices[0].message.content) > 0)
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=2,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(response.choices[0].logprobs is not None)
        self.assertEqual(0, len(response.choices[0].logprobs.content[0].top_logprobs))
        self.assertTrue(response.choices[0].logprobs.content[0].logprob < 0)
        self.assertTrue(hasattr(response, "prompt_token_ids"))
        self.assertTrue(len(response.prompt_token_ids) > 0)
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(len(response.choices[0].token_ids) > 0)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 3)
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=4,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 4)
        self.assertEqual(len(self.model_wrapper.extract_experience_from_history()), 0)
        response = self.model_wrapper_no_history.get_openai_client().chat.completions.create(
            model=model_id, messages=messages, n=2
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(len(response.choices[0].token_ids) > 0)
        with self.assertRaises(ValueError):
            self.model_wrapper_no_history.extract_experience_from_history()
        self.assertEqual(len(self.model_wrapper_no_history.history), 0)


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
        token_ids, action_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE,
        )
        token_ids_hf, action_mask_hf, prompt_length_hf = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE,
        )
        self.assertEqual(token_ids.shape, token_ids_hf.shape)
        self.assertEqual(action_mask.shape, action_mask_hf.shape)
        self.assertTrue(torch.equal(token_ids, token_ids_hf))
        self.assertTrue(torch.equal(action_mask, action_mask_hf))
        self.assertEqual(prompt_length, prompt_length_hf)
