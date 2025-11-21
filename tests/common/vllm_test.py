import unittest

import ray
import torch
from openai import BadRequestError
from parameterized import parameterized_class
from transformers import AutoTokenizer

from tests.tools import (
    RayUnittestBaseAysnc,
    get_api_model_path,
    get_model_path,
    get_template_config,
)
from trinity.common.constants import ROLLOUT_WEIGHT_SYNC_GROUP_NAME
from trinity.common.models import create_inference_models
from trinity.common.models.model import ModelWrapper
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)

DEBUG = False


def print_debug(*args):
    if DEBUG:
        print(*args)


# Qwen2.5 chat template with {% generation %} mark
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
        {{- '<|im_start|>' + message.role + '\n'}}{% generation %}
        {%- if message.content %}
            {{- message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>\n' }}
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
    (
        "tensor_parallel_size",
        "engine_num",
        "repeat_times",
        "enable_history",
        "use_async",
    ),
    [
        (2, 2, 2, True, False),
        (1, 2, 1, False, True),
        (2, 1, 3, True, True),
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
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.algorithm.repeat_times = self.repeat_times
        self.config.explorer.rollout_model.enable_history = self.enable_history
        self.config.check_and_update()
        from pprint import pprint

        pprint(self.config)
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(
            self.engines[0], engine_type="vllm", enable_history=self.enable_history
        )

    async def test_generate(
        self,
    ):
        await self.model_wrapper.prepare()
        self.assertEqual(self.model_wrapper.model_path, self.config.model.model_path)
        self.assertEqual(await self.model_wrapper.model_path_async, self.config.model.model_path)
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
        self.assertTrue(exp.logprobs.shape[0] == exp.tokens.shape[0] - prompt_length)
        self.assertTrue(torch.equal(result_dict["input_ids"][0], exp.tokens))
        self.assertRaises(ValueError, self.model_wrapper.get_openai_client)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertTrue(len(history_experiences) == 0)


@parameterized_class(
    (
        "max_model_len",
        "max_prompt_tokens",
        "max_response_tokens",
    ),
    [
        (20, 19, None),
        (20, None, 1),
    ],
)
class TestModelLen(RayUnittestBaseAysnc):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = self.max_model_len
        self.config.model.max_prompt_tokens = self.max_prompt_tokens
        self.config.model.max_response_tokens = self.max_response_tokens
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    async def test_model_len(self):
        await self.model_wrapper.prepare()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
        ]
        response = self.model_wrapper.chat(messages)
        self.assertEqual(len(response), 1)
        self.assertEqual(len(response[0].tokens), self.max_model_len)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertEqual(len(exps[0].tokens), self.max_model_len)

        # max_prompt_tokens and max_response_tokens do not work with openai api
        openai_client = self.model_wrapper.get_openai_client()
        model_id = openai_client.models.list().data[0].id
        with self.assertRaises(BadRequestError):
            # the prompt is longer than max_model_len
            openai_client.chat.completions.create(model=model_id, messages=messages, n=1)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 0)

        response = openai_client.chat.completions.create(model=model_id, messages=messages[1:], n=1)
        self.assertEqual(len(response.choices), 1)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        # only generate max_response_tokens tokens
        self.assertEqual(
            len(exps[0].tokens),
            response.usage.prompt_tokens + self.config.model.max_response_tokens,
        )


class TestAPIServer(RayUnittestBaseAysnc):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)
        self.model_wrapper_no_history = ModelWrapper(
            self.engines[0], engine_type="vllm", enable_history=False
        )

    async def test_api(self):
        await self.model_wrapper.prepare()
        await self.model_wrapper_no_history.prepare()
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
        self.assertEqual(0, len(response.choices[0].logprobs.content[2].top_logprobs))
        # here we check the 3rd token logprob, because the first two tokens (`<think>`,`\n` usually have zero logprob)
        self.assertTrue(response.choices[0].logprobs.content[2].logprob < 0)
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
        for exp in exps:
            self.assertTrue(len(exp.tokens) > 0)
            self.assertTrue(len(exp.logprobs) > 0)
            self.assertTrue(exp.prompt_length + len(exp.logprobs) == len(exp.tokens))
        self.assertEqual(len(self.model_wrapper.extract_experience_from_history()), 0)
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].tokens) > 0)
        self.assertTrue(len(exps[0].logprobs) > 0)
        self.assertTrue(exps[0].prompt_length + len(exps[0].logprobs) == len(exps[0].tokens))
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            logprobs=False,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].logprobs) == 0)
        response = self.model_wrapper_no_history.get_openai_client().chat.completions.create(
            model=model_id, messages=messages, n=2
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(len(response.choices[0].token_ids) > 0)
        with self.assertRaises(ValueError):
            self.model_wrapper_no_history.extract_experience_from_history()
        self.assertEqual(len(self.model_wrapper_no_history.history), 0)


class DummySynchronizer:
    def __init__(self):
        pass

    def do_nothing(self):
        pass


class TestLogprobs(RayUnittestBaseAysnc):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    async def test_logprobs(self):
        # use init process group to apply patches
        sync = (
            ray.remote(DummySynchronizer)
            .options(name="synchronizer", namespace=self.config.ray_namespace)
            .remote()
        )
        await sync.__ray_ready__.remote()
        await self.model_wrapper.prepare()
        master_address, master_port = await self.engines[0].get_available_address.remote()
        await self.engines[0].init_process_group.remote(
            master_address,
            master_port,
            world_size=1,
            rank_offset=0,
            group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
            explorer_name=self.config.explorer.name,
            timeout=20,
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        response_1 = self.model_wrapper.chat(messages, n=1, temperature=1.0, logprobs=True)[0]
        response_2 = self.model_wrapper.chat(messages, n=1, temperature=0.8, logprobs=True)[0]
        self.assertTrue(response_1.logprobs is not None)
        self.assertTrue(len(response_1.logprobs) > 0)
        self.assertTrue(response_2.logprobs is not None)
        self.assertTrue(len(response_2.logprobs) > 0)
        logprobs_1 = self.model_wrapper.logprobs(response_1.tokens.tolist(), temperature=1.0)
        logprobs_2 = self.model_wrapper.logprobs(response_1.tokens.tolist(), temperature=0.8)
        logprobs_3 = self.model_wrapper.logprobs(response_2.tokens.tolist(), temperature=1.0)
        logprobs_4 = self.model_wrapper.logprobs(response_2.tokens.tolist(), temperature=0.8)
        self.assertEqual(logprobs_1.shape, logprobs_2.shape)
        self.assertEqual(logprobs_3.shape, logprobs_4.shape)
        self.assertFalse(torch.allclose(logprobs_1, logprobs_2, rtol=0.4))
        self.assertFalse(torch.allclose(logprobs_3, logprobs_4, atol=0.4))
        logprobs_1_prompt = logprobs_1[: response_1.prompt_length - 1]
        logprobs_2_prompt = logprobs_2[: response_1.prompt_length - 1]
        logprobs_3_prompt = logprobs_3[: response_2.prompt_length - 1]
        logprobs_4_prompt = logprobs_4[: response_2.prompt_length - 1]
        self.assertEqual(logprobs_1_prompt.shape, logprobs_2_prompt.shape)
        self.assertFalse(torch.allclose(logprobs_1_prompt, logprobs_2_prompt, rtol=0.4))
        self.assertFalse(torch.allclose(logprobs_3_prompt, logprobs_4_prompt, rtol=0.4))
        self.assertTrue(torch.allclose(logprobs_1_prompt, logprobs_3_prompt, rtol=0.4))
        self.assertTrue(torch.allclose(logprobs_2_prompt, logprobs_4_prompt, rtol=0.4))
        logprobs_1_response = logprobs_1[response_1.prompt_length - 1 :]
        logprobs_2_response = logprobs_2[response_1.prompt_length - 1 :]
        logprobs_3_response = logprobs_3[response_2.prompt_length - 1 :]
        logprobs_4_response = logprobs_4[response_2.prompt_length - 1 :]
        self.assertEqual(logprobs_1_response.shape, logprobs_2_response.shape)
        self.assertEqual(logprobs_3_response.shape, logprobs_4_response.shape)
        self.assertEqual(logprobs_1_response.shape, logprobs_2_response.shape)
        self.assertEqual(response_1.logprobs.shape, logprobs_1_response.shape)
        self.assertTrue(torch.allclose(response_1.logprobs, logprobs_1_response, rtol=0.5))
        self.assertFalse(torch.allclose(response_1.logprobs, logprobs_2_response, rtol=0.5))
        self.assertTrue(torch.allclose(response_2.logprobs, logprobs_4_response, rtol=0.8))
        self.assertFalse(torch.allclose(response_2.logprobs, logprobs_3_response, rtol=0.8))


class TestAsyncAPIServer(RayUnittestBaseAysnc):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)
        self.model_wrapper_no_history = ModelWrapper(
            self.engines[0], engine_type="vllm", enable_history=False
        )

    async def test_api_async(self):
        await self.model_wrapper.prepare()
        await self.model_wrapper_no_history.prepare()
        openai_client = self.model_wrapper.get_openai_async_client()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        model_id = (await openai_client.models.list()).data[0].id
        response = await openai_client.chat.completions.create(
            model=model_id, messages=messages, n=1
        )
        self.assertEqual(1, len(response.choices))
        self.assertTrue(len(response.choices[0].message.content) > 0)
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=2,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(response.choices[0].logprobs is not None)
        self.assertEqual(0, len(response.choices[0].logprobs.content[2].top_logprobs))
        # here we check the 3rd token logprob, because the first two tokens (`<think>`,`\n` usually have zero logprob)
        self.assertTrue(response.choices[0].logprobs.content[2].logprob < 0)
        self.assertTrue(hasattr(response, "prompt_token_ids"))
        self.assertTrue(len(response.prompt_token_ids) > 0)
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(len(response.choices[0].token_ids) > 0)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 3)
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=4,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 4)
        for exp in exps:
            self.assertTrue(len(exp.tokens) > 0)
            self.assertTrue(len(exp.logprobs) > 0)
            self.assertTrue(exp.prompt_length + len(exp.logprobs) == len(exp.tokens))
        self.assertEqual(len(self.model_wrapper.extract_experience_from_history()), 0)
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].tokens) > 0)
        self.assertTrue(len(exps[0].logprobs) > 0)
        self.assertTrue(exps[0].prompt_length + len(exps[0].logprobs) == len(exps[0].tokens))
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            logprobs=False,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].logprobs) == 0)
        response = (
            await self.model_wrapper_no_history.get_openai_async_client().chat.completions.create(
                model=model_id, messages=messages, n=2
            )
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(len(response.choices[0].token_ids) > 0)
        with self.assertRaises(ValueError):
            self.model_wrapper_no_history.extract_experience_from_history()
        self.assertEqual(len(self.model_wrapper_no_history.history), 0)


class TestTokenizer(unittest.TestCase):
    def test_action_mask(self):
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

    def test_action_mask_with_tools(self):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to various tools. Use them when needed to help users.",
            },
            {"role": "user", "content": "What's the weather like in Beijing today?"},
            {
                "role": "assistant",
                "content": "Let me get the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Beijing", "unit": "celsius"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temperature": 22, "condition": "sunny", "humidity": 45}',
                "tool_call_id": "call_abc123",
            },
            {
                "role": "assistant",
                "content": "The weather in Beijing today is sunny with a temperature of 22°C and humidity at 45%. It's a pleasant day!",
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        tokenizer = AutoTokenizer.from_pretrained(get_model_path())
        token_ids, action_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            chat_template=CHAT_TEMPLATE,
        )
        token_ids_hf, action_mask_hf, prompt_length_hf = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            chat_template=CHAT_TEMPLATE,
        )
        self.assertEqual(token_ids.shape, token_ids_hf.shape)
        self.assertEqual(action_mask.shape, action_mask_hf.shape)
        self.assertTrue(torch.equal(token_ids, token_ids_hf))
        self.assertTrue(torch.equal(action_mask, action_mask_hf))
        self.assertEqual(prompt_length, prompt_length_hf)


@parameterized_class(
    ("enable_thinking", "reasoning_parser"),
    [
        (True, "deepseek_r1"),
        (False, None),
    ],
)
class TestAPIServerToolCall(RayUnittestBaseAysnc):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_api_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        # added for toolcalls
        self.config.explorer.rollout_model.enable_auto_tool_choice = True
        self.config.explorer.rollout_model.tool_call_parser = "hermes"
        self.config.explorer.rollout_model.enable_thinking = self.enable_thinking
        self.config.explorer.rollout_model.reasoning_parser = self.reasoning_parser

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)
        self.model_wrapper_no_history = ModelWrapper(
            self.engines[0], engine_type="vllm", enable_history=False
        )

    async def test_api_tool_calls(self):
        """Tests the full conversation flow of a tool call via the OpenAI API.
        Note: This test require a model that supports tool calls and thinking mode, e.g. Qwen3-1.7B.
        """
        import json
        import time

        await self.model_wrapper.prepare()
        await self.model_wrapper_no_history.prepare()
        tokenizer = AutoTokenizer.from_pretrained(get_api_model_path())
        print_debug("\n\n" + "=" * 30 + " Running test_api_tool_calls " + "=" * 30)
        start_time = time.time()

        # --- Step 0: Get OpenAI Client ---
        print_debug(f"[{time.time() - start_time:.2f}s] Getting OpenAI client...")
        openai_client = self.model_wrapper.get_openai_client()
        model_id = openai_client.models.list().data[0].id
        print_debug(
            f"[{time.time() - start_time:.2f}s] Successfully got client. Model ID: {model_id}"
        )

        # --- Step 1: Define Tools and Messages ---
        print_debug(f"[{time.time() - start_time:.2f}s] Defining tools and initial message...")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
        print_debug(
            f"[{time.time() - start_time:.2f}s] Initial user message: {messages[0]['content']}"
        )
        print_debug("-" * 80)

        # --- Step 2: First API Call (Expecting a tool call) ---
        print_debug(f"[{time.time() - start_time:.2f}s] Making first API call to the model...")
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                },  # default to True
            },
        )
        print_debug(f"[{time.time() - start_time:.2f}s] First API call completed.")

        # --- Step 3: Assert and Print the Tool Call Response ---
        print_debug(f"[{time.time() - start_time:.2f}s] Asserting response is a tool call...")
        self.assertEqual(len(response.choices), 1)
        choice = response.choices[0]
        print_debug(f"    > Finish Reason: {choice.finish_reason}")
        self.assertEqual(choice.finish_reason, "tool_calls")
        if self.enable_thinking:
            self.assertIsNotNone(choice.message.reasoning_content)
        self.assertIsNotNone(choice.message.tool_calls)
        self.assertEqual(len(choice.message.tool_calls), 1)

        tool_call = choice.message.tool_calls[0]
        print_debug(f"    > Tool Call ID: {tool_call.id}")
        print_debug(f"    > Function Name: {tool_call.function.name}")
        print_debug(f"    > Function Arguments: {tool_call.function.arguments}")
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function.name, "get_current_weather")
        self.assertIn("Boston", tool_call.function.arguments)
        print_debug(f"[{time.time() - start_time:.2f}s] Assertions for tool call passed.")
        print_debug("-" * 80)

        # --- Step 4: Check Experience History ---
        print_debug(f"[{time.time() - start_time:.2f}s] Checking experience history...")
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        # The response text in the experience should contain the tool call info
        print_debug(f"    > Recorded experience response_text: {exps[0].response_text}")
        print_debug(f"    > Recorded experience: {exps[0]}")
        print_debug(f"    > message: {choice.message}")

        exp = exps[0]
        print_debug("\n" + "-" * 15 + " Decoding Experience Tokens " + "-" * 15)

        full_decoded_text = tokenizer.decode(exp.tokens, skip_special_tokens=False)
        print_debug(
            f"    > Full Decoded Text ({len(exp.tokens)} tokens):\n---\n{full_decoded_text}\n---"
        )

        prompt_length = exp.prompt_length
        prompt_tokens = exp.tokens[:prompt_length]
        response_tokens = exp.tokens[prompt_length:]

        prompt_decoded_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        response_decoded_text = tokenizer.decode(response_tokens, skip_special_tokens=False)

        print_debug(
            f"    > Decoded Prompt Part ({len(prompt_tokens)} tokens):\n---\n{prompt_decoded_text}\n---"
        )
        print_debug(
            f"    > Decoded Response Part ({len(response_tokens)} tokens):\n---\n{response_decoded_text}\n---"
        )

        action_mask = getattr(exp, "action_mask", None)
        if action_mask is not None:
            print_debug(f"\n    > Action Mask (Length: {len(action_mask)}):")
            masked_tokens_info = []
            for i, token_id in enumerate(response_tokens):
                token_text = tokenizer.decode([token_id])
                mask_value = action_mask[i] if i < len(action_mask) else "N/A"
                masked_tokens_info.append(f"({repr(token_text)}, Mask: {mask_value})")

            print_debug("      " + " ".join(masked_tokens_info))

            self.assertTrue(
                abs(len(action_mask) - len(response_tokens)) <= 1,
                f"Length of action_mask ({len(action_mask)}) does not match "
                f"length of response_tokens ({len(response_tokens)})",
            )
        else:
            print_debug("    > Action Mask: Not found in experience.")

        print_debug("-" * 52 + "\n")

        # pass this part
        # self.assertIn("get_current_weather", exps[0].response_text)

        self.assertEqual(
            len(self.model_wrapper.extract_experience_from_history()), 0
        )  # Verify cleared
        print_debug(f"[{time.time() - start_time:.2f}s] Experience history check passed.")
        print_debug("-" * 80)

        # --- Step 5: Second API Call (Providing tool result) ---
        print_debug(
            f"[{time.time() - start_time:.2f}s] Preparing for the second API call with tool result..."
        )
        messages.append(response.choices[0].message)  # Add assistant's tool call message

        # Mock the result of our tool
        tool_response_content = json.dumps(
            {"location": "Boston", "temperature": "72", "unit": "fahrenheit"}
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response_content,
            }
        )
        print_debug(f"[{time.time() - start_time:.2f}s] Full message list for second call:")
        for msg in messages:
            print_debug(f"    - {msg}")

        print_debug(f"[{time.time() - start_time:.2f}s] Making second API call...")
        second_response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools,
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                },  # default to True
            },
        )
        print_debug(f"[{time.time() - start_time:.2f}s] Second API call completed.")

        # --- Step 6: Assert and Print the Final Response ---
        print_debug(
            f"[{time.time() - start_time:.2f}s] Asserting final natural language response..."
        )
        self.assertEqual(len(second_response.choices), 1)
        final_choice = second_response.choices[0]
        print_debug(f"    > Final Finish Reason: {final_choice.finish_reason}")
        print_debug(f"    > Final Message Content: {final_choice.message.content}")
        print_debug(f"    > Final Message: {final_choice.message}")
        self.assertEqual(final_choice.finish_reason, "stop")
        # self.assertIsNone(final_choice.message.tool_calls)
        self.assertEqual(final_choice.message.tool_calls, [])
        self.assertIsNotNone(final_choice.message.content)
        # Check if the model used the information from the tool response
        self.assertIn("72", final_choice.message.content)
        self.assertIn("Boston", final_choice.message.content)
        print_debug(f"[{time.time() - start_time:.2f}s] Assertions for final response passed.")
        print_debug("-" * 80)

        # --- Step 7: Check Final Experience History ---
        print_debug(f"[{time.time() - start_time:.2f}s] Checking final experience history...")
        final_exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(final_exps), 1)
        print_debug(f"    > Final recorded experience response_text: {final_exps[0].response_text}")
        self.assertEqual(final_exps[0].response_text, final_choice.message.content)
        print_debug(f"[{time.time() - start_time:.2f}s] Final experience history check passed.")

        exp = final_exps[0]
        print_debug("\n" + "-" * 15 + " Decoding Experience Tokens " + "-" * 15)

        full_decoded_text = tokenizer.decode(exp.tokens, skip_special_tokens=False)
        print_debug(
            f"    > Full Decoded Text ({len(exp.tokens)} tokens):\n---\n{full_decoded_text}\n---"
        )

        prompt_length = exp.prompt_length
        prompt_tokens = exp.tokens[:prompt_length]
        response_tokens = exp.tokens[prompt_length:]

        prompt_decoded_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        response_decoded_text = tokenizer.decode(response_tokens, skip_special_tokens=False)

        print_debug(
            f"    > Decoded Prompt Part ({len(prompt_tokens)} tokens):\n---\n{prompt_decoded_text}\n---"
        )
        print_debug(
            f"    > Decoded Response Part ({len(response_tokens)} tokens):\n---\n{response_decoded_text}\n---"
        )

        action_mask = getattr(exp, "action_mask", None)
        if action_mask is not None:
            print_debug(f"\n    > Action Mask (Length: {len(action_mask)}):")
            masked_tokens_info = []
            for i, token_id in enumerate(response_tokens):
                token_text = tokenizer.decode([token_id])
                mask_value = action_mask[i] if i < len(action_mask) else "N/A"
                masked_tokens_info.append(f"({repr(token_text)}, Mask: {mask_value})")

            print_debug("      " + " ".join(masked_tokens_info))

            self.assertTrue(
                abs(len(action_mask) - len(response_tokens)) <= 1,
                f"Length of action_mask ({len(action_mask)}) does not match "
                f"length of response_tokens ({len(response_tokens)})",
            )
        else:
            print_debug("    > Action Mask: Not found in experience.")

        total_time = time.time() - start_time
        print_debug(
            "\n" + "=" * 28 + f" test_api_tool_calls PASSED in {total_time:.2f}s " + "=" * 28 + "\n"
        )
