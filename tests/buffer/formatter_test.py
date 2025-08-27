import unittest

from transformers import AutoTokenizer

from tests.tools import get_model_path
from trinity.buffer.schema.formatter import (
    DPOMessagesFormatter,
    DPOPlaintextFormatter,
    SFTMessagesFormatter,
    SFTPlaintextFormatter,
)
from trinity.common.config import FormatConfig
from trinity.common.constants import PromptType
from trinity.common.experience import Experience


class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(get_model_path())

    def test_sft_messages_formatter(self):
        config = FormatConfig(
            prompt_type=PromptType.MESSAGES,
            messages_key="message_list",
        )
        formatter = SFTMessagesFormatter(tokenizer=self.tokenizer, format_config=config)
        sample = {
            "message_list": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }

        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        sequence = self.tokenizer.decode(exp.tokens)

        self.assertIn("Hi", sequence)
        self.assertIn("Hello", sequence)

        # test tool
        config = FormatConfig(
            prompt_type=PromptType.MESSAGES,
            messages_key="messages",
            tools_key="tools",
        )
        formatter = SFTMessagesFormatter(tokenizer=self.tokenizer, format_config=config)
        sample = {
            "messages": [
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
            ],
            "tools": [
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
            ],
        }
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        sequence = self.tokenizer.decode(exp.tokens)
        self.assertIn("What's the weather like in Beijing today?", sequence)
        self.assertIn(
            "The weather in Beijing today is sunny with a temperature of 22°C and humidity at 45%. It's a pleasant day!",
            sequence,
        )
        self.assertIn("get_weather", sequence)

    def test_sft_plaintext_formatter(self):
        # with system prompt key
        config = FormatConfig(
            prompt_type=PromptType.PLAINTEXT,
            system_prompt_key="system",
            system_prompt="You are a programmer.",  # has lower priority than system_prompt_key
            prompt_key="prompt",
            response_key="response",
        )
        formatter = SFTPlaintextFormatter(tokenizer=self.tokenizer, format_config=config)
        sample = {
            "system": "You are a helpful assistant.",
            "prompt": "What is 2+2?",
            "response": "2+2=4",
        }
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        # detokenize exp.tokens into text
        sequence = self.tokenizer.decode(exp.tokens)
        self.assertIn("You are a helpful assistant.", sequence)
        self.assertIn("What is 2+2?", sequence)
        self.assertIn("2+2=4", sequence)

        # with system prompt
        config = FormatConfig(
            prompt_type=PromptType.PLAINTEXT,
            system_prompt="You are a programmer.",
            prompt_key="prompt",
            response_key="response",
        )
        formatter = SFTPlaintextFormatter(tokenizer=self.tokenizer, format_config=config)

        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        # detokenize exp.tokens into text
        sequence = self.tokenizer.decode(exp.tokens)
        self.assertIn("You are a programmer.", sequence)
        self.assertIn("What is 2+2?", sequence)
        self.assertIn("2+2=4", sequence)

    def test_dpo_plaintext_formatter(self):
        config = FormatConfig(
            prompt_type=PromptType.PLAINTEXT,
            prompt_key="prompt",
            chosen_key="chosen",
            rejected_key="rejected",
        )
        formatter = DPOPlaintextFormatter(tokenizer=self.tokenizer, format_config=config)
        sample = {"prompt": "What is 2+2?", "chosen": "2+2=4", "rejected": "2+2=5"}
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.chosen)
        self.assertIsNotNone(exp.rejected)
        self.assertIsNotNone(exp.prompt_length)
        prompt = self.tokenizer.decode(exp.tokens)
        chosen = self.tokenizer.decode(exp.chosen)
        rejected = self.tokenizer.decode(exp.rejected)
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("2+2=4", chosen)
        self.assertIn("2+2=5", rejected)
        self.assertNotIn("What is 2+2?", chosen)
        self.assertNotIn("What is 2+2?", rejected)
        self.assertNotIn("2+2=4", prompt)
        self.assertNotIn("2+2=5", prompt)

    def test_dpo_messages_formatter(self):
        config = FormatConfig(
            prompt_type=PromptType.MESSAGES,
            messages_key="messages",
            chosen_key="chosen",
            rejected_key="rejected",
        )
        formatter = DPOMessagesFormatter(tokenizer=self.tokenizer, format_config=config)
        sample = {
            "messages": [
                {"role": "user", "content": "What is your name?"},
            ],
            "chosen": [
                {"role": "assistant", "content": "My name is Assistant."},
            ],
            "rejected": [{"role": "assistant", "content": "I don't have a favorite color."}],
        }
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        # detokenize exp.tokens into text
        prompt = self.tokenizer.decode(exp.tokens)
        chosen = self.tokenizer.decode(exp.chosen)
        rejected = self.tokenizer.decode(exp.rejected)
        self.assertIn("What is your name?", prompt)
        self.assertIn("My name is Assistant.", chosen)
        self.assertIn("I don't have a favorite color.", rejected)
