import json
import re
from typing import Dict

from agentscope.models import DashScopeChatWrapper, ModelResponse
from data_juicer.config import get_init_configs, prepare_side_configs
from jsonargparse import Namespace
from loguru import logger

from trinity.common.config import DataPipelineConfig
from trinity.data.core.dataset import RftDataset

from .default_ops import (
    DEFAULT_CLEANER,
    DEFAULT_HUMAN_ANNOTATOR,
    DEFAULT_OP_ARGS,
    DEFAULT_SYNTHESIZER,
)

CONFIG_PARSER_SYSTEM_PROMPT_TEMPLATE = """
*Role*: You are a text preprocessing configurator. Your task is to **analyze** the user’s natural language instruction and translate it into weighted filter configurations for data cleaning.

*Context*:
Available filters (grouped by dimension):

{op_list}

*Rules*:
1. Parse the user’s instruction to identify emphasis/dismissal of **specific filters**, **dimensions** ({dimensions}), or **general cleaning goals**.
2. Assign weights between 0 (disabled) and 1 (enabled) to each filter:
   - `1`: Explicitly mentioned, critical to the request.
   - `0.5`: Implied or partially relevant.
   - `0`: Contradicts the request or unmentioned.
3. Default all filters to `0` unless the instruction implies their relevance.
4. Omit filters/keys with weight=0 in final output.

**Step 1: Analysis Phase**
1. Interpret the user’s intent to assign **nuanced weights**:
   - `1.0`: Core requirement (e.g., "strictly remove X")
   - `0.6-0.9`: Strong emphasis but not absolute (e.g., "mainly focus on X")
   - `0.3-0.5`: Secondary consideration (e.g., "slightly adjust X")
   - `0.0-0.2`: Weak/indirect relevance (e.g., "don’t prioritize X")
2. Weight assignment must reflect **linguistic intensity** (adverbs, adjectives) and **contextual priority**.

**Step 2: JSON Output Rules**
- Weights can be any float between 0.0 and 1.0 (e.g., 0.7, 0.25)
- Include **only filters with weight > 0**
- Omit entire dimension keys (e.g., `"difficulty"`) if all child filters are 0
- Ensure valid JSON without placeholders

**Output Template**:
```
Analysis:
1. [Keyword/phrase]: Linked to [filter], weight [X] because [reason].
2. [General intent]: Mapped to [dimension], enabling [filters] due to [logic].
...

JSON Output:
```json
{{
  "dimension1": {{ /* only include filters with weight > 0 */ }},
  "dimension2": {{ /* only include filters with weight > 0 */ }},
  ...
}}
```

**Examples**:
*User Input*: "Remove texts with repetitive characters and non-English content"
*Analysis*:
1. "Repetitive characters" → `character_repetition_filter` (quality, 1.0)
2. "Non-English content" → `language_id_score_filter` (difficulty, 1.0)

*JSON Output*:
```json
{{
  "quality": {{"character_repetition_filter": 1.0}},
  "difficulty": {{"language_id_score_filter": 1.0}}
}}
```

*User Input*: "Prioritize high LLM-rated quality"
*Analysis*:
1. "High LLM-rated quality" → `llm_quality_score_filter` (quality, 1.0). Other quality filters omitted (not explicitly requested).

*JSON Output*:
```json
{{
  "quality": {{"llm_quality_score_filter": 1.0}}
}}
```

*User Input*: "Severely limit texts with repetitive words, moderately check special characters, and lightly consider language scores."
*Analysis*:
1. "Severely limit" → `word_repetition_filter` (quality, 0.95)
2. "Moderately check" → `special_characters_filter` (quality, 0.6)
3. "Lightly consider" → `language_id_score_filter` (difficulty, 0.3)

*JSON Output*:
```json
{{
  "quality": {{
    "word_repetition_filter": 0.95,
    "special_characters_filter": 0.6
  }},
  "difficulty": {{
    "language_id_score_filter": 0.3
  }}
}}
```


**Critical Instructions**:
- Always generate the `Analysis` section **before** JSON
- Never include weights ≤0 or empty keys
- If no filters match, return `{{}}`
- If the user asks for your advice, try to give your recommendation.
"""


class DataTaskParser:
    """Parser to convert RFT task configs to DJ configs

    Supports:
    1. Direct config mapping
    2. LLM-based config generation
    3. Config validation and normalization
    """

    def __init__(
        self,
        data_pipeline_config: DataPipelineConfig,
        llm_agent: DashScopeChatWrapper = None,
        dataset: RftDataset = None,
        validate_config: bool = True,
    ):
        """
        Initialization method.

        :param data_pipeline_config: All configs of specified data pipeline.
        :param llm_agent: The LLM agent for natural language parsing.
        :param dataset: The dataset to be processed.
        :param validate_config: If execute the config validation check.
        """
        self.config = data_pipeline_config
        self.llm_agent = llm_agent
        self.validate_config = validate_config
        # TODO: refer dataset to support natural language parsing.
        self.dataset = dataset

    def parse_to_dj_config(self, extra_op_args=None):
        """Convert RFT config to DJ config"""
        if self.config.dj_config_path is not None:
            dj_config = self._direct_mapping()
        elif self.config.dj_process_desc is not None and self.llm_agent is not None:
            logger.warning("Agent-based config parsing only works for Cleaners right now.")
            dj_config = self._agent_based_parsing(extra_op_args)
        else:
            dj_config = None

        hit_cleaner, hit_synthesizer, hit_human_annotator = self._check_types_of_processors(
            dj_config
        )

        return dj_config, hit_cleaner, hit_synthesizer, hit_human_annotator

    def _check_types_of_processors(self, dj_config):
        hit_cleaner, hit_synthesizer, hit_human_annotator = False, False, False
        for op in dj_config.process:
            op_name = list(op.keys())[0]
            if op_name in DEFAULT_CLEANER:
                hit_cleaner = True
            elif op_name in DEFAULT_SYNTHESIZER:
                hit_synthesizer = True
            elif op_name in DEFAULT_HUMAN_ANNOTATOR:
                hit_human_annotator = True
        return hit_cleaner, hit_synthesizer, hit_human_annotator

    def _update_common_op_args(self, dj_config: Namespace, extra_op_args: Dict) -> Namespace:
        """Update common op args for RFT project"""
        for op in dj_config.process:
            print(op)
            op_name = list(op.keys())[0]
            for key, val in extra_op_args.items():
                op[op_name][key] = val
            print(op)
        return dj_config

    def _direct_mapping(self) -> Namespace:
        """Direct mapping from RFT config to DJ config"""
        dj_config = prepare_side_configs(self.config.dj_config_path)
        dj_config = get_init_configs(dj_config)
        return dj_config

    def _agent_based_parsing(self, extra_op_args=None, try_num=3) -> Namespace:
        """Generate DJ config using LLM agent"""
        if not self.llm_agent:
            raise ValueError("LLM agent required for unstructured config parsing")

        dj_config = None
        for _ in range(try_num):
            try:
                messages = self._construct_parsing_prompt()
                response = self.llm_agent(messages=messages)
                dj_config = self._parse_llm_response(response, extra_op_args)
                if dj_config is not None:
                    break
            except Exception as e:
                logger.warning(f"Exception when parsing dj config from agent: {e}")

        return dj_config

    def _construct_parsing_prompt(self):
        """Construct prompt for LLM agent"""
        system_prompt_template = CONFIG_PARSER_SYSTEM_PROMPT_TEMPLATE
        cleaners = DEFAULT_CLEANER

        op_list_str = ""
        for dim in cleaners:
            op_list_str += f"- **{dim}**\n"
            for op in cleaners[dim]:
                op_list_str += f"   - `{op}`: {cleaners[dim][op]}\n"
            op_list_str += "\n"

        dims = "/".join(list(cleaners.keys()))

        system_prompt = system_prompt_template.format(
            op_list=op_list_str,
            dimensions=dims,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.config.dj_process_desc},
        ]

        return messages

    def _parse_llm_response(self, response: ModelResponse, extra_op_args=None):
        """Parse LLM response to dj config."""
        cleaners = DEFAULT_CLEANER
        other_op_args = DEFAULT_OP_ARGS

        dj_process = []

        def json_to_dj_config(parsed_json):
            for dim in set(parsed_json.keys()) & set(cleaners.keys()):
                for op_name in set(parsed_json[dim].keys()) & set(cleaners[dim].keys()):
                    dj_process.append({op_name: {}})

        json_match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                parsed_json = json.loads(json_str)
                json_to_dj_config(parsed_json)
            except Exception as e:
                logger.warning(f"JSON parsing error: {e}")
                return None
        else:
            logger.warning("JSON block not found.")
            return None

        if not dj_process:
            return None

        for op in dj_process:
            op_name = list(op.keys())[0]
            if op_name in other_op_args:
                op[op_name].update(other_op_args[op_name])
            if extra_op_args is not None:
                for key, val in extra_op_args.items():
                    op[op_name][key] = val
        dj_config = Namespace(process=dj_process)
        dj_config = get_init_configs(dj_config)

        return dj_config
