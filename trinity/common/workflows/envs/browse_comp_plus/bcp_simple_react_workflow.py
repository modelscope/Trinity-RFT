# -*- coding: utf-8 -*-
"""BrowseComp-Plus Simple Tool-based ReAct Workflow for Trinity-RFT.

This workflow implements a simpler approach using direct OpenAI API function calling
for web search and document retrieval tasks.

Key Features:
- Uses OpenAI function calling API with local searcher (BM25, etc.)
- Simple conversation loop with tool execution
- LLM-as-judge evaluation with JSON output format for reliable parsing
- Supports both search and get_document tools
"""

import json
import re
import sys
from typing import List, Optional

import openai

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


class SearchToolHandler:
    """
    Search tool handler that executes search and get_document tools locally.
    Handles tool definitions and execution for OpenAI function calling.
    """

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            except Exception:
                self.tokenizer = None

    def get_tool_definitions(self):
        """Get tool definitions in OpenAI function calling format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_document",
                        "description": self.searcher.get_document_description(),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "docid": {
                                    "type": "string",
                                    "description": "Document ID to retrieve",
                                }
                            },
                            "required": ["docid"],
                        },
                    },
                }
            )

        return tools

    def execute_tool(self, tool_name: str, arguments: dict):
        """Execute a tool locally."""
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str):
        """Execute search tool."""
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        """Execute get_document tool."""
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


@WORKFLOWS.register_module("bcp_simple_react_workflow")
class BCPSimpleToolReActWorkflow(Workflow):
    """
    Simple tool-based ReAct workflow for BrowseComp-Plus using direct OpenAI API.

    This workflow:
    - Uses OpenAI function calling API directly
    - Uses local searcher (BM25, etc.) for document retrieval
    - Simple conversation loop with tool execution
    - LLM-as-judge evaluation with JSON output format
    - Supports configurable search parameters and tool settings
    """

    can_reset: bool = True
    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        """Initialize the workflow.

        Args:
            task: Task object containing the query and ground truth answer
            model: The main model wrapper for agent inference
            auxiliary_models: List of auxiliary models, first one used as judge
        """
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

        # Get OpenAI async client from model
        self.openai_async_client = model.get_openai_async_client()
        self.model_name = self.openai_async_client.model_path

        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task."""
        self.workflow_args = task.workflow_args

        # Workflow configuration
        self.max_iterations = int(self.workflow_args.get("max_iterations", 100))
        self.max_model_tokens = int(self.workflow_args.get("max_model_tokens", 24000))
        self.top_k = int(self.workflow_args.get("top_k", 5))
        self.snippet_max_tokens = int(self.workflow_args.get("snippet_max_tokens", 512))
        self.include_get_document = bool(self.workflow_args.get("include_get_document", False))

        # Searcher configuration
        self.searcher_type = self.workflow_args.get("searcher_type", "bm25")
        self.index_path = self.workflow_args.get("index_path", "indexes/bm25")

        # Task information
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        # Initialize searcher and tool handler (lazy init in run_async)
        self.searcher = None
        self.tool_handler = None

    def _init_searcher(self):
        """Initialize searcher lazily."""
        if self.searcher is not None:
            return

        try:
            # Import searcher from BrowseComp-Plus
            # If you have BrowseComp-Plus in a custom location, set BROWSECOMP_PATH env variable
            # or add it to workflow_args as 'browsecomp_path'
            browsecomp_path = self.workflow_args.get("browsecomp_path")
            if browsecomp_path is None:
                import os

                browsecomp_path = os.environ.get("BROWSECOMP_PATH")

            if browsecomp_path and str(browsecomp_path) not in sys.path:
                sys.path.insert(0, str(browsecomp_path))

            from searcher.searchers import SearcherType

            # Get searcher class
            searcher_class = SearcherType.get_searcher_class(self.searcher_type)

            # Create a minimal args object for searcher
            class SearcherArgs:
                def __init__(self, index_path):
                    self.index_path = index_path
                    # Add other common args
                    self.hf_token = None
                    self.hf_home = None

            args = SearcherArgs(self.index_path)
            self.searcher = searcher_class(args)

            self.logger.info(f"Initialized {self.searcher_type} searcher from {self.index_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize searcher: {e}")
            raise RuntimeError(
                f"Failed to initialize searcher. Make sure BrowseComp-Plus is in your Python path "
                f"(set BROWSECOMP_PATH env variable or browsecomp_path in workflow_args) "
                f"and index exists at {self.index_path}"
            )

        # Initialize tool handler
        self.tool_handler = SearchToolHandler(
            searcher=self.searcher,
            snippet_max_tokens=self.snippet_max_tokens,
            k=self.top_k,
            include_get_document=self.include_get_document,
        )

    def _get_user_prompt(self, question: str) -> str:
        """Get user prompt for the search agent."""
        prompt = f"""You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner.

NOTE:
  - You should always call one tool at a time. Use short keyword as the query for the search tool call.
  - You should first provide your reasoning process(your chain of thought) before each tool call step.
  - When you stop calling tools, you should provide your final answer alone with the explanation for your final answer.

Question: {question}

Your final response should be in the following format:
Exact Answer: {{your succinct, final answer}}"""
        return prompt

    def _truncate_tool_results(self, messages: list, max_tool_result_length: int = 2000) -> list:
        """Truncate tool results to prevent context overflow.

        Args:
            messages: List of messages
            max_tool_result_length: Maximum length for each tool result in characters

        Returns:
            Modified messages list with truncated tool results
        """
        truncated_messages = []
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("content"):
                content = msg["content"]
                if len(content) > max_tool_result_length:
                    # Truncate and add indicator
                    msg = msg.copy()
                    msg["content"] = (
                        content[:max_tool_result_length]
                        + f"\n\n... (truncated from {len(content)} chars)"
                    )
            truncated_messages.append(msg)
        return truncated_messages

    async def _run_conversation_with_tools(
        self,
        messages: list,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> tuple[str, list, list]:
        """Run conversation loop with function calling.

        Args:
            messages: Initial messages
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            Tuple of (status, all_messages, normalized_results)
        """
        all_messages = messages.copy()
        normalized_results = []
        tools = self.tool_handler.get_tool_definitions()

        for iteration in range(self.max_iterations):
            # Truncate tool results to prevent context overflow
            truncated_messages = self._truncate_tool_results(
                all_messages, max_tool_result_length=2500
            )

            # Call the model
            try:
                response = await self.openai_async_client.chat.completions.create(
                    model=self.model_name,
                    messages=truncated_messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self.task.rollout_args.max_tokens or 4096,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                self.logger.error(f"API call failed at iteration {iteration}: {e}")
                return "error", all_messages, normalized_results

            message = response.choices[0].message

            # Convert message to dict for all_messages
            message_dict = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            all_messages.append(message_dict)

            # Check if there are tool calls
            if message.tool_calls:
                # Execute each tool call
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse tool arguments: {e}")
                        function_args = {}

                    # Execute the tool locally
                    try:
                        result = self.tool_handler.execute_tool(function_name, function_args)
                    except Exception as e:
                        error_msg = f"Error executing {function_name}: {str(e)}"
                        self.logger.error(error_msg)
                        result = json.dumps({"error": error_msg})

                    # Record the tool call
                    normalized_results.append(
                        {
                            "type": "tool_call",
                            "tool_name": function_name,
                            "arguments": function_args,
                            "output": result,
                        }
                    )

                    # Add tool response to messages
                    all_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

                # Continue to next iteration
                continue

            # No tool calls, check if we have a final response
            if message.content:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": message.content,
                    }
                )
                return "completed", all_messages, normalized_results
            else:
                return "incomplete", all_messages, normalized_results

        # Hit max iterations
        self.logger.warning(f"Conversation hit max iterations ({self.max_iterations})")
        return "incomplete", all_messages, normalized_results

    def _parse_judge_response(self, judge_response: str) -> dict:
        """Parse judge model response (expects JSON format).

        Returns:
            dict with keys: extracted_answer, ground_truth, reasoning, is_correct, parse_error
        """
        result = {
            "extracted_answer": None,
            "ground_truth": None,
            "reasoning": None,
            "is_correct": False,
            "parse_error": False,
        }

        if not judge_response or not judge_response.strip():
            result["parse_error"] = True
            result["reasoning"] = "Empty response from judge model"
            return result

        # Try to extract JSON from the response
        # Sometimes the model might add text before/after the JSON
        json_match = re.search(r"\{.*\}", judge_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = judge_response.strip()

        try:
            parsed = json.loads(json_str)

            # Validate required fields
            if "is_correct" not in parsed:
                result["parse_error"] = True
                result["reasoning"] = "Missing 'is_correct' field in JSON response"
                return result

            # Extract fields
            result["extracted_answer"] = parsed.get("extracted_answer")
            result["ground_truth"] = parsed.get("ground_truth")
            result["reasoning"] = parsed.get("reasoning", "")
            result["is_correct"] = bool(parsed.get("is_correct", False))

            # Additional validation
            if result["extracted_answer"] is None:
                # If no answer was extracted, it should be marked as incorrect
                result["is_correct"] = False

        except json.JSONDecodeError as e:
            result["parse_error"] = True
            result["reasoning"] = f"JSON parsing error: {str(e)}"

            # Fallback: try to extract is_correct with regex
            is_correct_match = re.search(
                r'"is_correct"\s*:\s*(true|false)', judge_response, re.IGNORECASE
            )
            if is_correct_match:
                result["is_correct"] = is_correct_match.group(1).lower() == "true"
                result["parse_error"] = False  # We got the essential info
                result["reasoning"] = "Partial parse via regex"

        except Exception as e:
            result["parse_error"] = True
            result["reasoning"] = f"Unexpected error: {str(e)}"

        return result

    async def judge_result(
        self, normalized_results: list, question: str, correct_answer: str, judge_model
    ) -> bool:
        """Judge whether the agent's answer is correct using LLM-as-judge.

        Uses JSON-based judge template for reliable parsing.

        Args:
            normalized_results: List of conversation results
            question: The original question
            correct_answer: The ground truth answer
            judge_model: The judge model (OpenAI async client)

        Returns:
            True if answer is correct, False otherwise
        """
        if not normalized_results:
            return False

        # Extract final answer from last output_text
        final_answer = ""
        for result in reversed(normalized_results):
            if result.get("type") == "output_text":
                final_answer = result.get("output", "")
                break

        if not final_answer:
            self.logger.warning("No final answer found in results")
            return False

        # Judge prompt with JSON output format
        judge_prompt = f"""You are an expert judge tasked with evaluating whether an agent's response to a question is correct.

**Question**:
<question>
{question}
</question>

**Ground Truth Answer**:
<correct_answer>
{correct_answer}
</correct_answer>

**Agent's Response**:
<response>
{final_answer}
</response>

Your task is to determine if the agent's response is correct based on the ground truth answer. Be strict and precise in your judgment.

**Evaluation Criteria**:
1. Extract the final answer from the agent's response
2. Compare it with the ground truth answer
3. The agent's answer is correct ONLY if it is semantically equivalent to the ground truth
4. Allow for minor variations in phrasing, but the core information must match exactly
5. For numerical answers, allow small rounding differences (within 1% or 0.1 units)
6. If the agent's response contains additional correct information beyond the ground truth, it can still be marked as correct
7. If the agent's response is ambiguous, contradictory, or contains incorrect information, mark it as incorrect
8. If the agent did not provide a clear final answer, mark it as incorrect

**Output Format**: You MUST respond with a valid JSON object in the following format (no additional text):

{{
  "extracted_answer": "The exact answer extracted from the agent's response, or null if no answer was provided",
  "ground_truth": "{correct_answer}",
  "reasoning": "Brief explanation of why the extracted answer is correct or incorrect compared to the ground truth. If no answer was provided in the agent's response, mark it as incorrect.",
  "is_correct": true or false
}}

Respond ONLY with the JSON object, no additional text before or after."""

        # Call judge model asynchronously with lower temperature for more consistent JSON
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise evaluator. Always respond with valid JSON format as requested.",
                },
                {"role": "user", "content": judge_prompt},
            ]
            completion = await judge_model.chat.completions.create(
                model=judge_model.model_path,
                messages=messages,
                stream=False,
                temperature=0.3,  # Lower temperature for more consistent JSON output
            )
            judge_output = completion.choices[0].message.content

            self.logger.info(
                f"[judge_result] Question:\n{question}\n\n"
                f"[judge_result] Response:\n{final_answer}\n\n"
                f"[judge_result] Ground truth:\n{correct_answer}\n\n"
                f"[judge_result] Judge output:\n{judge_output}"
            )

            # Parse judge response using JSON parser
            judge_result = self._parse_judge_response(judge_output)

            if judge_result.get("parse_error"):
                self.logger.warning(
                    f"Could not parse judge response. Treating as incorrect. "
                    f"Parse error: {judge_result.get('reasoning')}"
                )
                return False

            return judge_result.get("is_correct", False)

        except Exception as e:
            self.logger.error(f"Error in judge_result: {e}")
            return False

    async def run_async(self):
        """Run the agent asynchronously to generate experiences.

        Returns:
            List of Experience objects for RL training
        """
        # Initialize searcher and tool handler
        self._init_searcher()

        # Prepare initial messages with user prompt
        user_prompt = self._get_user_prompt(self.task_desc)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful search assistant. Use the available search tools to find relevant information and provide comprehensive answers.",
            },
            {"role": "user", "content": user_prompt},
        ]

        # Run conversation with tools
        status, all_messages, normalized_results = await self._run_conversation_with_tools(
            messages,
            temperature=self.task.rollout_args.temperature,
            top_p=getattr(self.task.rollout_args, "top_p", 0.8),
        )

        self.logger.info(f"Conversation status: {status}")

        # Calculate reward using judge model
        reward = 0
        try:
            judge_model = self.auxiliary_models[0] if self.auxiliary_models else None
            if judge_model is None:
                self.logger.warning(
                    "No judge model provided. Using reward=0. "
                    "Please provide a judge model in auxiliary_models."
                )
            else:
                is_correct = await self.judge_result(
                    normalized_results, self.task_desc, self.truth, judge_model
                )
                reward = 1 if is_correct else 0
        except Exception as e:
            self.logger.error(f"Error in judge_model judging: {e}")
            reward = 0

        self.logger.info(f"Task result - Reward: {reward}")

        # Extract experiences from model history
        experiences = self.model.extract_experience_from_history(clear_history=True)
        return_experiences = []

        self.logger.debug(f"Experiences extracted: {len(experiences)}")

        # Count tool calls for metrics
        tool_call_counts = {}
        for result in normalized_results:
            if result.get("type") == "tool_call":
                tool_name = result.get("tool_name", "unknown")
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

        # Process and filter experiences
        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.reward = reward

            # Add agent-specific metrics (only numeric values that can be averaged)
            agent_metrics = {
                "conversation_turns": len(all_messages),
                "total_tool_calls": sum(tool_call_counts.values()),
                "search_calls": tool_call_counts.get("search", 0),
                "get_document_calls": tool_call_counts.get("get_document", 0),
                "accuracy": reward,
            }

            if experience.metrics is None:
                experience.metrics = {}
            experience.metrics.update(agent_metrics)

            # Filter out experiences that are too long
            if len(experience.tokens) > self.max_model_tokens:
                self.logger.debug(
                    f"Skipping experience {i} due to length: "
                    f"{len(experience.tokens)} > {self.max_model_tokens}"
                )
                continue

            return_experiences.append(experience)

        if return_experiences:
            self.logger.info(
                f"Returning {len(return_experiences)} experiences, "
                f"run_id: {str(return_experiences[-1].eid.run)}, "
                f"final reward: {return_experiences[-1].reward}"
            )
        else:
            self.logger.warning("No valid experiences to return (all filtered out).")

        return return_experiences
