import json
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, Union

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse
from pydantic import BaseModel, Field

from trinity.common.rewards import MathBoxedRewardFn, RewardFn
from trinity.common.workflows.envs.email_searcher.utils import (
    FinalRubric,
    QueryModel,
    judge_correctness,
    read_email_tool,
    search_emails_tool,
)
from trinity.common.workflows.workflow import Task
from trinity.utils.log import get_logger

# For GSM8K task
GSM8KSystemPrompt = """You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}."""


# For Email Search task
def get_email_search_system_prompt(
    task: Task,
) -> str:
    meta_prompt = """You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {max_turns} turns to find the answer, so if your first seach doesn't find the answer, you can try with different keywords.
    Always describe what you see and plan your next steps clearly. When taking actions, explain what you're doing and why. When the answer to the task is found, call `generate_response` to finish the process. Only call `generate_response` when answer is found. You should not respond any next steps in `generate_response`. Complete all steps and then call `generate_response`.

    User's email address is {inbox_address}
    Today's date is {query_date}
    """
    query = QueryModel.model_validate(task.raw_task)
    workflow_args = task.workflow_args
    max_turns = int(workflow_args.get("max_turns", 10))
    tool_obs_char_limit = int(workflow_args.get("tool_obs_char_limit", 2000))

    return meta_prompt.format(
        max_turns=max_turns,
        inbox_address=query.inbox_address,
        query_date=query.query_date,
    )


class GSM8KResponseStructure(BaseModel):
    result: str = Field(
        description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
    )


class EmailSearchResponseStructure(BaseModel):
    answer: str = Field(
        description="It should be called with the answer and the sources. If you cannot find the answer, you should return 'I don't know' with an empty list of sources."
    )
    sources: List[str] = Field(
        description="a list of message ids that are relevant to the query. Usually there will be only one. If you cannot find the answer, you should return an empty list."
    )


class GSM8KRewardFn(MathBoxedRewardFn):
    def __call__(  # type: ignore [override]
        self,
        response: dict,
        truth: str,
        format_score_coef: float = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        # parse GSM8K truth
        if isinstance(truth, str) and "####" in truth:
            truth = truth.split("####")[1].strip()
        else:
            truth = str(truth)
        return super().__call__(
            response=response.get("result", ""),
            truth=truth,
            with_think=False,
            format_score_coef=format_score_coef,
            **kwargs,
        )


class EmailSearchRewardFn(RewardFn):
    def __init__(self, *args, **kwargs):
        self.llm_as_a_judge = kwargs.pop("llm_as_a_judge", True)
        super().__init__(*args, **kwargs)
        self.logger = get_logger(__name__)

    def __call__(  # type: ignore [override]
        self,
        response: dict,
        truth: str,
        format_score_coef: float = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        """Ref: calculate_reward in https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L64"""
        answer_and_sources = response
        num_turns = kwargs.get("num_turns")
        query: QueryModel = kwargs.get("query")
        max_turns = kwargs.get("max_turns", 10)
        message_id_list = kwargs.get("message_id_list", [])
        ever_read_message_ids = kwargs.get("ever_read_message_ids", [])
        auxiliary_models = kwargs.get("auxiliary_models", None)

        try:
            answer = answer_and_sources.get("answer", None)
            sources = answer_and_sources.get("sources", [])
        except Exception as e:
            self.logger.error(f"Error extracting answer and sources: {e}")
            result = {"accuracy": 0.0, "format": -1.0}
            return result

        if answer is None:
            result = {"accuracy": 0.0, "format": -1.0}
            return result

        if not self.llm_as_a_judge:
            result = {"accuracy": float(answer.lower() in truth.lower()), "format": 0.0}
            return result

        rubric = FinalRubric()
        rubric.attempted_answer = answer is not None and answer != ""
        rubric.returned_i_dont_know = answer == "I don't know"
        if len(query.message_ids) > 0:
            rubric.ever_found_right_email = query.message_ids[0] in message_id_list
            rubric.ever_read_right_email = query.message_ids[0] in ever_read_message_ids
            rubric.sources_correct = query.message_ids[0] in sources
        rubric.num_sources = len(sources)
        rubric.num_turns = num_turns
        self.logger.debug(f"Rubric: {rubric.model_dump()}")

        try:
            judge_model = auxiliary_models[0] if auxiliary_models else None
            judge_response = judge_correctness(answer, query, judge_model)
            rubric.answer_correct = judge_response

        except Exception as e:
            self.logger.error(f"Error judging correctness: {e}")
            rubric.answer_correct = False

        # Note: make sure all possible partial rewards always sum to less than 0.5.
        partial_rewards = 0
        partial_rewards += 0.1 if rubric.ever_found_right_email else 0
        partial_rewards += 0.1 if rubric.ever_read_right_email else 0
        partial_rewards += 0.1 if rubric.sources_correct else 0

        # No formatting error, but wrong answer: reward will be -1 to 0
        if rubric.attempted_answer and not rubric.answer_correct:
            result = {"accuracy": -1.0, "format": partial_rewards}
            return result

        # Returned no answer at all: reward will be 0 to 1
        if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
            result = {"accuracy": 0.0, "format": partial_rewards}
            return result

        # Answer is correct: reward will be 1 to 2
        if rubric.answer_correct:
            # Partial credit calculation is different for correct answers.

            reward = 1
            reward += 0.3 if rubric.sources_correct else 0

            # Extra credit for not including extra sources.
            reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0

            # Extra credit for being faster (taking fewer turns).
            reward += 0.1 * (1 - rubric.num_turns / max_turns)
            result = {"accuracy": 1.0, "format": reward}
            return result

        self.logger.error(f"Rubric {rubric} not handled properly")
        raise ValueError("Rubric is not handled properly")


class ToolkitManager:
    def __init__(self, *args, **kwargs):
        self.toolkit = Toolkit()

    def get_status(self):
        return {}


class EmailSearchToolkitManager(ToolkitManager):
    """
    A customized ReAct agent with pre-defined tools for email search and reading.
    Ref: https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L260
    """

    def __init__(self, *args, **kwargs):
        self.logger = get_logger(__name__)
        task = kwargs.get("task")
        self.query = QueryModel.model_validate(task.raw_task)
        self.workflow_args = task.workflow_args
        self.max_turns = int(self.workflow_args.get("max_turns", 10))
        self.tool_obs_char_limit = int(self.workflow_args.get("tool_obs_char_limit", 2000))
        super().__init__(*args, **kwargs)
        self.message_id_list = []  # List to store message IDs found during search
        self.ever_read_message_ids = []  # List to store message IDs that have been read
        self.toolkit.register_tool_function(self.search_emails)
        self.toolkit.register_tool_function(self.read_email)

    def get_status(self):
        return {
            "query": self.query,
            "max_turns": self.max_turns,
            "tool_obs_char_limit": self.tool_obs_char_limit,
            "message_id_list": self.message_id_list,
            "ever_read_message_ids": self.ever_read_message_ids,
        }

    def search_emails(
        self,
        inbox_address: str,
        query_date: str,
        keywords: list[str],
        **kwargs: Any,
    ) -> ToolResponse:
        """
        Search the user's email inbox for emails that match the given keywords.

        Args:
            inbox_address: The user's email address.
            query_date: The date of the query in 'YYYY-MM-DD' format.
            keywords (list[str]): A list of keywords to search for in the user's email inbox.

        Returns:
            ToolResponse:
                A ToolResponse object containing a list of TextBlock objects in the `content` field.
                On success, the text field of the TextBlock contains a JSON string representing
                a list of email summaries (e.g., message_id, snippet) matching
                the search criteria. Each email summary is converted to a dictionary via `asdict`.
                On failure, the text indicates an error message.
        """

        try:
            next_day = (datetime.strptime(query_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            res = search_emails_tool(inbox=inbox_address, sent_before=next_day, keywords=keywords)

            self.message_id_list.extend([r.message_id for r in res])

            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=json.dumps([asdict(r) for r in res]),
                    ),
                ],
            )
        except Exception as e:
            self.logger.info(f"Error in tool: {e}, traceback: {traceback.format_exc()}")
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"Error: Failed to search emails.\nError message: {e}",
                    ),
                ],
            )

    def read_email(self, message_id: str, **kwargs: Any) -> ToolResponse:
        """
        Read the content of an email from the user's email inbox. Returns the email content.
        Args:
            message_id (str): The unique identifier of the email to read.

        Returns:
            ToolResponse:
                A ToolResponse object containing the email content or an error message if the email is not found.
        """

        try:
            email_content = read_email_tool(message_id)

            self.ever_read_message_ids.append(message_id)

            if email_content is None:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=f"Error: Email (message_id = {message_id}) not found.",
                        ),
                    ],
                )
            else:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=json.dumps(email_content.model_dump()),
                        ),
                    ],
                )
        except Exception:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: Timeout to read email.",
                    ),
                ],
            )


# Registry for different templates


@dataclass
class Template:
    """A template for different task types, including system prompt and response structure."""

    system_prompt: Union[str, Callable]
    response_structure: Type[BaseModel]
    reward_fn_cls: Type[RewardFn]
    toolkit_manager: Type[ToolkitManager] = ToolkitManager


TEMPLATE_MAP: Dict[str, Optional[Template]] = {
    "gsm8k": Template(
        system_prompt=GSM8KSystemPrompt,
        response_structure=GSM8KResponseStructure,
        reward_fn_cls=GSM8KRewardFn,
    ),
    "email_search": Template(
        system_prompt=get_email_search_system_prompt,
        response_structure=EmailSearchResponseStructure,
        reward_fn_cls=EmailSearchRewardFn,
        toolkit_manager=EmailSearchToolkitManager,
    ),
    # Add more templates for different task types as needed
}
