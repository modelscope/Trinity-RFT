import subprocess

from typing import Any
from dataclasses import asdict

from trinity.utils.log import get_logger
from trinity.common.workflows.envs.email_searcher.utils import search_emails_tool, read_email_tool

try:
    import agentscope
except ImportError as e:
    error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
    print(error_message)
    agentscope = None

from agentscope.agents import ReActAgentV2 
from agentscope.service import ServiceResponse, ServiceExecStatus

logger = get_logger(__name__)


class ReActAgent(ReActAgentV2):
    """
    A customized ReAct agent that overrides the generate_response method by return_final_answer
    Ref: https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L260
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.service_toolkit.add(self.search_emails)
        self.service_toolkit.add(self.read_email)

        self.message_id_list = []
        self.ever_read_message_ids = []

        self.register_hook(
            "post_reply",
            "as_clear_message_ids",
            self._clear_message_ids_hook,
        )

    def search_emails(
        self,
        inbox_address: str,
        query_date: str,
        keywords: list[str],
        **kwargs: Any,
    ):
        """
        Search the user's email inbox for emails that match the given keywords.

        Args:
            inbox_address: The user's email address.
            query_date: The date of the query in 'YYYY-MM-DD' format.
            keywords (list[str]): A list of keywords to search for in the user's email inbox.

        Returns:
            ServiceResponse:
                The status field indicates whether the tool call was successful.
                The content field contains a list of SearchResult objects with message_id and snippet. If no emails are found, it returns an empty list.
        """

        try:
            res = subprocess.run(
                search_emails_tool(inbox=inbox_address, sent_before=query_date, keywords=keywords),
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )

            self.message_id_list.extend([r.message_id for r in res])

            return ServiceResponse(
                status=ServiceExecStatus.SUCCESS,
                content=[asdict(r) for r in res],
            )
        except subprocess.TimeoutExpired:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=[],
            )

    def read_email(self, message_id: str, **kwargs: Any):
        """
        Read the content of an email from the user's email inbox. Returns the email content.
        Args:
            message_id (str): The unique identifier of the email to read.

        Returns:
            ServiceResponse:
                The status field indicates whether the tool call was successful.
                The content field contains the email content or an error message if the email is not found.
        """
        try:
            email_content = subprocess.run(
                read_email_tool(message_id),
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )

            self.ever_read_message_ids.append(message_id)

            if email_content is None:
                return ServiceResponse(
                    status=ServiceExecStatus.ERROR,
                    content={"error": "Email not found"},
                )
            else:
                return ServiceResponse(
                    status=ServiceExecStatus.SUCCESS,
                    content=email_content.model_dump(),
                )
        except subprocess.TimeoutExpired:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content={"error": "Timeout"},
            )

    def _clear_message_ids_hook(self, *args: Any, **kwargs: Any) -> None:
        """Clear the message ids after each reply."""
        self.message_ids = []
        self.ever_read_message_ids = []
