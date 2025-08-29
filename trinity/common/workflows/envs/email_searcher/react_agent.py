from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from trinity.common.workflows.envs.email_searcher.utils import (
    read_email_tool,
    search_emails_tool,
)

BaseAgentClass = object
try:
    from agentscope.agents import ReActAgentV2

    BaseAgentClass = ReActAgentV2  # type: ignore[misc]
except ImportError as e:
    error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
    pass


class EmailSearchAgent(BaseAgentClass):
    """
    A customized ReAct agent with pre-defined tools for email search and reading.
    Ref: https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L260
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.service_toolkit.add(self.search_emails)
        self.service_toolkit.add(self.read_email)

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
        from agentscope.service import ServiceExecStatus, ServiceResponse

        try:
            next_day = (datetime.strptime(query_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            res = search_emails_tool(inbox=inbox_address, sent_before=next_day, keywords=keywords)

            self.message_id_list.extend([r.message_id for r in res])

            return ServiceResponse(
                status=ServiceExecStatus.SUCCESS,
                content=[asdict(r) for r in res],
            )
        except Exception:
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
        from agentscope.service import ServiceExecStatus, ServiceResponse

        try:
            email_content = read_email_tool(message_id)

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
        except Exception:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content={"error": "Timeout"},
            )
