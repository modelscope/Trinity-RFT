import json
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from trinity.common.workflows.agentscope.react.react_agent import AgentScopeReActAgent
from trinity.common.workflows.envs.email_searcher.utils import (
    read_email_tool,
    search_emails_tool,
)
from trinity.utils.log import get_logger


class EmailSearchAgent(AgentScopeReActAgent):
    """
    A customized ReAct agent with pre-defined tools for email search and reading.
    Ref: https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L260
    """

    def __init__(self, *args, **kwargs):
        self.logger = get_logger(__name__)
        self.message_id_list = []  # List to store message IDs found during search
        self.ever_read_message_ids = []  # List to store message IDs that have been read
        toolkit = Toolkit()
        toolkit.register_tool_function(self.search_emails)
        toolkit.register_tool_function(self.read_email)
        super().__init__(*args, toolkit=toolkit, **kwargs)

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
            self.logger.info(f"Error in search_emails: {e}, traceback: {traceback.format_exc()}")
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
        except Exception as e:
            self.logger.info(f"Error in read_email: {e}, traceback: {traceback.format_exc()}")
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"Error: Failed to read email.\nError message: {e}",
                    ),
                ],
            )
