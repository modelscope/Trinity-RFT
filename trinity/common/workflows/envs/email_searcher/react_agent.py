import json
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


class EmailSearchAgent(AgentScopeReActAgent):
    """
    A customized ReAct agent with pre-defined tools for email search and reading.
    Ref: https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L260
    """

    def __init__(self, *args, **kwargs):
        self.message_id_list = []
        self.ever_read_message_ids = []
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
            ServiceResponse:
                The status field indicates whether the tool call was successful.
                The content field contains a list of SearchResult objects with message_id and snippet. If no emails are found, it returns an empty list.
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
            print(f"Error in tool: {e}")
            import traceback

            traceback.print_exc()
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: Failed to search emails.",
                    ),
                ],
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
            email_content = read_email_tool(message_id)

            self.ever_read_message_ids.append(message_id)

            if email_content is None:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text="Error: Email not found.",
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
