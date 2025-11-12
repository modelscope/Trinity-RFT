from typing import Dict, Type

import openai
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from pydantic import BaseModel


class AgentScopeReActAgent:
    def __init__(
        self,
        openai_client: openai.AsyncOpenAI,
        model_name: str,
        system_prompt: str,
        generate_kwargs: dict,
        response_structure: Type[BaseModel],
        max_iters: int = 10,
        toolkit: Toolkit | None = None,
    ):
        """Initialize the AgentScope ReAct agent with specified tools and model.

        Args:
            openai_client (openai.AsyncOpenAI): An instance of AsyncOpenAI client.
            model_name (str): The name of the model to use.
            system_prompt (str): The system prompt for the agent.
            generate_kwargs (dict): Generation parameters for the model.
            response_structure (Type[BaseModel]): A Pydantic model defining the expected response structure.
        """
        # patch the OpenAIChatModel to use the openai_client provided by Trinity-RFT
        self.agent_model = OpenAIChatModel(
            api_key="EMPTY",
            model_name=model_name,
            generate_kwargs=generate_kwargs,
            stream=False,
        )
        self.agent_model.client = openai_client
        self.agent = ReActAgent(
            name="react_agent",
            sys_prompt=system_prompt,
            model=self.agent_model,
            formatter=OpenAIChatFormatter(),
            # we enable agentscope's meta tool to allow agent to call tools dynamically without pre-registration
            enable_meta_tool=True,
            toolkit=toolkit,
            max_iters=max_iters,
        )
        self.response_structure = response_structure

    async def reply(self, query: str) -> Dict:
        """Generate a response from the agent given a query.

        Args:
            query (str): The input query for the agent.

        Returns:
            Dict: The structured response.
        """

        response = await self.agent.reply(
            Msg("user", query, role="user"), structured_model=self.response_structure
        )
        return response.metadata or {}
