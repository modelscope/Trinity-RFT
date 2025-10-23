from functools import partial

import httpx
import openai
import requests


class ExplorerClient:
    def __init__(self, explorer_api_url: str):
        self.explorer_api_url = explorer_api_url
        self.openai_base_url = f"{self.explorer_api_url}/v1"
        self.feedback_url = f"{self.explorer_api_url}/feedback"
        self.session_id = self.init_session()

    def init_session(self) -> str:
        response = requests.get(f"{self.explorer_api_url}/allocate")
        data = response.json()
        return data["session_id"]

    def get_openai_client(self) -> openai.OpenAI:
        client = openai.OpenAI(
            base_url=self.openai_base_url,
            api_key="EMPTY",
        )
        client.chat.completions.create = partial(
            client.chat.completions.create, extra_body={"session_id": self.session_id}
        )
        return client

    def get_openai_async_client(self) -> openai.AsyncOpenAI:
        client = openai.AsyncOpenAI(
            base_url=self.openai_base_url,
            api_key="EMPTY",
        )
        client.chat.completions.create = partial(
            client.chat.completions.create, extra_body={"session_id": self.session_id}
        )
        return client

    def feedback(self, reward: float) -> dict:
        response = requests.post(
            self.feedback_url, json={"session_id": self.session_id, "reward": reward}
        )
        return response.json()

    async def feedback_async(self, reward: float) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.feedback_url, json={"session_id": self.session_id, "reward": reward}
            )
            return response.json()
