from functools import partial

import httpx
import openai
import requests


class ExplorerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session_id = self.init_session()

    def init_session(self) -> str:
        response = requests.post(f"{self.base_url}/allocate")
        data = response.json()
        return data["session_id"]

    def get_openai_client(self) -> openai.OpenAI:
        client = openai.OpenAI(
            base_url=self.base_url + "/v1",
            api_key="EMPTY",
        )
        client.chat.completions.create = partial(
            client.chat.completions.create, extra_body={"session_id": self.session_id}
        )
        return client

    def get_openai_async_client(self) -> openai.AsyncOpenAI:
        client = openai.AsyncOpenAI(
            base_url=self.base_url + "/v1",
            api_key="EMPTY",
        )
        client.chat.completions.create = partial(
            client.chat.completions.create, extra_body={"session_id": self.session_id}
        )
        return client

    def feedback(self, reward: float):
        response = requests.post(
            f"{self.base_url}/feedback", json={"session_id": self.session_id, "reward": reward}
        )
        return response.json()

    async def feedback_async(self, reward: float):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/feedback", json={"session_id": self.session_id, "reward": reward}
            )
            return response.json()
