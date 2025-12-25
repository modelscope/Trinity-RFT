import uuid

import httpx
import openai
import requests


class TrinityClient:
    def __init__(self, proxy_url: str):
        self.proxy_url = proxy_url
        self.openai_base_url = f"{self.proxy_url}/v1"
        self.feedback_url = f"{self.proxy_url}/feedback"
        self.task_id = uuid.uuid4().hex[:6]

    def alive(self) -> bool:
        try:
            response = requests.get(f"{self.proxy_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_openai_client(self) -> openai.OpenAI:
        client = openai.OpenAI(
            base_url=self.openai_base_url,
            api_key="EMPTY",
        )
        return client

    def get_openai_async_client(self) -> openai.AsyncOpenAI:
        client = openai.AsyncOpenAI(
            base_url=self.openai_base_url,
            api_key="EMPTY",
        )
        return client

    def feedback(self, reward: float, msg_ids: list[str], timeout: float = 10) -> dict:
        response = requests.post(
            self.feedback_url,
            json={"reward": reward, "msg_ids": msg_ids, "task_id": self.task_id},
            timeout=timeout,
        )
        return response.json()

    async def feedback_async(self, reward: float, msg_ids: list[str], timeout: float = 10) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.feedback_url,
                json={"reward": reward, "msg_ids": msg_ids, "task_id": self.task_id},
                timeout=timeout,
            )
            return response.json()

    def commit(self, timeout: float = 10) -> dict:
        response = requests.post(f"{self.proxy_url}/commit", timeout=timeout)
        return response.json()

    async def commit_async(self, timeout: float = 10) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.proxy_url}/commit", timeout=timeout)
            return response.json()

    def get_metrics(self, timeout: float = 5) -> dict:
        response = requests.get(f"{self.proxy_url}/metrics", timeout=timeout)
        return response.json()

    async def get_metrics_async(self, timeout: float = 5) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.proxy_url}/metrics", timeout=timeout)
            if response.status_code != 200:
                raise ValueError(f"Failed to get metrics: {response.text}")
            return response.json()
