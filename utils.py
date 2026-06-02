import os
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def call_llm(prompt: str, system: str = "", max_tokens: int = 1500) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await _client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=messages
    )
    return response.choices[0].message.content.strip()


async def call_llm_json(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await _client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=messages,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content.strip()
