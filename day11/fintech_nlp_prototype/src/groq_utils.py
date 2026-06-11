from __future__ import annotations

import json
import os
import re

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency
    Groq = None


def groq_client():
    api_key = os.getenv("GROQ_API_KEY", "")
    if Groq is None or not api_key:
        return None
    return Groq(api_key=api_key)


def chat_completion(prompt: str, system: str = "", model: str | None = None) -> str:
    client = groq_client()
    if client is None:
        raise RuntimeError("Groq client is unavailable")

    from config import GROQ_MODEL

    response = client.chat.completions.create(
        model=model or GROQ_MODEL,
        messages=[
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))