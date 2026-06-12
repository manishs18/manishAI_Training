from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re


@dataclass
class ChunkScore:
    chunk: str
    score: int


def load_text(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def split_into_chunks(text: str, chunk_size: int = 220) -> List[str]:
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_size):
        chunk_words = words[start:start + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_chunk(question_tokens: List[str], chunk: str) -> int:
    chunk_tokens = set(tokenize(chunk))
    return sum(1 for token in question_tokens if token in chunk_tokens)


def retrieve_chunks(question: str, chunks: List[str], top_k: int = 3) -> List[ChunkScore]:
    question_tokens = tokenize(question)
    scored = [ChunkScore(chunk=chunk, score=score_chunk(question_tokens, chunk)) for chunk in chunks]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]
