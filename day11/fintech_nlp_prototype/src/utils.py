from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_parent_dir(path: str | Path) -> None:
	Path(path).parent.mkdir(parents=True, exist_ok=True)


def split_sentences(text: str) -> list[str]:
	sentences = re.split(r"(?<=[.!?])\s+", text.strip())
	return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize_tokens(text: str) -> set[str]:
	return {
		token
		for token in re.findall(r"[a-z0-9']+", text.lower())
		if len(token) > 2
	}


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
	text = text.strip()
	if not text:
		return []

	chunks: list[str] = []
	for start in range(0, len(text), chunk_size):
		chunk = text[start:start + chunk_size].strip()
		if chunk:
			chunks.append(chunk)
	return chunks
